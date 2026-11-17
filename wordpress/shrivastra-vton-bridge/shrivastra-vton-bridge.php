<?php
/**
 * Plugin Name: ShriVastra VTON API Bridge
 * Description: Authenticated, server-side bridge to FASHN VTON inference jobs.
 * Version: 0.1.0
 * Requires PHP: 8.0
 */
if (!defined('ABSPATH')) { exit; }

/**
 * In wp-config.php, define SHRIVASTRA_VTON_API_URL (an HTTPS origin)
 * and SHRIVASTRA_VTON_API_TOKEN (secret). Do not expose either in JS.
 *
 * Browser requests must use WP REST cookie authentication and X-WP-Nonce.
 */
add_action('rest_api_init', function () {
    register_rest_route('shrivastra-vton/v1', '/try-on', array(
        'methods' => 'POST',
        'permission_callback' => 'is_user_logged_in',
        'callback' => 'shrivastra_vton_submit_job',
    ));
    register_rest_route('shrivastra-vton/v1', '/jobs/(?P<job_id>[a-f0-9-]{36})', array(
        'methods' => 'GET',
        'permission_callback' => 'is_user_logged_in',
        'callback' => 'shrivastra_vton_get_job',
    ));
});

function shrivastra_vton_config() {
    if (!defined('SHRIVASTRA_VTON_API_URL') || !defined('SHRIVASTRA_VTON_API_TOKEN')) {
        return new WP_Error('vton_config', 'VTON API is not configured.', array('status' => 503));
    }
    $url = rtrim(SHRIVASTRA_VTON_API_URL, '/');
    $parsed = wp_parse_url($url);
    if (!$parsed || ($parsed['scheme'] ?? '') !== 'https' || empty($parsed['host'])
        || !empty($parsed['user']) || !empty($parsed['pass'])
        || !empty($parsed['query']) || !empty($parsed['fragment'])) {
        return new WP_Error('vton_config', 'A valid HTTPS API origin is required.', array('status' => 503));
    }
    if (!is_string(SHRIVASTRA_VTON_API_TOKEN) || strlen(SHRIVASTRA_VTON_API_TOKEN) < 24) {
        return new WP_Error('vton_config', 'VTON API token is not configured.', array('status' => 503));
    }
    return array($url, SHRIVASTRA_VTON_API_TOKEN);
}

function shrivastra_vton_validate_image($file) {
    if (!is_array($file) || !isset($file['tmp_name'], $file['error'], $file['size'])
        || $file['error'] !== UPLOAD_ERR_OK || !is_uploaded_file($file['tmp_name'])) {
        return new WP_Error('vton_upload', 'Invalid uploaded image.', array('status' => 422));
    }
    if ($file['size'] <= 0 || $file['size'] > 10 * 1024 * 1024) {
        return new WP_Error('vton_upload', 'Image exceeds 10 MiB.', array('status' => 413));
    }
    $info = @getimagesize($file['tmp_name']);
    if (!$info || !in_array($info['mime'], array('image/jpeg', 'image/png', 'image/webp'), true)
        || ($info[0] * $info[1]) > 20000000) {
        return new WP_Error('vton_upload', 'Only JPEG, PNG or WebP under 20 MP is supported.', array('status' => 422));
    }
    return $info['mime'];
}

function shrivastra_vton_submit_job(WP_REST_Request $request) {
    $config = shrivastra_vton_config();
    if (is_wp_error($config)) { return $config; }
    if (!function_exists('curl_init')) {
        return new WP_Error('vton_curl', 'PHP cURL is required.', array('status' => 503));
    }
    $user_id = get_current_user_id();
    $rate_key = 'sv_vton_rate_' . $user_id;
    $requests = (int) get_transient($rate_key);
    if ($requests >= 5) {
        return new WP_Error('vton_rate', 'Please try again later.', array('status' => 429));
    }
    $files = $request->get_file_params();
    if (empty($files['person_image']) || (empty($files['garment_image']) &&
        empty($files['upper_garment_image']) && empty($files['lower_garment_image']))) {
        return new WP_Error('vton_fields', 'A person and at least one garment are required.', array('status' => 422));
    }
    if (!empty($files['garment_image']) &&
        (!empty($files['upper_garment_image']) || !empty($files['lower_garment_image']))) {
        return new WP_Error('vton_fields', 'Use single or separate garment fields.', array('status' => 422));
    }
    $post_fields = array();
    foreach (array('person_image', 'garment_image', 'upper_garment_image', 'lower_garment_image') as $key) {
        if (!empty($files[$key])) {
            $mime = shrivastra_vton_validate_image($files[$key]);
            if (is_wp_error($mime)) { return $mime; }
            $post_fields[$key] = new CURLFile($files[$key]['tmp_name'], $mime, $key);
        }
    }
    $category = sanitize_text_field($request->get_param('category') ?: 'tops');
    if (!in_array($category, array('tops', 'bottoms', 'one-pieces'), true)) {
        return new WP_Error('vton_category', 'Invalid garment category.', array('status' => 422));
    }
    $post_fields['category'] = $category;
    $ch = curl_init($config[0] . '/api/v1/jobs');
    curl_setopt_array($ch, array(
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $post_fields,
        CURLOPT_HTTPHEADER => array('Authorization: Bearer ' . $config[1]),
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_CONNECTTIMEOUT => 10,
        CURLOPT_TIMEOUT => 45,
        CURLOPT_FOLLOWLOCATION => false,
        CURLOPT_SSL_VERIFYPEER => true,
        CURLOPT_SSL_VERIFYHOST => 2,
    ));
    $body = curl_exec($ch);
    $status = (int) curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
    curl_close($ch);
    if ($body === false || $status !== 202) {
        return new WP_Error('vton_upstream', 'Try-on service unavailable.', array('status' => 502));
    }
    $result = json_decode($body, true);
    if (!is_array($result) || empty($result['job_id']) ||
        !preg_match('/^[a-f0-9-]{36}$/', $result['job_id'])) {
        return new WP_Error('vton_upstream', 'Invalid try-on job response.', array('status' => 502));
    }
    set_transient($rate_key, $requests + 1, MINUTE_IN_SECONDS);
    set_transient('sv_vton_owner_' . $result['job_id'], $user_id, DAY_IN_SECONDS);
    return rest_ensure_response(array('job_id' => $result['job_id'], 'status' => 'queued'));
}

function shrivastra_vton_get_job(WP_REST_Request $request) {
    $job_id = $request['job_id'];
    $owner = get_transient('sv_vton_owner_' . $job_id);
    if (!$owner || (int) $owner !== get_current_user_id()) {
        return new WP_Error('vton_not_found', 'Job not found.', array('status' => 404));
    }
    $config = shrivastra_vton_config();
    if (is_wp_error($config)) { return $config; }
    $response = wp_remote_get($config[0] . '/api/v1/jobs/' . rawurlencode($job_id), array(
        'timeout' => 30,
        'redirection' => 0,
        'headers' => array('Authorization' => 'Bearer ' . $config[1]),
    ));
    if (is_wp_error($response) || wp_remote_retrieve_response_code($response) !== 200) {
        return new WP_Error('vton_upstream', 'Could not fetch job status.', array('status' => 502));
    }
    $result = json_decode(wp_remote_retrieve_body($response), true);
    if (!is_array($result) || !isset($result['status'])) {
        return new WP_Error('vton_upstream', 'Invalid job response.', array('status' => 502));
    }
    return rest_ensure_response($result);
}
