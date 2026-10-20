export const LIMITS = {
    screen_tiltMin: -120, // Degrees
    screen_tiltMax: 120,
    screen_zoomMin: 1.0,  // Distance (positive)
    screen_zoomMax: 10.0,
    screen_eye_sepMin: -0.15,
    screen_eye_sepMax: 0.15,
    screen_txMax: 0.5,
    screen_tyMax: 0.5,
    screen_curvatureMin: 0.0,
    screen_curvatureMax: 1.0,
    screen_edge_fadeMin: 0.0,
    screen_edge_fadeMax: 0.25,
    screen_bg_colorMin: 0.0,
    screen_bg_colorMax: 1.0,
    screen_rollMin: -45, 
    screen_rollMax: 45,
    color_brightnessMin: 0.0,
    color_brightnessMax: 2.0,
    color_contrastMin: 0.0,
    color_contrastMax: 2.0,
    color_gammaMin: 0.5,
    color_gammaMax: 2.5,
    color_saturationMin: 0.0,
    color_saturationMax: 2.0,
    color_hueMin: -45,
    color_hueMax: 45,
    environment_rotationMin: 0,
    environment_rotationMax: 360,
    environment_tiltMin: -90,
    environment_tiltMax: 90,
    environment_intensityMin: -6.0,
    environment_intensityMax: 6.0,
    // Model scale factor exponent: 10^x (x in [-1, 1] maps to [0.1, 10.0], 0 -> 1.0)
    environment_model_scaleMin: -1.0,
    environment_model_scaleMax: 1.0,
    environment_model_yMin: -5.0,
    environment_model_yMax: 5.0,
    subtitle_yMin: -1.0,
    subtitle_yMax: 1.0,
    subtitle_zMin: -1.0, // Offset from screen
    subtitle_zMax: 4.0,
    subtitle_font_sizeMin: 0.5,
    subtitle_font_sizeMax: 2.0,
    subtitle_eye_sepMin: -0.15,
    subtitle_eye_sepMax: 0.15,
    screen_height_indexMin: 0.0, // 2^0 = 1m
    screen_height_indexMax: 4.0, // 2^4 = 16m
    screen_scaleMin: 1.0,
    screen_scaleMax: 16.0
};

export const DEFAULTS = {
    screen_tilt: 0,
    screen_zoom: -3.0,
    screen_eye_sep: 0.0,
    screen_tx: 0.0,
    screen_ty: 0.0,
    screen_curvature: 0.1,
    screen_edge_fade: 0.0,
    screen_bg_color: 0.0,
    screen_roll: 0.0,
    screen_height_index: 1.0, // 2^1 = 2m
    screen_volume: 0.5,

    environment_name: "None",
    environment_model_name: "None",
    environment_rotation: 0,
    environment_tilt: 0,
    environment_intensity: 0.0,
    environment_model_scale: 0.0, // Exponent: 10^0 = 1.0 (1x scale)
    environment_model_y: 0.0,
    environment_ambient_min: 0.03,
    uiZ: -0.9,
    uiForwardLimit: 0.09,
    
    color_brightness: 1.0,
    color_contrast: 1.0,
    color_gamma: 1.0,
    color_saturation: 1.0,
    color_hue: 0.0,
    color_lut: "None",

    video_repeat: true,
    video_muted: false,
    video_speed: 1.0,

    render_antialias: true,
    render_ss: 2.0,
    render_fps: 0,
    render_font: 'Auto',
    render_xr_layers: true,

    menu_alignment: 'right',

    subtitle_visible: true,
    subtitle_y: -0.5,
    subtitle_z: 0.5,
    subtitle_font_size: 1.25,
    subtitle_eye_sep: 0.0
};


export const UI_CONFIG = {
    pixelSize: 0.001,
    occlusionTimeConstant: 250, // ms
    occlusionHysteresisFar: 0.05,
    occlusionHysteresisNear: 0.10,
    hapticHoverIntensity: 0.3,
    hapticHoverDuration: 10,
    hapticClickIntensity: 0.6,
    hapticClickDuration: 25,
    hapticLimitIntensity: 0.5,
    hapticLimitDuration: 15,
    translationSensitivity: 3.33,
    eyeSeparationSensitivity: 0.125,
    eyeSeparationSensitivityMin: 0.01875,
    menuMarginTop: -1180,
    menuMarginLeft: 450
};

export const STEREO_FORMATS = {
    SBS_FULL: "SBS_FULL",
    SBS_HALF: "SBS_HALF",
    SBS_FULL_CROSS: "SBS_FULL_CROSS",
    TB_FULL: "TB_FULL",
    TB_HALF: "TB_HALF",
    SBS_180: "SBS_180",
    FLAT: "FLAT"
};

export const VR_CONFIG = {
    screenSegments: 64
};

export const PREFETCH_CONFIG = {
    count: 3,
    maxSize: 10
};

export const FONT_CONFIG = {
    fontFamilies: {
        NotoSansJP: { 400: '/fonts/NotoSansJP.json' },
        NotoSansSC: { 400: '/fonts/NotoSansSC.json' },
        NotoSansTC: { 400: '/fonts/NotoSansTC.json' },
        NotoSans: { 400: '/fonts/NotoSans.json' },
        NotoSansKR: { 400: '/fonts/NotoSansKR.json' }
    }
};

export const COLORS = {
    bg: 0x111111,
    bgDark: 0x080808,
    accent: 0x6666cc,
    accentHover: 0x8080cc,
    hover: 0x666666,
    text: 0xeeeeee,
    textDim: 0xcccccc,
    button: 0x222222,
    buttonDisabled: 0x111111,
    track: 0x333333,
    thumb: 0xeeeeee,
    separator: 0x333333,
    border: 0x333333,
    bgTarget: 0xffffff,
    hint: 0xccccff,
    recent: 0x4444aa,
    recentHover: 0x5555bb,
    explorerActive: 0x333333
};

export const INPUT_ACTIONS = {
    MENU_TOGGLE: 'MENU_TOGGLE',
    NAV_NEXT: 'NAV_NEXT',
    NAV_PREV: 'NAV_PREV',
    PAGE_NEXT: 'PAGE_NEXT',
    PAGE_PREV: 'PAGE_PREV',
    ZOOM_IN: 'ZOOM_IN',
    ZOOM_OUT: 'ZOOM_OUT',
    EYE_SEP_INC: 'EYE_SEP_INC',
    EYE_SEP_DEC: 'EYE_SEP_DEC',
    SCREEN_SIZE_INC: 'SCREEN_SIZE_INC',
    SCREEN_SIZE_DEC: 'SCREEN_SIZE_DEC',
    SEEK_FWD: 'SEEK_FWD',
    SEEK_BWD: 'SEEK_BWD',
    PLAY_PAUSE: 'PLAY_PAUSE'
};

/**
 * Metadata for all settings to handle global/file-specific logic, UI conversions, labels, and limits.
 * Supports uniform values or screen-specific overrides ({ plane: ..., dome: ... }).
 */
export const SETTINGS_METADATA = {
    // --- Screen Settings ---
    screen_height_index: { 
        isFileSpecific: true,
        group: 'screen',
        label: { plane: "Screen Size", dome: "Fixed Size" },
        shortcut: "[Grip + Stick X]",
        default: DEFAULTS.screen_height_index,
        min: LIMITS.screen_height_indexMin,
        max: LIMITS.screen_height_indexMax
    },
    screen_zoom: { 
        isFileSpecific: true, 
        group: 'screen',
        label: { plane: "Distance", dome: "Zoom" },
        shortcut: "[Stick Y]",
        default: DEFAULTS.screen_zoom,
        min: -LIMITS.screen_zoomMax,
        max: -LIMITS.screen_zoomMin
    },
    screen_eye_sep: { 
        isFileSpecific: true, 
        group: 'screen',
        label: "Scale",
        shortcut: "[Grip + Stick Y]",
        default: DEFAULTS.screen_eye_sep,
        min: LIMITS.screen_eye_sepMin,
        max: LIMITS.screen_eye_sepMax
    },
    screen_tx: { 
        isFileSpecific: true,
        group: 'screen',
        label: { plane: "Pos X", dome: "Pan" },
        shortcut: "[Grip + Move]",
        default: DEFAULTS.screen_tx,
        min: { plane: -LIMITS.screen_txMax, dome: -0.5 },
        max: { plane: LIMITS.screen_txMax, dome: 0.5 },
        dynamicRatio: LIMITS.screen_txMax
    },
    screen_ty: { 
        isFileSpecific: true,
        group: 'screen',
        label: "Pos Y",
        shortcut: "[Grip + Move]",
        default: DEFAULTS.screen_ty,
        min: { plane: -LIMITS.screen_tyMax, dome: -0.5 },
        max: { plane: LIMITS.screen_tyMax, dome: 0.5 },
        dynamicRatio: LIMITS.screen_tyMax
    },
    screen_roll: {
        isFileSpecific: true,
        group: 'screen',
        label: "Roll",
        shortcut: "[Trigger + Move X]",
        default: DEFAULTS.screen_roll,
        min: LIMITS.screen_rollMin,
        max: LIMITS.screen_rollMax
    },
    screen_tilt: { 
        group: 'screen',
        label: { plane: "Tilt", dome: "Tilt (Global)" },
        shortcut: "[Trigger + Tilt]",
        default: DEFAULTS.screen_tilt,
        min: LIMITS.screen_tiltMin,
        max: LIMITS.screen_tiltMax
    },
    screen_curvature: { 
        group: 'screen', 
        label: "Curvature",
        default: DEFAULTS.screen_curvature,
        min: LIMITS.screen_curvatureMin,
        max: LIMITS.screen_curvatureMax
    },
    screen_edge_fade: { 
        group: 'screen', 
        label: "Edge Fade",
        default: DEFAULTS.screen_edge_fade,
        min: LIMITS.screen_edge_fadeMin,
        max: LIMITS.screen_edge_fadeMax
    },
    screen_bg_color: { 
        group: 'screen', 
        label: "Background",
        default: DEFAULTS.screen_bg_color,
        min: LIMITS.screen_bg_colorMin,
        max: LIMITS.screen_bg_colorMax
    },
    screen_volume: {
        group: 'general',
        label: "Volume",
        min: 0,
        max: 1,
        default: DEFAULTS.screen_volume
    },
    
    // --- Subtitle Settings ---
    subtitle_visible: { isFileSpecific: true, group: 'subtitle', default: DEFAULTS.subtitle_visible },
    subtitle_y: { 
        isFileSpecific: true, 
        group: 'subtitle', 
        label: "Vertical Position",
        min: LIMITS.subtitle_yMin, 
        max: LIMITS.subtitle_yMax, 
        default: DEFAULTS.subtitle_y 
    },
    subtitle_z: { 
        isFileSpecific: true, 
        group: 'subtitle', 
        label: "Distance",
        min: LIMITS.subtitle_zMin, 
        max: LIMITS.subtitle_zMax, 
        default: DEFAULTS.subtitle_z 
    },
    subtitle_font_size: { 
        isFileSpecific: true, 
        group: 'subtitle', 
        label: "Font Size",
        min: LIMITS.subtitle_font_sizeMin, 
        max: LIMITS.subtitle_font_sizeMax, 
        default: DEFAULTS.subtitle_font_size 
    },
    subtitle_eye_sep: { 
        isFileSpecific: true, 
        group: 'subtitle', 
        label: "Eye Separation",
        min: LIMITS.subtitle_eye_sepMin, 
        max: LIMITS.subtitle_eye_sepMax, 
        default: DEFAULTS.subtitle_eye_sep 
    },
    subtitle_index: { isFileSpecific: true, isInternal: true, group: 'subtitle' }, 
    playback_time: { isFileSpecific: true, isInternal: true, group: 'playback' },
    stereo_format: { isFileSpecific: true, isInternal: true, group: 'format' },

    // --- Color Settings ---
    color_contrast: { 
        group: 'color', 
        label: "Contrast",
        min: LIMITS.color_contrastMin, 
        max: LIMITS.color_contrastMax, 
        default: DEFAULTS.color_contrast 
    },
    color_brightness: { 
        group: 'color', 
        label: "Brightness",
        min: LIMITS.color_brightnessMin, 
        max: LIMITS.color_brightnessMax, 
        default: DEFAULTS.color_brightness 
    },
    color_gamma: { 
        group: 'color', 
        label: "Gamma",
        min: LIMITS.color_gammaMin, 
        max: LIMITS.color_gammaMax, 
        default: DEFAULTS.color_gamma 
    },
    color_saturation: { 
        group: 'color', 
        label: "Saturation",
        min: LIMITS.color_saturationMin, 
        max: LIMITS.color_saturationMax, 
        default: DEFAULTS.color_saturation 
    },
    color_hue: { 
        group: 'color', 
        label: "Hue",
        min: LIMITS.color_hueMin, 
        max: LIMITS.color_hueMax, 
        default: DEFAULTS.color_hue 
    },
    color_lut: { group: 'color', default: DEFAULTS.color_lut },

    // --- Environment Settings ---
    environment_name: { group: 'environment', default: DEFAULTS.environment_name },
    environment_model_name: { group: 'environment', default: DEFAULTS.environment_model_name },
    environment_tilt: { 
        group: 'environment', 
        label: "Tilt",
        min: LIMITS.environment_tiltMin, 
        max: LIMITS.environment_tiltMax, 
        default: DEFAULTS.environment_tilt 
    },
    environment_rotation: { 
        group: 'environment', 
        label: "Rotation",
        min: LIMITS.environment_rotationMin, 
        max: LIMITS.environment_rotationMax, 
        default: DEFAULTS.environment_rotation 
    },
    environment_intensity: { 
        group: 'environment', 
        label: "Brightness",
        min: LIMITS.environment_intensityMin, 
        max: LIMITS.environment_intensityMax, 
        default: DEFAULTS.environment_intensity 
    },
    environment_model_scale: { 
        group: 'environment', 
        label: "Scale",
        min: LIMITS.environment_model_scaleMin, 
        max: LIMITS.environment_model_scaleMax, 
        default: DEFAULTS.environment_model_scale // Exponent: 10^x
    },
    environment_model_y: { 
        group: 'environment', 
        label: "Pos Y",
        min: LIMITS.environment_model_yMin, 
        max: LIMITS.environment_model_yMax, 
        default: DEFAULTS.environment_model_y 
    },

    // --- Render Settings ---
    render_antialias: { group: 'render', default: DEFAULTS.render_antialias },
    render_ss: { group: 'render', default: DEFAULTS.render_ss },
    render_fps: { group: 'render', default: DEFAULTS.render_fps },
    render_font: { group: 'render', default: DEFAULTS.render_font },
    render_xr_layers: { group: 'render', default: DEFAULTS.render_xr_layers },

    // --- General / Video Settings ---
    video_repeat: { group: 'general', default: DEFAULTS.video_repeat },
    video_muted: { group: 'general', default: DEFAULTS.video_muted },
    video_speed: {
        group: 'general',
        label: "Speed",
        min: 0.25,
        max: 4.0,
        default: DEFAULTS.video_speed
    },
    menu_alignment: { group: 'general', default: DEFAULTS.menu_alignment }
};

/**
 * Finds the metadata entry for a given setting key.
 */
export function findSettingMeta(key) {
    return SETTINGS_METADATA[key];
}

/**
 * Resolves min and max limits for a setting based on screen type and active screen dimensions.
 */
export function getSettingLimits(id, screenType = 'plane', activeScreen = null) {
    const meta = findSettingMeta(id);
    if (!meta) return { min: 0, max: 1 };

    const isDome = screenType === 'dome' || screenType === 'dome_xr';
    const typeKey = isDome ? 'dome' : 'plane';

    // Dynamic limit calculation for Plane screen (meters)
    if (!isDome && meta.dynamicRatio) {
        const height = activeScreen?.physicalScreenHeight || 1.0;
        const aspect = (id === 'screen_tx') ? (activeScreen?.aspectRatio || 1.0) : 1.0;
        const limit = meta.dynamicRatio * height * aspect;
        return { min: -limit, max: limit };
    }

    // 2. Static limits resolution (primitive or screen-specific object)
    const resolveVal = (prop, fallback) => {
        if (prop === undefined) return fallback;
        if (typeof prop === 'object' && prop !== null) {
            return prop[screenType] ?? prop[typeKey] ?? prop.plane ?? fallback;
        }
        return prop;
    };

    return {
        min: resolveVal(meta.min, 0),
        max: resolveVal(meta.max, 1)
    };
}

/**
 * Resolves label for a setting based on screen type.
 */
export function getSettingLabel(id, screenType = 'plane') {
    const meta = findSettingMeta(id);
    if (!meta?.label) return id;
    if (typeof meta.label === 'object') {
        const isDome = screenType === 'dome' || screenType === 'dome_xr';
        return meta.label[screenType] ?? meta.label[isDome ? 'dome' : 'plane'] ?? meta.label.plane ?? id;
    }
    return meta.label;
}

/**
 * Resolves default value for a setting based on screen type.
 */
export function getSettingDefault(id, screenType = 'plane') {
    const meta = findSettingMeta(id);
    if (!meta || meta.default === undefined) {
        return DEFAULTS[id];
    }
    if (typeof meta.default === 'object' && meta.default !== null && ('plane' in meta.default || 'dome' in meta.default)) {
        const isDome = screenType === 'dome' || screenType === 'dome_xr';
        return meta.default[screenType] ?? meta.default[isDome ? 'dome' : 'plane'] ?? meta.default.plane ?? DEFAULTS[id];
    }
    return meta.default;
}


