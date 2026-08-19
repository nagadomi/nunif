/** @type {import('eslint').Linter.Config[]} */
export default [
    {
        files: ["public_html/js/*.js"],
        languageOptions: {
            ecmaVersion: "latest",
            sourceType: "module",
            globals: {
                // Browser globals
                window: "readonly",
                document: "readonly",
                navigator: "readonly",
                console: "readonly",
                localStorage: "readonly",
                FileReader: "readonly",
                File: "readonly",
                Blob: "readonly",
                URL: "readonly",
                ImageData: "readonly",
                Uint8ClampedArray: "readonly",
                Float32Array: "readonly",
                BigInt64Array: "readonly",
                OffscreenCanvas: "readonly",
                DataTransfer: "readonly",
                Event: "readonly",
                setInterval: "readonly",
                clearInterval: "readonly",
                setTimeout: "readonly",
                Image: "readonly",
                // Project specific (onnxruntime-web)
                ort: "readonly"
            }
        },
        rules: {
            "no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }],
            "no-constant-condition": "off",
            "no-undef": "error",
            "quotes": ["error", "double", { "avoidEscape": true }],
            "semi": ["error", "always"]
        }
    }
];
