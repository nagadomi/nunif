/**
 * StorageManager handles persistent data using IndexedDB.
 * General settings are cached in memory for synchronous access,
 * while large file-specific configs are accessed asynchronously.
 * 
 * Supports transparent encryption for sensitive keys using Web Crypto API.
 */
class StorageManager {
    static DB_NAME = 'iw3_player_db';
    static DB_VERSION = 1;
    static STORE_CONFIGS = 'configs';
    static STORE_FILE_CONFIGS = 'file_configs';
    
    // Keys that should be stored encrypted in IndexedDB
    static SECURE_KEYS = new Set(['recent_file_name', 'recent_path']);

    constructor() {
        this.db = null;
        this.configCache = {};
        this.debugLog = null;
        this.cryptoKey = null;
        this.ready = this.init();
    }

    setDebugLog(debugLog) {
        this.debugLog = debugLog;
    }

    async init() {
        // First initialize DB
        await this.initDB();
        // Then try to fetch encryption key
        await this.initCrypto();
        // Load settings to cache (decrypting sensitive ones)
        await this._loadConfigsToCache();
    }

    _log(msg) {
        if (this.debugLog) this.debugLog.log(msg);
        else console.log(msg);
    }

    _warn(msg) {
        if (this.debugLog) this.debugLog.log(`[WARN] ${msg}`);
        else console.warn(msg);
    }

    _error(msg, err = null) {
        const fullMsg = err ? `${msg}: ${err.message || err}` : msg;
        if (this.debugLog) this.debugLog.log(`[ERROR] ${fullMsg}`);
        else console.error(fullMsg);
    }

    async initDB() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(StorageManager.DB_NAME, StorageManager.DB_VERSION);

            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(StorageManager.STORE_CONFIGS)) {
                    db.createObjectStore(StorageManager.STORE_CONFIGS);
                }
                if (!db.objectStoreNames.contains(StorageManager.STORE_FILE_CONFIGS)) {
                    db.createObjectStore(StorageManager.STORE_FILE_CONFIGS);
                }
            };

            request.onsuccess = (event) => {
                this.db = event.target.result;
                resolve();
            };

            request.onerror = (event) => {
                this._error("IndexedDB error", event.target.error);
                reject(event.target.error);
            };
        });
    }

    /**
     * Fetches the encryption key from the backend.
     */
    async initCrypto() {
        try {
            const response = await fetch('/api/key');
            if (!response.ok) throw new Error("Failed to fetch key");
            const data = await response.json();
            const rawKey = Uint8Array.from(atob(data.key), c => c.charCodeAt(0));
            
            this.cryptoKey = await window.crypto.subtle.importKey(
                'raw', rawKey, { name: 'AES-GCM' }, false, ['encrypt', 'decrypt']
            );
        } catch (e) {
            this._warn(`Encryption key not available. Secure storage will be disabled: ${e.message}`);
        }
    }

    /**
     * Encrypts a string value using AES-GCM.
     * Returns base64(IV + Ciphertext)
     */
    async _encrypt(plaintext) {
        if (!this.cryptoKey) return plaintext;
        const iv = window.crypto.getRandomValues(new Uint8Array(12));
        const encoded = new TextEncoder().encode(plaintext);
        const encrypted = await window.crypto.subtle.encrypt(
            { name: 'AES-GCM', iv: iv }, this.cryptoKey, encoded
        );
        
        // Combine IV and encrypted data
        const combined = new Uint8Array(iv.length + encrypted.byteLength);
        combined.set(iv);
        combined.set(new Uint8Array(encrypted), iv.length);
        
        return btoa(String.fromCharCode.apply(null, combined));
    }

    /**
     * Decrypts a base64 encoded value.
     */
    async _decrypt(base64Data) {
        if (!this.cryptoKey || !base64Data) return base64Data;
        try {
            const combined = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
            const iv = combined.slice(0, 12);
            const data = combined.slice(12);
            
            const decrypted = await window.crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: iv }, this.cryptoKey, data
            );
            return new TextDecoder().decode(decrypted);
        } catch (e) {
            this._error("Decryption failed (maybe key changed?)", e);
            return null;
        }
    }

    async _loadConfigsToCache() {
        const tx = this.db.transaction(StorageManager.STORE_CONFIGS, 'readonly');
        const store = tx.objectStore(StorageManager.STORE_CONFIGS);
        
        return new Promise((resolve) => {
            const req = store.getAll();
            const keysReq = store.getAllKeys();

            tx.oncomplete = async () => {
                const keys = keysReq.result;
                const vals = req.result;
                
                for (let i = 0; i < keys.length; i++) {
                    const key = keys[i];
                    let val = vals[i];
                    
                    if (StorageManager.SECURE_KEYS.has(key) && typeof val === 'string') {
                        val = await this._decrypt(val);
                    }
                    this.configCache[key] = val ? structuredClone(val) : val;
                }
                resolve();
            };
        });
    }

    async getHash(file) {
        if (!file) return null;
        const msg = `${file.path}|${file.mtime}`;
        const msgBuffer = new TextEncoder().encode(msg);
        const hashBuffer = await crypto.subtle.digest('SHA-256', msgBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * General settings API
     */
    get(key, defaultValue = null) {
        const val = this.configCache[key];
        return val !== undefined ? val : defaultValue;
    }

    async set(key, value) {
        if (JSON.stringify(this.configCache[key]) === JSON.stringify(value)) return;

        this.configCache[key] = structuredClone(value);

        const isSecure = StorageManager.SECURE_KEYS.has(key);
        let storedValue = value;
        if (isSecure && typeof value === 'string') {
            storedValue = await this._encrypt(value);
        }

        const msg = `[Storage] Writing general config: ${key}${isSecure ? ' (encrypted)' : ''}`;
        this._log(msg);

        return new Promise((resolve, reject) => {
            const tx = this.db.transaction(StorageManager.STORE_CONFIGS, 'readwrite');
            tx.objectStore(StorageManager.STORE_CONFIGS).put(storedValue, key);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error);
        });
    }

    async getFileConfig(file) {
        await this.ready;
        const key = await this.getHash(file);
        if (!key) return null;

        return new Promise((resolve) => {
            const tx = this.db.transaction(StorageManager.STORE_FILE_CONFIGS, 'readonly');
            const store = tx.objectStore(StorageManager.STORE_FILE_CONFIGS);
            const req = store.get(key);
            
            req.onsuccess = () => {
                const config = req.result || null;
                if (config) {
                    this._log(`[Storage] Loaded file config [${key.substring(0, 8)}]`);
                }
                resolve(config);
            };
            req.onerror = () => resolve(null);
        });
    }

    async updateFileConfig(file, updates) {
        await this.ready;
        const key = await this.getHash(file);
        if (!key) return;

        return new Promise((resolve) => {
            const tx = this.db.transaction(StorageManager.STORE_FILE_CONFIGS, 'readwrite');
            const store = tx.objectStore(StorageManager.STORE_FILE_CONFIGS);
            const req = store.get(key);
            
            req.onsuccess = () => {
                let config = req.result || {};
                let changed = false;
                for (const k in updates) {
                    const val = updates[k];
                    if (val === undefined || val === null) {
                        if (config[k] !== undefined) {
                            delete config[k];
                            changed = true;
                        }
                    } else if (config[k] !== val) {
                        config[k] = val;
                        changed = true;
                    }
                }

                if (changed) {
                    if (Object.keys(config).length === 0) {
                        this._log(`[Storage] Removed file config [${key.substring(0, 8)}]`);
                        store.delete(key);
                    } else {
                        this._log(`[Storage] Writing file config [${key.substring(0, 8)}] (keys: ${Object.keys(updates).join(', ')})`);
                        store.put(config, key);
                    }
                }
            };
            tx.oncomplete = () => resolve();
            tx.onerror = () => resolve();
        });
    }

    /**
     * Clears all data (Factory Reset).
     * Clears all IndexedDB object stores without closing/deleting the DB connection,
     * so reopening is not required and the instance remains immediately usable.
     */
    async clearAll() {
        await this.ready;
        this.configCache = {};

        // 1. Clear all IndexedDB object stores dynamically
        await new Promise((resolve, reject) => {
            if (!this.db) {
                resolve();
                return;
            }
            const storeNames = Array.from(this.db.objectStoreNames);
            if (storeNames.length === 0) {
                resolve();
                return;
            }
            try {
                const tx = this.db.transaction(storeNames, 'readwrite');
                for (const name of storeNames) {
                    tx.objectStore(name).clear();
                }
                tx.oncomplete = () => resolve();
                tx.onerror = (e) => reject(tx.error || e);
                tx.onabort = (e) => reject(tx.error || e);
            } catch (err) {
                reject(err);
            }
        });

        // 2. Clear browser local and session storage
        try {
            localStorage.clear();
        } catch (e) {
            this._warn(`Failed to clear localStorage: ${e.message}`);
        }
        try {
            sessionStorage.clear();
        } catch (e) {
            this._warn(`Failed to clear sessionStorage: ${e.message}`);
        }

        // 3. Clear Cache Storage API if available
        if (typeof window !== 'undefined' && 'caches' in window) {
            try {
                const cacheNames = await window.caches.keys();
                await Promise.all(cacheNames.map(name => window.caches.delete(name)));
            } catch (e) {
                this._warn(`Failed to clear CacheStorage: ${e.message}`);
            }
        }
    }
}

export const storage = new StorageManager();
