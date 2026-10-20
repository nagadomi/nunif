import { storage } from './storage.js';

/**
 * Manages the file list, sorting, navigation and playback queue.
 */
class GalleryManager {
    constructor(debugLog) {
        this.debugLog = debugLog;
        
        this.currentPath = "/";
        this.items = [];            // All items in currently browsed directory (including folders)
        this.playbackGallery = [];  // Playable items in currently browsed directory
        
        this.playingItem = null;    // Currently playing item
        this.playingGallery = [];   // Gallery the playing item belongs to
        this.playingIndex = -1;     // Index in playingGallery
        
        this.sortMode = 'name_asc';
    }

    loadSettings() {
        this.sortMode = storage.get('sortMode', 'name_asc');
    }

    async fetchDirectory(path) {
        try {
            const response = await fetch(`/api/files?path=${encodeURIComponent(path)}`);
            if (!response.ok) throw new Error('Failed to fetch file list');
            
            const list = await response.json();
            this.currentPath = path;
            this.items = this._sortItems(list);
            
            // Update browsing playback gallery (only images and videos)
            this.playbackGallery = this.items.filter(i => i.type === 'image' || i.type === 'video');
            
            return this.items;
        } catch (e) {
            if (this.debugLog) this.debugLog.log(`Gallery Error: ${e.message}`);
            throw e;
        }
    }

    async setSortMode(mode) {
        this.sortMode = mode;
        await storage.set('sortMode', mode);
        
        // 1. Re-sort current browsing items
        this.items = this._sortItems(this.items);
        this.playbackGallery = this.items.filter(i => i.type === 'image' || i.type === 'video');

        // 2. Re-sort playing gallery to maintain playback order consistency
        if (this.playingItem && this.playingGallery.length > 0) {
            this.playingGallery = this._sortItems(this.playingGallery);
            this.playingIndex = this.playingGallery.findIndex(i => i.path === this.playingItem.path);
        }
    }

    _sortItems(items) {
        return items.sort((a, b) => {
            const isDirA = (a.type === 'directory' || a.type === 'archive');
            const isDirB = (b.type === 'directory' || b.type === 'archive');

            // Folders/Archives always come first
            if (isDirA && !isDirB) return -1;
            if (!isDirA && isDirB) return 1;

            // Same type (both folders or both files) - apply current sort mode
            switch (this.sortMode) {
                case 'name_desc': 
                    return b.name.localeCompare(a.name);
                case 'date_asc': 
                    return (a.mtime || 0) - (b.mtime || 0) || a.name.localeCompare(b.name);
                case 'date_desc': 
                    return (b.mtime || 0) - (a.mtime || 0) || b.name.localeCompare(a.name);
                case 'name_asc':
                default: 
                    return a.name.localeCompare(b.name);
            }
        });
    }

    getParentPath() {
        if (this.currentPath === "/") return null;
        const p = this.currentPath.split("/").filter(x => x);
        p.pop();
        return "/" + p.join("/");
    }

    /**
     * Finds index in the browsing playback gallery.
     */
    findPlaybackIndexByPath(path) {
        return this.playbackGallery.findIndex(i => i.path === path);
    }

    getNextIndex(direction) {
        if (this.playingGallery.length === 0) return -1;
        const next = this.playingIndex + direction;
        if (next >= 0 && next < this.playingGallery.length) {
            return next;
        }
        return -1;
    }

    getCurrentItem() {
        return this.playingItem;
    }

    setPlayingItem(index) {
        if (index >= 0 && index < this.playbackGallery.length) {
            this.playingItem = this.playbackGallery[index];
            this.playingGallery = [...this.playbackGallery];
            this.playingIndex = index;
        }
    }
}

export { GalleryManager };