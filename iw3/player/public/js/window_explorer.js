import { Container, Image, Text } from '@pmndrs/uikit';
import { COLORS, UI_CONFIG, FONT_CONFIG } from './constants.js';
import { UIUtils, ThreeUtils } from './ui_common.js';

class ExplorerWindow {
    constructor(uiManager, defaultFont, galleryManager = null) {
        this.uiManager = uiManager;
        this.defaultFont = defaultFont;
        this.galleryManager = galleryManager;
        this.container = new Container({
            flexDirection: 'column',
            backgroundColor: COLORS.bgDark,
            backgroundOpacity: 0.95,
            borderRadius: 16,
            borderWidth: 2,
            borderColor: COLORS.border,
            padding: 24,
            width: 800,
            height: 800,
            display: 'none',
            position: 'absolute',
            marginLeft: -0,
            marginTop: UI_CONFIG.menuMarginTop - 60,
            depthWrite: false,
            ...FONT_CONFIG,
            fontFamily: defaultFont
        });
        this.container.visible = false;

        this.setupHeader();

        this.content = new Container({
            flexDirection: 'column',
            width: '100%',
            flexGrow: 1,
            overflow: 'visible',
            ...FONT_CONFIG,
            fontFamily: defaultFont
        });
        this.container.add(this.content);

        this.setupFooter();
    }

    setupHeader() {
        this.header = new Container({
            flexDirection: 'row',
            width: '100%',
            height: 80,
            alignItems: 'center',
            gap: 16,
            padding: 16,
            backgroundColor: COLORS.bg,
            borderBottomWidth: 2,
            borderBottomColor: COLORS.border,
        });
        this.container.add(this.header);

        const u = this.uiManager;
        const upBtn = UIUtils.createButton('icons/arrow-big-up.svg', () => u.navigateUp(), u, 32, 48);
        upBtn.container.setProperties({ borderRadius: 8 });
        this.header.add(upBtn.container);

        this.pathText = new Text({
            text: "/",
            fontSize: 20,
            color: COLORS.textDim,
            flexGrow: 1
        });
        this.header.add(this.pathText);

        const reloadBtn = UIUtils.createButton('icons/refresh.svg', () => {
            const currentPath = this.galleryManager ? this.galleryManager.currentPath : "/";
            u.fetchDirectory(currentPath);
        }, u, 32, 48);
        reloadBtn.container.setProperties({ borderRadius: 8 });
        this.header.add(reloadBtn.container);
    }

    setupFooter() {
        this.footer = new Container({
            flexDirection: 'row',
            width: '100%',
            height: 80,
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: COLORS.bg,
            borderTopWidth: 2,
            borderTopColor: COLORS.border,
            borderRadius: 16,
            paddingX: 16,
        });
        this.container.add(this.footer);

        const u = this.uiManager;

        // Left side: Sort Buttons
        const sortGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 12 });
        this.footer.add(sortGroup);
        this.sortButtons = {};

        const modes = [
            { id: 'name_asc', icon: 'icons/sort-ascending-letters.svg' },
            { id: 'name_desc', icon: 'icons/sort-descending-letters.svg' },
            { id: 'date_asc', icon: 'icons/sort-ascending-calendar.svg' },
            { id: 'date_desc', icon: 'icons/sort-descending-calendar.svg' },
        ];

        modes.forEach(m => {
            const btn = UIUtils.createButton(m.icon, () => u.setSortMode(m.id), u, 28, 44);
            btn.container.setProperties({ borderRadius: 8 });
            sortGroup.add(btn.container);
            this.sortButtons[m.id] = btn;
        });

        // Right side: Paging Buttons
        const pagingGroup = new Container({ flexDirection: 'row', alignItems: 'center', gap: 12 });
        this.footer.add(pagingGroup);

        const prevPageBtn = UIUtils.createButton("icons/circle-arrow-left.svg", () => u.changePage(-1), u, 32, 48);
        prevPageBtn.container.setProperties({ borderRadius: 8 });
        pagingGroup.add(prevPageBtn.container);

        this.pageText = new Text({
            text: "1 / 1",
            fontSize: 18,
            color: COLORS.text,
            width: 80,
            textAlign: 'center'
        });
        pagingGroup.add(this.pageText);

        const nextPageBtn = UIUtils.createButton("icons/circle-arrow-right.svg", () => u.changePage(1), u, 32, 48);
        nextPageBtn.container.setProperties({ borderRadius: 8 });
        pagingGroup.add(nextPageBtn.container);
    }

    render(items, path, page, itemsPerPage, activePath = null) {
        const children = [...this.content.children];
        children.forEach(child => {
            ThreeUtils.disposeObject(child, this.content);
        });

        this.pathText.setProperties({ text: path });

        const totalPages = Math.ceil(items.length / itemsPerPage);
        this.pageText.setProperties({ text: `${page + 1} / ${Math.max(1, totalPages)}` });

        if (this.sortButtons) {
            const current = this.galleryManager ? this.galleryManager.sortMode : 'name_asc';
            Object.entries(this.sortButtons).forEach(([id, btn]) => {
                if (btn.setSelected) btn.setSelected(id === current);
            });
        }

        const grid = new Container({
            flexDirection: 'row',
            flexWrap: 'wrap',
            width: '100%',
            gap: 15,
            padding: 20,
            paddingBottom: 20,
            ...FONT_CONFIG,
            fontFamily: this.defaultFont
        });
        this.content.add(grid);

        const u = this.uiManager;
        items.slice(page * itemsPerPage, (page + 1) * itemsPerPage).forEach(item => {
            const isActive = item.path === activePath;
            const b = new Container({
                flexDirection: 'column',
                width: 120,
                height: 180,
                padding: 5,
                alignItems: 'center',
                gap: 4,
                borderRadius: 12,
                backgroundColor: isActive ? COLORS.explorerActive : undefined,
                cursor: 'pointer',
                hover: { backgroundColor: COLORS.hover },
                active: { backgroundColor: COLORS.accent },
                onPointerEnter: (e) => {
                    if (!u.visible || u.activeSubMenu !== this) return;
                    UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticHoverIntensity, UI_CONFIG.hapticHoverDuration, u);
                },
                onClick: (e) => {
                    if (!u.visible || u.activeSubMenu !== this) return;
                    UIUtils.vibratePointer(e.pointerId, UI_CONFIG.hapticClickIntensity, UI_CONFIG.hapticClickDuration, u);
                    if (item.type === 'directory' || item.type === 'archive') {
                        u.fetchDirectory("/" + item.path);
                    } else {
                        u.openImageFromExplorer(items, item);
                    }
                },
            });
            
            let thumbUrl;
            let iconColor = undefined;
            if (item.type === 'directory') {
                thumbUrl = 'icons/folder.svg';
                iconColor = COLORS.text;
            } else if (item.type === 'archive') {
                thumbUrl = 'icons/file-zip.svg';
                iconColor = COLORS.text;
            } else {
                thumbUrl = `/api/thumbnail?path=${encodeURIComponent(item.path)}`;
            }
            b.add(new Image({ src: thumbUrl, width: 80, height: 80, aspectRatio: 1, color: iconColor }));
            
            const txtWrap = new Container({
                width: '100%',
                height: 66,
                alignItems: 'flex-start',
                overflow: 'hidden',
                paddingTop: 4
            });
            txtWrap.add(new Text({ 
                text: item.name,
                fontSize: 14,
                color: COLORS.text,
                textAlign: 'center',
                width: 110,
                wordBreak: 'break-all'
            }));
            b.add(txtWrap);
            grid.add(b);
        });
    }
}

export { ExplorerWindow };
