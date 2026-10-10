import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture,event,load} from './ui-fixture.mjs';

test('compact speed readout opens a separate hidden popup on right click or VR click',async()=>{
    const {u,UIUtils}=await fixture();
    const MainMenu=await load('menu_main','MainMenu',{UIUtils});
    const menu=new MainMenu(u,'NotoSans');
    assert.ok(menu.speedReadout,'compact readout replaces full-width speed row');
    assert.equal(menu.speedReadoutText.props.text,'1.0x');
    assert.equal(menu.speedMenu.container.props.display,'none');
    assert.ok(!menu.container.children.includes(menu.speedMenu.container),'popup is not an extra main-menu row');
    const right={...event(),button:2,stopPropagation(){}};
    menu.speedReadout.props.onPointerDown(right);
    assert.equal(u.activeSubMenu,menu.speedMenu);
    assert.equal(menu.speedMenu.container.props.display,'flex');
    menu.speedReadout.props.onClick(right);
    assert.equal(u.activeSubMenu,menu.speedMenu,'right click must not toggle twice');
    menu.speedClose.props.onClick(event());
    assert.equal(u.activeSubMenu,null);
    menu.speedReadout.props.onClick({...event(),button:0});
    assert.equal(u.activeSubMenu,menu.speedMenu);
    await u.setPlaybackSpeed(1.8);menu.sync({});
    assert.equal(menu.speedReadoutText.props.text,'1.8x');
    u.visible=false;
    menu.speedReadout.props.onPointerDown(right);
    assert.equal(u.activeSubMenu,menu.speedMenu,'hidden readout ignores input');
});
