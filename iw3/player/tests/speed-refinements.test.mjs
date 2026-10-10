import test from 'node:test';
import assert from 'node:assert/strict';
import {PlaybackSpeed} from '../public/js/playback_speed.js';
import {fixture,event,load} from './ui-fixture.mjs';

test('Quest defaults, proportional steps, and intentional previous defaults persist',()=>{
    const speed=new PlaybackSpeed();
    assert.equal(speed.min,0.1);assert.equal(speed.max,16);
    assert.equal(new PlaybackSpeed({min:.25,max:16,rate:1}).min,.1);
    speed.setBounds(.25,16);
    assert.equal(new PlaybackSpeed(speed.toJSON()).min,.25);
    for (const rate of [.25,1,8]) {
        speed.setRate(rate);speed.step(1);
        assert.ok(Math.abs(speed.rate-rate*1.1)<1e-12);
        speed.step(-1);assert.ok(Math.abs(speed.rate-rate)<1e-12);
    }
    speed.setRate(16);speed.step(1);assert.equal(speed.rate,16);
});

test('popup defaults reset restores whole speed config and bound taps apply immediately',async()=>{
    const {u,UIUtils,cache}=await fixture();
    const MainMenu=await load('menu_main','MainMenu',{UIUtils});
    const menu=new MainMenu(u,'NotoSans');u.mainMenu=menu;u.speedMenu=menu.speedMenu;
    menu.speedReadout.props.onClick(event());
    assert.ok(menu.speedDefaults,'restore defaults control exists');
    await menu.speedBoundFields.min.props.onClick(event());
    assert.equal(u.stereoPlayer.videoElement.playbackRate,.1);
    await menu.speedBoundFields.max.props.onClick(event());
    assert.equal(u.stereoPlayer.videoElement.playbackRate,16);
    await u.setPlaybackSpeedBounds(.5,3);await u.setPlaybackSpeed(2);
    await menu.speedDefaults.props.onClick(event());
    assert.equal(u.playbackSpeed.min,.1);assert.equal(u.playbackSpeed.max,16);
    assert.equal(u.stereoPlayer.videoElement.playbackRate,1);
    assert.equal(cache.playback_speed.min,.1);assert.equal(cache.playback_speed.max,16);
    assert.equal(menu.speedKeypad.props.display,'none');
});
