const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./screen-D1LH42Nc.js","./jsx-runtime-D3jfb0Ew.js","./common-drawer-DEXepnE4.js","./Tooltip-DQC8fZSf.js","./themeProps-BYXlitcH.js","./useLayerDismissal-D1tCdpVO.js","./useScrollLock-C70ZmxM0.js","./focusOutline.stylex-Aej9P5T1.js","./common-drawer-fLF2jYG2.css","./primitives-CJnwTjcT.js","./primitives-Bo7WZ0Dj.css","./load-action-D45w7mCS.js","./load-profiles-bFkiV_aw.js","./chat-BAZbjt2p.js","./format-BGiGtWmI.js","./format-DA1scLDv.js","./server-defaults-BzrglPzR.js","./activity-C5mFReiM.js","./settings-screen-DcqdmwEk.js","./gallery-5S5835UE.js"])))=>i.map(i=>d[i]);
import{a as e,n as t,r as n,t as r}from"./jsx-runtime-D3jfb0Ew.js";import{d as i}from"./Tooltip-DQC8fZSf.js";import{a,i as o,o as s,p as c,r as l,t as u}from"./themeProps-BYXlitcH.js";import{y as d}from"./useLayerDismissal-D1tCdpVO.js";import{A as f,B as p,F as m,L as h,N as g,Q as _,T as v,Z as y,c as b,et as ee,g as te,h as ne,i as re,r as ie,s as x,t as ae,v as oe,y as se,z as ce}from"./common-drawer-DEXepnE4.js";import"./useScrollLock-C70ZmxM0.js";var le=n((e=>{function t(e,t){var n=e.length;e.push(t);a:for(;0<n;){var r=n-1>>>1,a=e[r];if(0<i(a,t))e[r]=t,e[n]=a,n=r;else break a}}function n(e){return e.length===0?null:e[0]}function r(e){if(e.length===0)return null;var t=e[0],n=e.pop();if(n!==t){e[0]=n;a:for(var r=0,a=e.length,o=a>>>1;r<o;){var s=2*(r+1)-1,c=e[s],l=s+1,u=e[l];if(0>i(c,n))l<a&&0>i(u,c)?(e[r]=u,e[l]=n,r=l):(e[r]=c,e[s]=n,r=s);else if(l<a&&0>i(u,n))e[r]=u,e[l]=n,r=l;else break a}}return t}function i(e,t){var n=e.sortIndex-t.sortIndex;return n===0?e.id-t.id:n}if(e.unstable_now=void 0,typeof performance==`object`&&typeof performance.now==`function`){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var c=[],l=[],u=1,d=null,f=3,p=!1,m=!1,h=!1,g=!1,_=typeof setTimeout==`function`?setTimeout:null,v=typeof clearTimeout==`function`?clearTimeout:null,y=typeof setImmediate<`u`?setImmediate:null;function b(e){for(var i=n(l);i!==null;){if(i.callback===null)r(l);else if(i.startTime<=e)r(l),i.sortIndex=i.expirationTime,t(c,i);else break;i=n(l)}}function ee(e){if(h=!1,b(e),!m){if(n(c)!==null)m=!0,te||(te=!0,oe());else{var t=n(l);t!==null&&le(ee,t.startTime-e)}}}var te=!1,ne=-1,re=5,ie=-1;function x(){return g?!0:!(e.unstable_now()-ie<re)}function ae(){if(g=!1,te){var t=e.unstable_now();ie=t;var i=!0;try{a:{m=!1,h&&(h=!1,v(ne),ne=-1),p=!0;var a=f;try{b:{for(b(t),d=n(c);d!==null&&!(d.expirationTime>t&&x());){var o=d.callback;if(typeof o==`function`){d.callback=null,f=d.priorityLevel;var s=o(d.expirationTime<=t);if(t=e.unstable_now(),typeof s==`function`){d.callback=s,b(t),i=!0;break b}d===n(c)&&r(c),b(t)}else r(c);d=n(c)}if(d!==null)i=!0;else{var u=n(l);u!==null&&le(ee,u.startTime-t),i=!1}}break a}finally{d=null,f=a,p=!1}i=void 0}}finally{i?oe():te=!1}}}var oe;if(typeof y==`function`)oe=function(){y(ae)};else if(typeof MessageChannel<`u`){var se=new MessageChannel,ce=se.port2;se.port1.onmessage=ae,oe=function(){ce.postMessage(null)}}else oe=function(){_(ae,0)};function le(t,n){ne=_(function(){t(e.unstable_now())},n)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(e){e.callback=null},e.unstable_forceFrameRate=function(e){0>e||125<e?console.error(`forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported`):re=0<e?Math.floor(1e3/e):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_next=function(e){switch(f){case 1:case 2:case 3:var t=3;break;default:t=f}var n=f;f=t;try{return e()}finally{f=n}},e.unstable_requestPaint=function(){g=!0},e.unstable_runWithPriority=function(e,t){switch(e){case 1:case 2:case 3:case 4:case 5:break;default:e=3}var n=f;f=e;try{return t()}finally{f=n}},e.unstable_scheduleCallback=function(r,i,a){var o=e.unstable_now();switch(typeof a==`object`&&a?(a=a.delay,a=typeof a==`number`&&0<a?o+a:o):a=o,r){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=a+s,r={id:u++,callback:i,priorityLevel:r,startTime:a,expirationTime:s,sortIndex:-1},a>o?(r.sortIndex=a,t(l,r),n(c)===null&&r===n(l)&&(h?(v(ne),ne=-1):h=!0,le(ee,a-o))):(r.sortIndex=s,t(c,r),m||p||(m=!0,te||(te=!0,oe()))),r},e.unstable_shouldYield=x,e.unstable_wrapCallback=function(e){var t=f;return function(){var n=f;f=t;try{return e.apply(this,arguments)}finally{f=n}}}})),ue=n(((e,t)=>{t.exports=le()})),de=n((e=>{var n=ue(),r=t(),a=i();function o(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function s(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function c(e){for(var t=e,n=t;n&&!n.alternate;)t=n,t.flags&4098&&(e=t.return),n=t.return;for(;t.return;)t=t.return;return t.tag===3?e:null}function l(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function u(e){if(e.tag===31){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function d(e){if(c(e)!==e)throw Error(o(188))}function f(e){var t=e.alternate;if(!t){if(t=c(e),t===null)throw Error(o(188));return t===e?e:null}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return d(i),e;if(a===r)return d(i),t;a=a.sibling}throw Error(o(188))}if(n.return!==r.return)n=i,r=a;else{for(var s=!1,l=i.child;l;){if(l===n){s=!0,n=i,r=a;break}if(l===r){s=!0,r=i,n=a;break}l=l.sibling}if(!s){for(l=a.child;l;){if(l===n){s=!0,n=a,r=i;break}if(l===r){s=!0,r=a,n=i;break}l=l.sibling}if(!s)throw Error(o(189))}}if(n.alternate!==r)throw Error(o(190))}if(n.tag!==3)throw Error(o(188));return n.stateNode.current===n?e:t}function p(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e;for(e=e.child;e!==null;){if(t=p(e),t!==null)return t;e=e.sibling}return null}function m(e,t,n,r,i,a){for(;e!==null;){if((e.tag===5||e.tag===27||e.tag===6)&&n(e,r,i,a)||(e.tag!==22||e.memoizedState===null)&&(t||e.tag!==5&&e.tag!==27)&&m(e.child,t,n,r,i,a))return!0;e=e.sibling}return!1}function h(e){for(e=e.return;e!==null;){if(e.tag===3||e.tag===5||e.tag===27)return e;e=e.return}return null}function g(e){var t=!1;for(e=e.return;e!==null&&(e.tag===4&&(t=!0),e.tag!==3&&e.tag!==5&&e.tag!==27);)e=e.return;return t}function _(e){var t=[null,null],n=h(e);return n===null||v(t,e,n.child,{foundSelf:!1}),t}function v(e,t,n,r){for(;n!==null;){if(n===t)r.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(r.foundSelf)return e[1]=n,!0;e[0]=n}else if((n.tag!==22||n.memoizedState===null)&&v(e,t,n.child,r))return!0;n=n.sibling}return!1}function y(e){switch(e.tag){case 5:case 27:case 6:return e.stateNode;case 3:return e.stateNode.containerInfo;default:throw Error(o(559))}}var b=null,ee=null;function te(e,t,n){return e===n||e===t&&(b=e,!0)}function ne(e,t,n){return e===n?(ee=e,!1):e===t&&(ee!==null&&(b=e),!0)}function re(e){if(e===null)return null;do e=e===null?null:e.return;while(e&&e.tag!==5&&e.tag!==27&&e.tag!==3);return e||null}function ie(e,t,n){for(var r=0,i=e;i;i=n(i))r++;i=0;for(var a=t;a;a=n(a))i++;for(;0<r-i;)e=n(e),r--;for(;0<i-r;)t=n(t),i--;for(;r--;){if(e===t||t!==null&&e===t.alternate)return e;e=n(e),t=n(t)}return null}var x=Object.assign,ae=Symbol.for(`react.element`),oe=Symbol.for(`react.transitional.element`),se=Symbol.for(`react.portal`),ce=Symbol.for(`react.fragment`),le=Symbol.for(`react.strict_mode`),de=Symbol.for(`react.profiler`),fe=Symbol.for(`react.consumer`),S=Symbol.for(`react.context`),pe=Symbol.for(`react.forward_ref`),me=Symbol.for(`react.suspense`),he=Symbol.for(`react.suspense_list`),ge=Symbol.for(`react.memo`),_e=Symbol.for(`react.lazy`),ve=Symbol.for(`react.activity`),ye=Symbol.for(`react.legacy_hidden`),be=Symbol.for(`react.memo_cache_sentinel`),xe=Symbol.for(`react.view_transition`),Se=Symbol.for(`react.recoverable`),Ce=Symbol.iterator;function C(e){return typeof e!=`object`||!e?null:(e=Ce&&e[Ce]||e[`@@iterator`],typeof e==`function`?e:null)}var we=Symbol.for(`react.client.reference`);function Te(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===we?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case ce:return`Fragment`;case de:return`Profiler`;case le:return`StrictMode`;case me:return`Suspense`;case he:return`SuspenseList`;case ve:return`Activity`;case xe:return`ViewTransition`}if(typeof e==`object`)switch(e.$$typeof){case se:return`Portal`;case S:return e.displayName||`Context`;case fe:return(e._context.displayName||`Context`)+`.Consumer`;case pe:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case ge:return t=e.displayName||null,t===null?Te(e.type)||`Memo`:t;case _e:t=e._payload,e=e._init;try{return Te(e(t))}catch{}}return null}var Ee=Array.isArray,w=r.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,T=a.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,De={pending:!1,data:null,method:null,action:null},Oe=[],ke=-1;function Ae(e){return{current:e}}function je(e){0>ke||(e.current=Oe[ke],Oe[ke]=null,ke--)}function E(e,t){ke++,Oe[ke]=e.current,e.current=t}var Me=Ae(null),Ne=Ae(null),Pe=Ae(null),Fe=Ae(null);function Ie(e,t){switch(E(Pe,t),E(Ne,e),E(Me,null),t.nodeType){case 9:case 11:e=(e=t.documentElement)&&(e=e.namespaceURI)?up(e):0;break;default:if(e=t.tagName,t=t.namespaceURI)t=up(t),e=dp(t,e);else switch(e){case`svg`:e=1;break;case`math`:e=2;break;default:e=0}}je(Me),E(Me,e)}function Le(){je(Me),je(Ne),je(Pe)}function Re(e){var t=e.memoizedState;t!==null&&(sh._currentValue=t.memoizedState,E(Fe,e)),t=Me.current;var n=dp(t,e.type);t!==n&&(E(Ne,e),E(Me,n))}function ze(e){Ne.current===e&&(je(Me),je(Ne)),Fe.current===e&&(je(Fe),sh._currentValue=De)}var Be,Ve;function He(e){if(Be===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);Be=t&&t[1]||``,Ve=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+Be+e+Ve}var Ue=!1;function We(e,t){if(!e||Ue)return``;Ue=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}n=!1;try{var i=Object.getOwnPropertyDescriptor(e.prototype,`props`);Object.defineProperty(e.prototype,"props",{configurable:!0,set:function(){throw Error()}}),n=!0,new e}finally{n&&(i===void 0?delete e.prototype.props:Object.defineProperty(e.prototype,"props",i))}}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,"name",{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Ue=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?He(n):``}function Ge(e,t){switch(e.tag){case 26:case 27:case 5:return He(e.type);case 16:return He(`Lazy`);case 13:return e.child!==t&&t!==null?He(`Suspense Fallback`):He(`Suspense`);case 19:return He(`SuspenseList`);case 0:case 15:return We(e.type,!1);case 11:return We(e.type.render,!1);case 1:return We(e.type,!0);case 31:return He(`Activity`);case 30:return He(`ViewTransition`);default:return``}}function Ke(e){try{var t=``,n=null;do t+=Ge(e,n),n=e,e=e.return;while(e);return t}catch(e){return`
Error generating stack: `+e.message+`
`+e.stack}}var qe=Object.prototype.hasOwnProperty,Je=n.unstable_scheduleCallback,Ye=n.unstable_cancelCallback,Xe=n.unstable_shouldYield,Ze=n.unstable_requestPaint,Qe=n.unstable_now,$e=n.unstable_getCurrentPriorityLevel,et=n.unstable_ImmediatePriority,tt=n.unstable_UserBlockingPriority,nt=n.unstable_NormalPriority,rt=n.unstable_LowPriority,it=n.unstable_IdlePriority,at=n.log,ot=n.unstable_setDisableYieldValue,st=null,ct=null;function lt(e){if(typeof at==`function`&&ot(e),ct&&typeof ct.setStrictMode==`function`)try{ct.setStrictMode(st,e)}catch{}}var D=Math.clz32?Math.clz32:ft,ut=Math.log,dt=Math.LN2;function ft(e){return e>>>=0,e===0?32:31-(ut(e)/dt|0)|0}var pt=256,mt=262144,ht=4194304;function gt(e){var t=e&42;if(t!==0)return t;switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return e&-e;case 262144:case 524288:case 1048576:case 2097152:return e&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return e&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return e}}function _t(e,t,n){var r=e.pendingLanes;if(r===0)return 0;var i=0,a=e.suspendedLanes,o=e.pingedLanes;e=e.warmLanes;var s=r&134217727;return s===0?(s=r&~a,s===0?o===0?n||(n=r&~e,n!==0&&(i=gt(n))):i=gt(o):i=gt(s)):(r=s&~a,r===0?(o&=s,o===0?n||(n=s&~e,n!==0&&(i=gt(n))):i=gt(o)):i=gt(r)),i===0?0:t!==0&&t!==i&&(t&a)===0&&(a=i&-i,n=t&-t,a>=n||a===32&&n&4194048)?t:i}function vt(e,t){return(e.pendingLanes&~(e.suspendedLanes&~e.pingedLanes)&t)===0}function yt(e,t){t&8&&(t|=t&32);var n=e.entangledLanes;if(n!==0)for(e=e.entanglements,n&=t;0<n;){var r=31-D(n),i=1<<r;t|=e[r],n&=~i}return t}function bt(e,t){switch(e){case 1:case 2:case 4:case 8:case 64:return t+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function xt(){var e=ht;return ht<<=1,!(ht&62914560)&&(ht=4194304),e}function St(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function Ct(e,t){e.pendingLanes|=t,t!==268435456&&(e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0)}function wt(e,t,n,r,i,a){var o=e.pendingLanes;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0,e.expiredLanes&=n,e.entangledLanes&=n,e.errorRecoveryDisabledLanes&=n,e.shellSuspendCounter=0;var s=e.entanglements,c=e.expirationTimes,l=e.hiddenUpdates;for(n=o&~n;0<n;){var u=31-D(n),d=1<<u;s[u]=0,c[u]=-1;var f=l[u];if(f!==null)for(l[u]=null,u=0;u<f.length;u++){var p=f[u];p!==null&&(p.lane&=-536870913)}n&=~d}r!==0&&Tt(e,r,0),a!==0&&i===0&&e.tag!==0&&(e.suspendedLanes|=a&~(o&~t))}function Tt(e,t,n){e.pendingLanes|=t,e.suspendedLanes&=~t;var r=31-D(t);e.entangledLanes|=t,e.entanglements[r]=e.entanglements[r]|1073741824|n&261930}function Et(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-D(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}function Dt(e,t){var n=t&-t;return n=n&42?1:Ot(n),(n&(e.suspendedLanes|t))===0?n:0}function Ot(e){switch(e){case 2:e=1;break;case 8:e=4;break;case 32:e=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:e=128;break;case 268435456:e=134217728;break;default:e=0}return e}function kt(e){return e&=-e,2<e?8<e?e&134217727?32:268435456:8:2}function At(){var e=T.p;return e===0?(e=window.event,e===void 0?32:Ch(e.type)):e}function jt(e,t){var n=T.p;try{return T.p=e,t()}finally{T.p=n}}var Mt=Math.random().toString(36).slice(2),Nt=`__reactFiber$`+Mt,Pt=`__reactProps$`+Mt,Ft=`__reactContainer$`+Mt,It=`__reactEvents$`+Mt,Lt=`__reactListeners$`+Mt,Rt=`__reactHandles$`+Mt,zt=`__reactResources$`+Mt,Bt=`__reactMarker$`+Mt,Vt=`__reactLoad$`+Mt;function Ht(e){delete e[Nt],delete e[Pt],delete e[Lt],delete e[Rt]}function Ut(e){var t;if(t=e[Nt])return t;for(var n=e.parentNode;n;){if(t=n[Ft]||n[Nt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=fm(e);e!==null;){if(n=e[Nt])return n;e=fm(e)}return t}e=n,n=e.parentNode}return null}function Wt(e){if(e=e[Nt]||e[Ft]){var t=e.tag;if(t===5||t===6||t===13||t===31||t===26||t===27||t===3)return e}return null}function Gt(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e.stateNode;throw Error(o(33))}function Kt(e){var t=e[zt];return t||=e[zt]={hoistableStyles:new Map,hoistableScripts:new Map},t}function O(e){e[Bt]=!0}function qt(e){e[Vt]=void 0}var Jt=new Set,Yt={};function Xt(e,t){Zt(e,t),Zt(e+`Capture`,t)}function Zt(e,t){for(Yt[e]=t,e=0;e<t.length;e++)Jt.add(t[e])}var Qt=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),$t={},en={};function tn(e){return qe.call(en,e)?!0:qe.call($t,e)?!1:Qt.test(e)?en[e]=!0:($t[e]=!0,!1)}var k=!1;function nn(){var e=k;return k=!1,e}function rn(e,t,n){if(tn(t)){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:e.removeAttribute(t);return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`){e.removeAttribute(t);return}}e.setAttribute(t,n)}}}function an(e,t,n){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(t);return}e.setAttribute(t,n)}}function on(e,t,n,r){if(r===null)e.removeAttribute(n);else{switch(typeof r){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(n);return}e.setAttributeNS(t,n,r)}}function sn(e){switch(typeof e){case`bigint`:case`boolean`:case`number`:case`string`:case`undefined`:return e;case`object`:return e;default:return``}}function cn(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()===`input`&&(t===`checkbox`||t===`radio`)}function ln(e,t,n){var r=Object.getOwnPropertyDescriptor(e.constructor.prototype,t);if(!e.hasOwnProperty(t)&&r!==void 0&&typeof r.get==`function`&&typeof r.set==`function`){var i=r.get,a=r.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(e){n=``+e,a.call(this,e)}}),Object.defineProperty(e,t,{enumerable:r.enumerable}),{getValue:function(){return n},setValue:function(e){n=``+e},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function un(e){if(!e._valueTracker){var t=cn(e)?`checked`:`value`;e._valueTracker=ln(e,t,``+e[t])}}function dn(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r=``;return e&&(r=cn(e)?e.checked?`true`:`false`:e.value),e=r,e!==n&&(t.setValue(e),!0)}var fn=/[\n"\\]/g;function pn(e){return e.replace(fn,function(e){return`\\`+e.charCodeAt(0).toString(16)+` `})}function mn(e,t,n,r,i,a,o,s){e.name=``,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`?e.type=o:e.removeAttribute(`type`),t==null?o!==`submit`&&o!==`reset`||e.removeAttribute(`value`):o===`number`?(t===0&&e.value===``||e.value!=t)&&(e.value=``+sn(t)):e.value!==``+sn(t)&&(e.value=``+sn(t)),t==null?n==null?r!=null&&e.removeAttribute(`value`):gn(e,sn(n)):o===`number`&&e.value==t?gn(e,sn(e.value)):gn(e,sn(t)),i==null&&a!=null&&(e.defaultChecked=!!a),i!=null&&(e.checked=i&&typeof i!=`function`&&typeof i!=`symbol`),s!=null&&typeof s!=`function`&&typeof s!=`symbol`&&typeof s!=`boolean`?e.name=``+sn(s):e.removeAttribute(`name`)}function hn(e,t,n,r,i,a,o,s){if(a!=null&&typeof a!=`function`&&typeof a!=`symbol`&&typeof a!=`boolean`&&(e.type=a),t!=null||n!=null){if(!(a!==`submit`&&a!==`reset`||t!=null)){un(e);return}n=n==null?``:``+sn(n),t=t==null?n:``+sn(t),s||t===e.value||(e.value=t),e.defaultValue=t}r??=i,r=typeof r!=`function`&&typeof r!=`symbol`&&!!r,e.checked=s?e.checked:!!r,e.defaultChecked=!!r,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`&&(e.name=o),un(e)}function gn(e,t){e.defaultValue!==``+t&&(e.defaultValue=``+t)}function _n(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t[`$`+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty(`$`+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=``+sn(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function vn(e,t,n){if(t!=null&&(t=``+sn(t),t!==e.value&&(e.value=t),n==null)){e.defaultValue!==t&&(e.defaultValue=t);return}e.defaultValue=n==null?``:``+sn(n)}function yn(e,t,n,r){if(t==null){if(r!=null){if(n!=null)throw Error(o(92));if(Ee(r)){if(1<r.length)throw Error(o(93));r=r[0]}n=r}n??=``,t=n}n=sn(t),e.defaultValue=n,r=e.textContent,r===n&&r!==``&&r!==null&&(e.value=r),un(e)}function bn(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var xn=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `));function Sn(e,t,n){var r=t.indexOf(`--`)===0;n==null||typeof n==`boolean`||n===``?r?e.setProperty(t,``):t===`float`?e.cssFloat=``:e[t]=``:r?e.setProperty(t,n):typeof n!=`number`||n===0||xn.has(t)?t===`float`?e.cssFloat=n:e[t]=(``+n).trim():e[t]=n+`px`}function Cn(e,t,n){if(t!=null&&typeof t!=`object`)throw Error(o(62));if(e=e.style,n!=null){for(var r in n)!n.hasOwnProperty(r)||t!=null&&t.hasOwnProperty(r)||(r.indexOf(`--`)===0?e.setProperty(r,``):r===`float`?e.cssFloat=``:e[r]=``,k=!0);for(var i in t)r=t[i],t.hasOwnProperty(i)&&n[i]!==r&&(Sn(e,i,r),k=!0)}else for(var a in t)t.hasOwnProperty(a)&&Sn(e,a,t[a])}function wn(e){if(e.indexOf(`-`)===-1)return!1;switch(e){case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:return!1;default:return!0}}var Tn=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`maskType`,`mask-type`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),En=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function Dn(e){return En.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}function On(){}var kn=null;function An(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var jn=null,Mn=null;function Nn(e){var t=Wt(e);if(t&&(e=t.stateNode)){var n=e[Pt]||null;a:switch(e=t.stateNode,t.type){case`input`:if(mn(e,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),t=n.name,n.type===`radio`&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll(`input[name="`+pn(``+t)+`"][type="radio"]`),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=r[Pt]||null;if(!i)throw Error(o(90));mn(r,i.value,i.defaultValue,i.defaultValue,i.checked,i.defaultChecked,i.type,i.name)}}for(t=0;t<n.length;t++)r=n[t],r.form===e.form&&dn(r)}break a;case`textarea`:vn(e,n.value,n.defaultValue);break a;case`select`:t=n.value,t!=null&&_n(e,!!n.multiple,t,!1)}}}var Pn=!1;function Fn(e,t,n){if(Pn)return e(t,n);Pn=!0;try{return e(t)}finally{if(Pn=!1,(jn!==null||Mn!==null)&&(zd(),jn&&(t=jn,e=Mn,Mn=jn=null,Nn(t),e)))for(t=0;t<e.length;t++)Nn(e[t])}}function In(e,t){var n=e.stateNode;if(n===null)return null;var r=n[Pt]||null;if(r===null)return null;n=r[t];a:switch(t){case`onClick`:case`onClickCapture`:case`onDoubleClick`:case`onDoubleClickCapture`:case`onMouseDown`:case`onMouseDownCapture`:case`onMouseMove`:case`onMouseMoveCapture`:case`onMouseUp`:case`onMouseUpCapture`:case`onMouseEnter`:(r=!r.disabled)||(e=e.type,r=e!==`button`&&e!==`input`&&e!==`select`&&e!==`textarea`),e=!r;break a;default:e=!1}if(e)return null;if(n&&typeof n!=`function`)throw Error(o(231,t,typeof n));return n}var Ln=!(typeof window>`u`||window.document===void 0||window.document.createElement===void 0),Rn=!1;if(Ln)try{var zn={};Object.defineProperty(zn,"passive",{get:function(){Rn=!0}}),window.addEventListener(`test`,zn,zn),window.removeEventListener(`test`,zn,zn)}catch{Rn=!1}var Bn=null,Vn=null,Hn=null;function Un(){if(Hn)return Hn;var e,t=Vn,n=t.length,r,i=`value`in Bn?Bn.value:Bn.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Hn=i.slice(e,1<r?1-r:void 0)}function Wn(e){var t=e.keyCode;return`charCode`in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Gn(){return!0}function Kn(){return!1}function qn(e){function t(t,n,r,i,a){for(var o in this._reactName=t,this._targetInst=r,this.type=n,this.nativeEvent=i,this.target=a,this.currentTarget=null,e)e.hasOwnProperty(o)&&(t=e[o],this[o]=t?t(i):i[o]);return this.isDefaultPrevented=(i.defaultPrevented==null?!1===i.returnValue:i.defaultPrevented)?Gn:Kn,this.isPropagationStopped=Kn,this}return x(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var e=this.nativeEvent;e&&(e.preventDefault?e.preventDefault():typeof e.returnValue!=`unknown`&&(e.returnValue=!1),this.isDefaultPrevented=Gn)},stopPropagation:function(){var e=this.nativeEvent;e&&(e.stopPropagation?e.stopPropagation():typeof e.cancelBubble!=`unknown`&&(e.cancelBubble=!0),this.isPropagationStopped=Gn)},persist:function(){},isPersistent:Gn}),t}var Jn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Yn=qn(Jn),Xn=x({},Jn,{view:0,detail:0}),Zn=qn(Xn),Qn,$n,er,tr=x({},Xn,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:fr,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return`movementX`in e?e.movementX:(e!==er&&(er&&e.type===`mousemove`?(Qn=e.screenX-er.screenX,$n=e.screenY-er.screenY):$n=Qn=0,er=e),Qn)},movementY:function(e){return`movementY`in e?e.movementY:$n}}),nr=qn(tr),rr=qn(x({},tr,{dataTransfer:0})),ir=qn(x({},Xn,{relatedTarget:0})),ar=qn(x({},Jn,{animationName:0,elapsedTime:0,pseudoElement:0})),or=qn(x({},Jn,{clipboardData:function(e){return`clipboardData`in e?e.clipboardData:window.clipboardData}})),sr=qn(x({},Jn,{data:0})),cr={Esc:`Escape`,Spacebar:` `,Left:`ArrowLeft`,Up:`ArrowUp`,Right:`ArrowRight`,Down:`ArrowDown`,Del:`Delete`,Win:`OS`,Menu:`ContextMenu`,Apps:`ContextMenu`,Scroll:`ScrollLock`,MozPrintableKey:`Unidentified`},lr={8:`Backspace`,9:`Tab`,12:`Clear`,13:`Enter`,16:`Shift`,17:`Control`,18:`Alt`,19:`Pause`,20:`CapsLock`,27:`Escape`,32:` `,33:`PageUp`,34:`PageDown`,35:`End`,36:`Home`,37:`ArrowLeft`,38:`ArrowUp`,39:`ArrowRight`,40:`ArrowDown`,45:`Insert`,46:`Delete`,112:`F1`,113:`F2`,114:`F3`,115:`F4`,116:`F5`,117:`F6`,118:`F7`,119:`F8`,120:`F9`,121:`F10`,122:`F11`,123:`F12`,144:`NumLock`,145:`ScrollLock`,224:`Meta`},ur={Alt:`altKey`,Control:`ctrlKey`,Meta:`metaKey`,Shift:`shiftKey`};function dr(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=ur[e])?!!t[e]:!1}function fr(){return dr}var pr=qn(x({},Xn,{key:function(e){if(e.key){var t=cr[e.key]||e.key;if(t!==`Unidentified`)return t}return e.type===`keypress`?(e=Wn(e),e===13?`Enter`:String.fromCharCode(e)):e.type===`keydown`||e.type===`keyup`?lr[e.keyCode]||`Unidentified`:``},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:fr,charCode:function(e){return e.type===`keypress`?Wn(e):0},keyCode:function(e){return e.type===`keydown`||e.type===`keyup`?e.keyCode:0},which:function(e){return e.type===`keypress`?Wn(e):e.type===`keydown`||e.type===`keyup`?e.keyCode:0}})),mr=qn(x({},tr,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0})),hr=qn(x({},Jn,{submitter:0})),gr=qn(x({},Xn,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:fr})),_r=qn(x({},Jn,{propertyName:0,elapsedTime:0,pseudoElement:0})),vr=qn(x({},tr,{deltaX:function(e){return`deltaX`in e?e.deltaX:`wheelDeltaX`in e?-e.wheelDeltaX:0},deltaY:function(e){return`deltaY`in e?e.deltaY:`wheelDeltaY`in e?-e.wheelDeltaY:`wheelDelta`in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0})),yr=qn(x({},Jn,{newState:0,oldState:0,source:0})),br=[9,13,27,32],xr=Ln&&`CompositionEvent`in window,Sr=null;Ln&&`documentMode`in document&&(Sr=document.documentMode);var Cr=Ln&&`TextEvent`in window&&!Sr,wr=Ln&&(!xr||Sr&&8<Sr&&11>=Sr),Tr=` `,Er=!1;function Dr(e,t){switch(e){case`keyup`:return br.indexOf(t.keyCode)!==-1;case`keydown`:return t.keyCode!==229;case`keypress`:case`mousedown`:case`focusout`:return!0;default:return!1}}function Or(e){return e=e.detail,typeof e==`object`&&`data`in e?e.data:null}var kr=!1;function Ar(e,t){switch(e){case`compositionend`:return Or(t);case`keypress`:return t.which===32?(Er=!0,Tr):null;case`textInput`:return e=t.data,e===Tr&&Er?null:e;default:return null}}function jr(e,t){if(kr)return e===`compositionend`||!xr&&Dr(e,t)?(e=Un(),Hn=Vn=Bn=null,kr=!1,e):null;switch(e){case`paste`:return null;case`keypress`:if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case`compositionend`:return wr&&t.locale!==`ko`?null:t.data;default:return null}}var Mr={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Nr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t===`input`?!!Mr[e.type]:t===`textarea`}function Pr(e,t,n,r){jn?Mn?Mn.push(r):Mn=[r]:jn=r,t=Jf(t,`onChange`),0<t.length&&(n=new Yn(`onChange`,`change`,null,n,r),e.push({event:n,listeners:t}))}var Fr=null,A=null;function Ir(e){Vf(e,0)}function Lr(e){if(dn(Gt(e)))return e}function Rr(e,t){if(e===`change`)return t}var zr=!1;if(Ln){var Br;if(Ln){var Vr=`oninput`in document;if(!Vr){var Hr=document.createElement(`div`);Hr.setAttribute(`oninput`,`return;`),Vr=typeof Hr.oninput==`function`}Br=Vr}else Br=!1;zr=Br&&(!document.documentMode||9<document.documentMode)}function Ur(){Fr&&(Fr.detachEvent(`onpropertychange`,Wr),A=Fr=null)}function Wr(e){if(e.propertyName===`value`&&Lr(A)){var t=[];Pr(t,A,e,An(e)),Fn(Ir,t)}}function Gr(e,t,n){e===`focusin`?(Ur(),Fr=t,A=n,Fr.attachEvent(`onpropertychange`,Wr)):e===`focusout`&&Ur()}function Kr(e){if(e===`selectionchange`||e===`keyup`||e===`keydown`)return Lr(A)}function qr(e,t){if(e===`click`)return Lr(t)}function Jr(e,t){if(e===`input`||e===`change`)return Lr(t)}function Yr(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var Xr=typeof Object.is==`function`?Object.is:Yr;function Zr(e,t){if(Xr(e,t))return!0;if(typeof e!=`object`||!e||typeof t!=`object`||!t)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!qe.call(t,i)||!Xr(e[i],t[i]))return!1}return!0}function Qr(e){if(e||=typeof document<`u`?document:void 0,e===void 0)return null;try{return e.activeElement||e.body}catch{return e.body}}function $r(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function ei(e,t){var n=$r(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}a:{for(;n;){if(n.nextSibling){n=n.nextSibling;break a}n=n.parentNode}n=void 0}n=$r(n)}}function ti(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?ti(e,t.parentNode):`contains`in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function ni(e){e=e!=null&&e.ownerDocument!=null&&e.ownerDocument.defaultView!=null?e.ownerDocument.defaultView:window;for(var t=Qr(e.document);t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href==`string`}catch{n=!1}if(n)e=t.contentWindow;else break;t=Qr(e.document)}return t}function ri(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t===`input`&&(e.type===`text`||e.type===`search`||e.type===`tel`||e.type===`url`||e.type===`password`)||t===`textarea`||e.contentEditable===`true`)}var ii=Ln&&`documentMode`in document&&11>=document.documentMode,ai=null,oi=null,si=null,ci=!1;function li(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;ci||ai==null||ai!==Qr(r)||(r=ai,`selectionStart`in r&&ri(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),si&&Zr(si,r)||(si=r,r=Jf(oi,`onSelect`),0<r.length&&(t=new Yn(`onSelect`,`select`,null,t,n),e.push({event:t,listeners:r}),t.target=ai)))}function ui(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n[`Webkit`+e]=`webkit`+t,n[`Moz`+e]=`moz`+t,n}var di={animationend:ui(`Animation`,`AnimationEnd`),animationiteration:ui(`Animation`,`AnimationIteration`),animationstart:ui(`Animation`,`AnimationStart`),transitionrun:ui(`Transition`,`TransitionRun`),transitionstart:ui(`Transition`,`TransitionStart`),transitioncancel:ui(`Transition`,`TransitionCancel`),transitionend:ui(`Transition`,`TransitionEnd`)},fi={},pi={};Ln&&(pi=document.createElement(`div`).style,`AnimationEvent`in window||(delete di.animationend.animation,delete di.animationiteration.animation,delete di.animationstart.animation),`TransitionEvent`in window||delete di.transitionend.transition);function mi(e){if(fi[e])return fi[e];if(!di[e])return e;var t=di[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in pi)return fi[e]=t[n];return e}var hi=mi(`animationend`),gi=mi(`animationiteration`),_i=mi(`animationstart`),vi=mi(`transitionrun`),yi=mi(`transitionstart`),bi=mi(`transitioncancel`),xi=mi(`transitionend`),Si=new Map,Ci=`abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel`.split(` `);Ci.push(`scrollEnd`);function wi(e,t){Si.set(e,t),Xt(t,[e])}var Ti=0;function Ei(e,t){if(e.name!=null&&e.name!==`auto`)return e.name;if(t.autoName!==null)return t.autoName;e=bd.identifierPrefix;var n=Ti++;return e=`_`+e+`t_`+n.toString(32)+`_`,t.autoName=e}function Di(e){if(e==null||typeof e==`string`)return e;var t=null,n=Od;if(n!==null)for(var r=0;r<n.length;r++){var i=e[n[r]];if(i!=null){if(i===`none`)return`none`;t=t==null?i:t+(` `+i)}}return t??e.default}function Oi(e,t){return e=Di(e),t=Di(t),t==null?e===`auto`?null:e:t===`auto`?null:t}var ki=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)},Ai=[],ji=0,Mi=0;function Ni(){for(var e=ji,t=Mi=ji=0;t<e;){var n=Ai[t];Ai[t++]=null;var r=Ai[t];Ai[t++]=null;var i=Ai[t];Ai[t++]=null;var a=Ai[t];if(Ai[t++]=null,r!==null&&i!==null){var o=r.pending;o===null?i.next=i:(i.next=o.next,o.next=i),r.pending=i}a!==0&&Li(n,i,a)}}function Pi(e,t,n,r){Ai[ji++]=e,Ai[ji++]=t,Ai[ji++]=n,Ai[ji++]=r,Mi|=r,e.lanes|=r,e=e.alternate,e!==null&&(e.lanes|=r)}function Fi(e,t,n,r){return Pi(e,t,n,r),Ri(e)}function Ii(e,t){return Pi(e,null,null,t),Ri(e)}function Li(e,t,n){e.lanes|=n;var r=e.alternate;r!==null&&(r.lanes|=n);for(var i=!1,a=e.return;a!==null;)a.childLanes|=n,r=a.alternate,r!==null&&(r.childLanes|=n),a.tag===22&&(e=a.stateNode,e===null||e._visibility&1||(i=!0)),e=a,a=a.return;return e.tag===3?(a=e.stateNode,i&&t!==null&&(i=31-D(n),e=a.hiddenUpdates,r=e[i],r===null?e[i]=[t]:r.push(t),t.lane=n|536870912),a):null}function Ri(e){if(50<kd)throw kd=0,Ad=null,Error(o(185));for(var t=e.return;t!==null;)e=t,t=e.return;return e.tag===3?e.stateNode:null}var zi={};function Bi(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Vi(e,t,n,r){return new Bi(e,t,n,r)}function Hi(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Ui(e,t){var n=e.alternate;return n===null?(n=Vi(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&1206910976,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n.refCleanup=e.refCleanup,n}function Wi(e,t){e.flags&=1206910978;var n=e.alternate;return n===null?(e.childLanes=0,e.lanes=t,e.child=null,e.subtreeFlags=0,e.memoizedProps=null,e.memoizedState=null,e.updateQueue=null,e.dependencies=null,e.stateNode=null):(e.childLanes=n.childLanes,e.lanes=n.lanes,e.child=n.child,e.subtreeFlags=0,e.deletions=null,e.memoizedProps=n.memoizedProps,e.memoizedState=n.memoizedState,e.updateQueue=n.updateQueue,e.type=n.type,t=n.dependencies,e.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),e}function Gi(e,t,n,r,i,a){var s=0;if(r=e,typeof r==`function`)Hi(r)&&(s=1);else if(typeof r==`string`)s=qm(e,n,Me.current)?26:e===`html`||e===`head`||e===`body`?27:5;else a:switch(r){case ve:return e=Vi(31,n,t,i),e.elementType=ve,e.lanes=a,e;case ce:return Ki(n.children,i,a,t);case le:s=8,i|=24;break;case de:return e=Vi(12,n,t,i|2),e.elementType=de,e.lanes=a,e;case me:return e=Vi(13,n,t,i),e.elementType=me,e.lanes=a,e;case he:return e=Vi(19,n,t,i),e.elementType=he,e.lanes=a,e;case ye:case xe:return e=i|32,e=Vi(30,n,t,e),e.elementType=xe,e.lanes=a,e.stateNode={autoName:null,paired:null,clones:null,ref:null},e;default:if(typeof r==`object`&&r)switch(r.$$typeof){case S:s=10;break a;case fe:s=9;break a;case pe:s=11;break a;case ge:s=14;break a;case _e:s=16,r=null;break a}s=29,n=Error(o(130,e===null?`null`:typeof e,``)),r=null}return t=Vi(s,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function Ki(e,t,n,r){return e=Vi(7,e,r,t),e.lanes=n,e}function qi(e,t,n){return e=Vi(6,e,null,t),e.lanes=n,e}function Ji(e){var t=Vi(18,null,null,0);return t.stateNode=e,t}function Yi(e,t,n){return t=Vi(4,e.children===null?[]:e.children,e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}var Xi=new WeakMap;function Zi(e,t){if(typeof e==`object`&&e){var n=Xi.get(e);return n===void 0?(t={value:e,source:t,stack:Ke(t)},Xi.set(e,t),t):n}return{value:e,source:t,stack:Ke(t)}}var Qi=[],$i=0,ea=null,ta=0,na=[],ra=0,ia=null,aa=1,oa=``;function sa(e,t){Qi[$i++]=ta,Qi[$i++]=ea,ea=e,ta=t}function ca(e,t,n){na[ra++]=aa,na[ra++]=oa,na[ra++]=ia,ia=e;var r=aa;e=oa;var i=32-D(r)-1;r&=~(1<<i),n+=1;var a=32-D(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,aa=1<<32-D(t)+i|n<<i|r,oa=a+e}else aa=1<<a|n<<i|r,oa=e}function la(e){e.return!==null&&(sa(e,1),ca(e,1,0))}function ua(e){for(;e===ea;)ea=Qi[--$i],Qi[$i]=null,ta=Qi[--$i],Qi[$i]=null;for(;e===ia;)ia=na[--ra],na[ra]=null,oa=na[--ra],na[ra]=null,aa=na[--ra],na[ra]=null}function da(e,t){na[ra++]=aa,na[ra++]=oa,na[ra++]=ia,aa=t.id,oa=t.overflow,ia=e}var fa=null,j=null,M=!1,pa=null,ma=!1,ha=Error(o(519));function ga(e){throw Sa(Zi(Error(o(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?`text`:`HTML`,``)),e)),ha}function _a(e){var t=e.stateNode,n=e.type,r=e.memoizedProps;switch(t[Nt]=e,t[Pt]=r,n){case`dialog`:Q(`cancel`,t),Q(`close`,t);break;case`iframe`:case`object`:case`embed`:Q(`load`,t);break;case`video`:case`audio`:for(n=0;n<zf.length;n++)Q(zf[n],t);break;case`source`:Q(`error`,t);break;case`img`:case`image`:case`link`:Q(`error`,t),Q(`load`,t);break;case`details`:Q(`toggle`,t);break;case`input`:Q(`invalid`,t),hn(t,r.value,r.defaultValue,r.checked,r.defaultChecked,r.type,r.name,!0);break;case`select`:Q(`invalid`,t);break;case`textarea`:Q(`invalid`,t),yn(t,r.value,r.defaultValue,r.children)}n=r.children,typeof n!=`string`&&typeof n!=`number`&&typeof n!=`bigint`||t.textContent===``+n||!0===r.suppressHydrationWarning||ep(t.textContent,n)?(r.popover!=null&&(Q(`beforetoggle`,t),Q(`toggle`,t)),r.onScroll!=null&&Q(`scroll`,t),r.onScrollEnd!=null&&Q(`scrollend`,t),r.onClick!=null&&(t.onclick=On),t=!0):t=!1,t||ga(e,!0)}function va(e){for(fa=e.return;fa;)switch(fa.tag){case 5:case 31:case 13:ma=!1;return;case 27:case 3:ma=!0;return;default:fa=fa.return}}function ya(e){if(e!==fa)return!1;if(!M)return va(e),M=!0,!1;var t=e.tag,n;if((n=t!==3&&t!==27)&&((n=t===5)&&(n=e.type,n=n===`form`||n===`button`||pp(e.type,e.memoizedProps)),n=!n),n&&j&&ga(e),va(e),t===13){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(317));j=dm(e)}else if(t===31){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(317));j=dm(e)}else t===27?(t=j,Sp(e.type)?(e=um,um=null,j=e):j=t):j=fa?lm(e.stateNode.nextSibling):null;return!0}function ba(){j=fa=null,M=!1}function xa(){var e=pa;return e!==null&&(pd===null?pd=e:pd.push.apply(pd,e),pa=null),e}function Sa(e){pa===null?pa=[e]:pa.push(e)}var Ca=Ae(null),wa=null,Ta=null;function Ea(e,t,n){E(Ca,t._currentValue),t._currentValue=n}function Da(e){e._currentValue=Ca.current,je(Ca)}function Oa(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)===t?r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t):(e.childLanes|=t,r!==null&&(r.childLanes|=t)),e===n)break;e=e.return}}function ka(e,t,n,r){var i=e.child;for(i!==null&&(i.return=e);i!==null;){var a=i.dependencies;if(a!==null){var s=i.child;a=a.firstContext;a:for(;a!==null;){var c=a;a=i;for(var l=0;l<t.length;l++)if(c.context===t[l]){a.lanes|=n,c=a.alternate,c!==null&&(c.lanes|=n),Oa(a.return,n,e),r||(s=null);break a}a=c.next}}else if(i.tag===18){if(s=i.return,s===null)throw Error(o(341));s.lanes|=n,a=s.alternate,a!==null&&(a.lanes|=n),Oa(s,n,e),s=null}else i.tag===13&&i.memoizedState!==null&&i.memoizedState.dehydrated===null?(i.lanes|=n,s=i.alternate,s!==null&&(s.lanes|=n),Oa(i.return,n,e),s=i.child,s=s===null?null:s.sibling):s=i.child;if(s!==null)s.return=i;else for(s=i;s!==null;){if(s===e){s=null;break}if(i=s.sibling,i!==null){i.return=s.return,s=i;break}s=s.return}i=s}}function Aa(e,t,n,r){e=null;for(var i=t,a=!1;i!==null;){if(!a){if(i.flags&524288)a=!0;else if(i.flags&262144)break}if(i.tag===10){var s=i.alternate;if(s===null)throw Error(o(387));if(s=s.memoizedProps,s!==null){var c=i.type;Xr(i.pendingProps.value,s.value)||(e===null?e=[c]:e.push(c))}}else if(i===Fe.current){if(s=i.alternate,s===null)throw Error(o(387));s.memoizedState.memoizedState!==i.memoizedState.memoizedState&&(e===null?e=[sh]:e.push(sh))}i=i.return}return e!==null&&ka(t,e,n,r),t.flags|=262144,e!==null}function N(e){for(e=e.firstContext;e!==null;){if(!Xr(e.context._currentValue,e.memoizedValue))return!0;e=e.next}return!1}function ja(e){wa=e,Ta=null,e=e.dependencies,e!==null&&(e.firstContext=null)}function Ma(e){return Pa(wa,e)}function Na(e,t){return wa===null&&ja(e),Pa(e,t)}function Pa(e,t){var n=t._currentValue;if(t={context:t,memoizedValue:n,next:null},Ta===null){if(e===null)throw Error(o(308));Ta=t,e.dependencies={lanes:0,firstContext:t},e.flags|=524288}else Ta=Ta.next=t;return n}var Fa=typeof AbortController<`u`?AbortController:function(){var e=[],t=this.signal={aborted:!1,addEventListener:function(t,n){e.push(n)}};this.abort=function(){t.aborted=!0,e.forEach(function(e){return e()})}},Ia=n.unstable_scheduleCallback,La=n.unstable_NormalPriority,Ra={$$typeof:S,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function za(){return{controller:new Fa,data:new Map,refCount:0}}function Ba(e){e.refCount--,e.refCount===0&&Ia(La,function(){e.controller.abort()})}function Va(e,t){if(e.pendingLanes&4194048){var n=e.transitionTypes;for(n===null&&(n=e.transitionTypes=[]),e=0;e<t.length;e++){var r=t[e];n.indexOf(r)===-1&&n.push(r)}}}var Ha=null;function Ua(e){var t=e.transitionTypes;return e.transitionTypes=null,t}var Wa=null,Ga=0,Ka=0,qa=null;function Ja(e,t){if(Wa===null){var n=Wa=[];Ga=0,Ka=Pf(),qa={status:`pending`,value:void 0,then:function(e){n.push(e)}}}return Ga++,t.then(Ya,Ya),t}function Ya(){if(--Ga===0&&(Ha=null,Wa!==null)){qa!==null&&(qa.status=`fulfilled`);var e=Wa;Wa=null,Ka=0,qa=null;for(var t=0;t<e.length;t++)(0,e[t])()}}function Xa(e,t){var n=[],r={status:`pending`,value:null,reason:null,then:function(e){n.push(e)}};return e.then(function(){r.status=`fulfilled`,r.value=t;for(var e=0;e<n.length;e++)(0,n[e])(t)},function(e){for(r.status=`rejected`,r.reason=e,e=0;e<n.length;e++)(0,n[e])(void 0)}),r}var Za=w.S;w.S=function(e,t){if(gd=Qe(),typeof t==`object`&&t&&typeof t.then==`function`&&Ja(e,t),Ha!==null)for(var n=bf;n!==null;)Va(n,Ha),n=n.next;if(n=e.types,n!==null){for(var r=bf;r!==null;)Va(r,n),r=r.next;if(Ka!==0){r=Ha,r===null&&(r=Ha=[]);for(var i=0;i<n.length;i++){var a=n[i];r.indexOf(a)===-1&&r.push(a)}}}Za!==null&&Za(e,t)};var Qa=Ae(null);function $a(){var e=Qa.current;return e===null?G.pooledCache:e}function eo(e,t){t===null?E(Qa,Qa.current):E(Qa,t.pool)}function to(){var e=$a();return e===null?null:{parent:Ra._currentValue,pool:e}}var no=Error(o(460)),ro=Error(o(474)),io=Error(o(542)),ao={then:function(){}};function oo(e){return e=e.status,e===`fulfilled`||e===`rejected`}function so(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(On,On),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,fo(e),e===void 0&&!(`reason`in t)?Error(o(600)):e;default:if(typeof t.status==`string`)t.then(On,On);else{if(e=G,e!==null&&100<e.shellSuspendCounter)throw Error(o(482));e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})}switch(t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,fo(e),e}throw lo=t,no}}function co(e){try{var t=e._init;return t(e._payload)}catch(e){throw typeof e==`object`&&e&&typeof e.then==`function`?(lo=e,no):e}}var lo=null;function uo(){if(lo===null)throw Error(o(459));var e=lo;return lo=null,e}function fo(e){if(e===no||e===io)throw Error(o(483))}var po=null,mo=0;function ho(e){var t=mo;return mo+=1,po===null&&(po=[]),so(po,e,t)}function go(e,t){t=t.props.ref,e.ref=t===void 0?null:t}function _o(e,t){throw t.$$typeof===ae?Error(o(525)):(e=Object.prototype.toString.call(t),Error(o(31,e===`[object Object]`?`object with keys {`+Object.keys(t).join(`, `)+`}`:e)))}function vo(e){function t(t,n){if(e){var r=t.deletions;r===null?(t.deletions=[n],t.flags|=16):r.push(n)}}function n(n,r){if(!e)return null;for(;r!==null;)t(n,r),r=r.sibling;return null}function r(e){for(var t=new Map;e!==null;)e.key===null?t.set(e.index,e):t.set(e.key,e),e=e.sibling;return t}function i(e,t){return e=Ui(e,t),e.index=0,e.sibling=null,e}function a(t,n,r){return t.index=r,e?(r=t.alternate,r===null?(t.flags|=134217730,n):(r=r.index,r<n?(t.flags|=2,n):r)):(t.flags|=1048576,n)}function s(t){return e&&t.alternate===null&&(t.flags|=134217730),t}function c(e,t,n,r){return t===null||t.tag!==6?(t=qi(n,e.mode,r),t.return=e,t):(t=i(t,n),t.return=e,t)}function l(e,t,n,r){var a=n.type;return a===ce?(e=d(e,t,n.props.children,r,n.key),go(e,n),e):t!==null&&(t.elementType===a||typeof a==`object`&&a&&a.$$typeof===_e&&co(a)===t.type)?(t=i(t,n.props),go(t,n),t.return=e,t):(t=Gi(n.type,n.key,n.props,null,e.mode,r),go(t,n),t.return=e,t)}function u(e,t,n,r){return t===null||t.tag!==4||t.stateNode.containerInfo!==n.containerInfo||t.stateNode.implementation!==n.implementation?(t=Yi(n,e.mode,r),t.return=e,t):(t=i(t,n.children||[]),t.return=e,t)}function d(e,t,n,r,a){return t===null||t.tag!==7?(t=Ki(n,e.mode,r,a),t.return=e,t):(t=i(t,n),t.return=e,t)}function f(e,t,n){if(typeof t==`string`&&t!==``||typeof t==`number`||typeof t==`bigint`)return t=qi(``+t,e.mode,n),t.return=e,t;if(typeof t==`object`&&t){switch(t.$$typeof){case oe:return n=Gi(t.type,t.key,t.props,null,e.mode,n),go(n,t),n.return=e,n;case se:return t=Yi(t,e.mode,n),t.return=e,t;case _e:return t=co(t),f(e,t,n)}if(Ee(t)||C(t))return t=Ki(t,e.mode,n,null),t.return=e,t;if(typeof t.then==`function`)return f(e,ho(t),n);if(t.$$typeof===S)return f(e,Na(e,t),n);_o(e,t)}return null}function p(e,t,n,r){var i=t===null?null:t.key;if(typeof n==`string`&&n!==``||typeof n==`number`||typeof n==`bigint`)return i===null?c(e,t,``+n,r):null;if(typeof n==`object`&&n){switch(n.$$typeof){case oe:return n.key===i?l(e,t,n,r):null;case se:return n.key===i?u(e,t,n,r):null;case _e:return n=co(n),p(e,t,n,r)}if(Ee(n)||C(n))return i===null?d(e,t,n,r,null):null;if(typeof n.then==`function`)return p(e,t,ho(n),r);if(n.$$typeof===S)return p(e,t,Na(e,n),r);_o(e,n)}return null}function m(e,t,n,r,i){if(typeof r==`string`&&r!==``||typeof r==`number`||typeof r==`bigint`)return e=e.get(n)||null,c(t,e,``+r,i);if(typeof r==`object`&&r){switch(r.$$typeof){case oe:return e=e.get(r.key===null?n:r.key)||null,l(t,e,r,i);case se:return e=e.get(r.key===null?n:r.key)||null,u(t,e,r,i);case _e:return r=co(r),m(e,t,n,r,i)}if(Ee(r)||C(r))return e=e.get(n)||null,d(t,e,r,i,null);if(typeof r.then==`function`)return m(e,t,n,ho(r),i);if(r.$$typeof===S)return m(e,t,n,Na(t,r),i);_o(t,r)}return null}function h(i,o,s,c){for(var l=null,u=null,d=o,h=o=0,g=null;d!==null&&h<s.length;h++){d.index>h?(g=d,d=null):g=d.sibling;var _=p(i,d,s[h],c);if(_===null){d===null&&(d=g);break}e&&d&&_.alternate===null&&t(i,d),o=a(_,o,h),u===null?l=_:u.sibling=_,u=_,d=g}if(h===s.length)return n(i,d),M&&sa(i,h),l;if(d===null){for(;h<s.length;h++)d=f(i,s[h],c),d!==null&&(o=a(d,o,h),u===null?l=d:u.sibling=d,u=d);return M&&sa(i,h),l}for(d=r(d);h<s.length;h++)g=m(d,i,h,s[h],c),g!==null&&(e&&(_=g.alternate,_!==null&&d.delete(_.key===null?h:_.key)),o=a(g,o,h),u===null?l=g:u.sibling=g,u=g);return e&&d.forEach(function(e){return t(i,e)}),M&&sa(i,h),l}function g(i,s,c,l){if(c==null)throw Error(o(151));for(var u=null,d=null,h=s,g=s=0,_=null,v=c.next();h!==null&&!v.done;g++,v=c.next()){h.index>g?(_=h,h=null):_=h.sibling;var y=p(i,h,v.value,l);if(y===null){h===null&&(h=_);break}e&&h&&y.alternate===null&&t(i,h),s=a(y,s,g),d===null?u=y:d.sibling=y,d=y,h=_}if(v.done)return n(i,h),M&&sa(i,g),u;if(h===null){for(;!v.done;g++,v=c.next())v=f(i,v.value,l),v!==null&&(s=a(v,s,g),d===null?u=v:d.sibling=v,d=v);return M&&sa(i,g),u}for(h=r(h);!v.done;g++,v=c.next())v=m(h,i,g,v.value,l),v!==null&&(e&&(_=v.alternate,_!==null&&h.delete(_.key===null?g:_.key)),s=a(v,s,g),d===null?u=v:d.sibling=v,d=v);return e&&h.forEach(function(e){return t(i,e)}),M&&sa(i,g),u}function _(e,r,a,c){if(typeof a==`object`&&a&&a.type===ce&&a.key===null&&a.props.ref===void 0&&(a=a.props.children),typeof a==`object`&&a){switch(a.$$typeof){case oe:a:{for(var l=a.key;r!==null;){if(r.key===l){if(l=a.type,l===ce){if(r.tag===7){n(e,r.sibling),c=i(r,a.props.children),go(c,a),c.return=e,e=c;break a}}else if(r.elementType===l||typeof l==`object`&&l&&l.$$typeof===_e&&co(l)===r.type){n(e,r.sibling),c=i(r,a.props),go(c,a),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}a.type===ce?(c=Ki(a.props.children,e.mode,c,a.key),go(c,a),c.return=e,e=c):(c=Gi(a.type,a.key,a.props,null,e.mode,c),go(c,a),c.return=e,e=c)}return s(e);case se:a:{for(l=a.key;r!==null;){if(r.key===l){if(r.tag===4&&r.stateNode.containerInfo===a.containerInfo&&r.stateNode.implementation===a.implementation){n(e,r.sibling),c=i(r,a.children||[]),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}c=Yi(a,e.mode,c),c.return=e,e=c}return s(e);case _e:return a=co(a),_(e,r,a,c)}if(Ee(a))return h(e,r,a,c);if(C(a)){if(l=C(a),typeof l!=`function`)throw Error(o(150));return a=l.call(a),g(e,r,a,c)}if(typeof a.then==`function`)return _(e,r,ho(a),c);if(a.$$typeof===S)return _(e,r,Na(e,a),c);_o(e,a)}return typeof a==`string`&&a!==``||typeof a==`number`||typeof a==`bigint`?(a=``+a,r!==null&&r.tag===6?(n(e,r.sibling),c=i(r,a),c.return=e,e=c):(n(e,r),c=qi(a,e.mode,c),c.return=e,e=c),s(e)):n(e,r)}return function(e,t,n,r){try{mo=0;var i=_(e,t,n,r);return po=null,i}catch(t){if(t===no||t===io)throw t;var a=Vi(29,t,null,e.mode);return a.lanes=r,a.return=e,a}}}var yo=vo(!0),bo=vo(!1),xo=!1;function So(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function Co(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,callbacks:null})}function wo(e){return{lane:e,tag:0,payload:null,callback:null,next:null}}function To(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,W&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,t=Ri(e),Li(e,null,n),t}return Pi(e,r,t,n),Ri(e)}function Eo(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,n&4194048)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Et(e,n)}}function Do(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,callbacks:r.callbacks},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}var Oo=!1;function ko(){if(Oo){var e=qa;if(e!==null)throw e}}function Ao(e,t,n,r){Oo=!1;var i=e.updateQueue;xo=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var c=s,l=c.next;c.next=null,o===null?a=l:o.next=l,o=c;var u=e.alternate;u!==null&&(u=u.updateQueue,s=u.lastBaseUpdate,s!==o&&(s===null?u.firstBaseUpdate=l:s.next=l,u.lastBaseUpdate=c))}if(a!==null){var d=i.baseState;o=0,u=l=c=null,s=a;do{var f=s.lane&-536870913,p=f!==s.lane;if(p?(q&f)===f:(r&f)===f){f!==0&&f===Ka&&(Oo=!0),u!==null&&(u=u.next={lane:0,tag:s.tag,payload:s.payload,callback:null,next:null});a:{var m=e,h=s;f=t;var g=n;switch(h.tag){case 1:if(m=h.payload,typeof m==`function`){d=m.call(g,d,f);break a}d=m;break a;case 3:m.flags=m.flags&-65537|128;case 0:if(m=h.payload,f=typeof m==`function`?m.call(g,d,f):m,f==null)break a;d=x({},d,f);break a;case 2:xo=!0}}f=s.callback,f!==null&&(e.flags|=64,p&&(e.flags|=8192),p=i.callbacks,p===null?i.callbacks=[f]:p.push(f))}else p={lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},u===null?(l=u=p,c=d):u=u.next=p,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(1);u===null&&(c=d),i.baseState=c,i.firstBaseUpdate=l,i.lastBaseUpdate=u,a===null&&(i.shared.lanes=0),sd|=o,e.lanes=o,e.memoizedState=d}}function jo(e,t){if(typeof e!=`function`)throw Error(o(191,e));e.call(t)}function Mo(e,t){var n=e.callbacks;if(n!==null)for(e.callbacks=null,e=0;e<n.length;e++)jo(n[e],t)}var No=Ae(null),Po=Ae(0);function Fo(e,t){e=od,E(Po,e),E(No,t),od=e|t.baseLanes}function P(){E(Po,od),E(No,No.current)}function Io(){od=Po.current,je(No),je(Po)}var Lo=Ae(null),Ro=null;function zo(e){var t=e.alternate;E(Wo,Wo.current&1),E(Lo,e),Ro===null&&(t===null||No.current!==null||t.memoizedState!==null)&&(Ro=e)}function Bo(e){E(Wo,Wo.current),E(Lo,e),Ro===null&&(Ro=e)}function Vo(e){e.tag===22?(E(Wo,Wo.current),E(Lo,e),Ro===null&&(Ro=e)):Ho()}function Ho(){E(Wo,Wo.current),E(Lo,Lo.current)}function Uo(e){je(Lo),Ro===e&&(Ro=null),je(Wo)}var Wo=Ae(0);function Go(e,t){E(Lo,Lo.current),E(Wo,t)}function Ko(e){je(Wo),je(Lo),Ro===e&&(Ro=null)}function qo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||om(n)||sm(n)))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==`independent`){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Jo=0,F=null,I=null,L=null,Yo=!1,Xo=!1,Zo=!1,Qo=0,$o=0,es=null,ts=0;function R(){throw Error(o(321))}function ns(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Xr(e[n],t[n]))return!1;return!0}function rs(e,t,n,r,i,a){return Jo=a,F=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,w.H=e===null||e.memoizedState===null?vc:yc,Zo=!1,a=n(r,i),Zo=!1,Xo&&(a=as(t,n,r,i)),is(e),a}function is(e){w.H=_c;var t=I!==null&&I.next!==null;if(Jo=0,L=I=F=null,Yo=!1,$o=0,es=null,t)throw Error(o(300));e===null||Ic||(e=e.dependencies,e!==null&&N(e)&&(Ic=!0))}function as(e,t,n,r){F=e;var i=0;do{if(Xo&&(es=null),$o=0,Xo=!1,25<=i)throw Error(o(301));if(i+=1,L=I=null,e.updateQueue!=null){var a=e.updateQueue;a.lastEffect=null,a.events=null,a.stores=null,a.memoCache!=null&&(a.memoCache.index=0)}w.H=bc,a=t(n,r)}while(Xo);return a}function os(){var e=w.H,t=e.useState()[0];return t=typeof t.then==`function`?fs(t):t,e=e.useState()[0],(I===null?null:I.memoizedState)!==e&&(F.flags|=1024),t}function ss(){var e=Qo!==0;return Qo=0,e}function cs(e,t,n){t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~n}function ls(e){if(Yo){for(e=e.memoizedState;e!==null;){var t=e.queue;t!==null&&(t.pending=null),e=e.next}Yo=!1}Jo=0,L=I=F=null,Xo=!1,$o=Qo=0,es=null}function us(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return L===null?F.memoizedState=L=e:L=L.next=e,L}function z(){if(I===null){var e=F.alternate;e=e===null?null:e.memoizedState}else e=I.next;var t=L===null?F.memoizedState:L.next;if(t!==null)L=t,I=e;else{if(e===null)throw F.alternate===null?Error(o(467)):Error(o(310));I=e,e={memoizedState:I.memoizedState,baseState:I.baseState,baseQueue:I.baseQueue,queue:I.queue,next:null},L===null?F.memoizedState=L=e:L=L.next=e}return L}function ds(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function fs(e){var t=$o;return $o+=1,es===null&&(es=[]),e=so(es,e,t),t=F,(L===null?t.memoizedState:L.next)===null&&(t=t.alternate,w.H=t===null||t.memoizedState===null?vc:yc),e}function ps(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return fs(e);if(e.$$typeof===Se)return;if(e.$$typeof===S)return Ma(e)}throw Error(o(438,String(e)))}function ms(e){var t=null,n=F.updateQueue;if(n!==null&&(t=n.memoCache),t==null){var r=F.alternate;r!==null&&(r=r.updateQueue,r!==null&&(r=r.memoCache,r!=null&&(t={data:r.data.map(function(e){return e.slice()}),index:0})))}if(t??={data:[],index:0},n===null&&(n=ds(),F.updateQueue=n),n.memoCache=t,n=t.data[t.index],n===void 0)for(n=t.data[t.index]=Array(e),r=0;r<e;r++)n[r]=be;return t.index++,n}function hs(e,t){return typeof t==`function`?t(e):t}function gs(e){return _s(z(),I,e)}function _s(e,t,n){var r=e.queue;if(r===null)throw Error(o(311));r.lastRenderedReducer=n;var i=e.baseQueue,a=r.pending;if(a!==null){if(i!==null){var s=i.next;i.next=a.next,a.next=s}t.baseQueue=i=a,r.pending=null}if(a=e.baseState,i===null)e.memoizedState=a;else{t=i.next;var c=s=null,l=null,u=t,d=!1;do{var f=u.lane&-536870913;if(f===u.lane?(Jo&f)===f:(q&f)===f){var p=u.revertLane;if(p===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),f===Ka&&(d=!0);else if((Jo&p)===p){u=u.next,p===Ka&&(d=!0);continue}else f={lane:0,revertLane:u.revertLane,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=f,s=a):l=l.next=f,F.lanes|=p,sd|=p;f=u.action,Zo&&n(a,f),a=u.hasEagerState?u.eagerState:n(a,f)}else p={lane:f,revertLane:u.revertLane,gesture:u.gesture,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=p,s=a):l=l.next=p,F.lanes|=f,sd|=f;u=u.next}while(u!==null&&u!==t);if(l===null?s=a:l.next=c,!Xr(a,e.memoizedState)&&(Ic=!0,d&&(n=qa,n!==null)))throw n;e.memoizedState=a,e.baseState=s,e.baseQueue=l,r.lastRenderedState=a}return i===null&&(r.lanes=0),[e.memoizedState,r.dispatch]}function vs(e){var t=z(),n=t.queue;if(n===null)throw Error(o(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var s=i=i.next;do a=e(a,s.action),s=s.next;while(s!==i);Xr(a,t.memoizedState)||(Ic=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function ys(e,t,n){var r=F,i=z(),a=M;if(a){if(n===void 0)throw Error(o(407));n=n()}else n=t();var s=!Xr((I||i).memoizedState,n);if(s&&(i.memoizedState=n,Ic=!0),i=i.queue,Ws(Ss.bind(null,r,i,e),[e]),e=i.getSnapshot!==t||s||L!==null&&!!(L.memoizedState.tag&1),zs(e?9:8,{destroy:void 0},xs.bind(null,r,i,n,t),null),e){if(r.flags|=2048,G===null)throw Error(o(349));a||Jo&127||bs(r,t,n)}return n}function bs(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=F.updateQueue,t===null?(t=ds(),F.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function xs(e,t,n,r){t.value=n,t.getSnapshot=r,Cs(t)&&ws(e)}function Ss(e,t,n){return n(function(){Cs(t)&&ws(e)})}function Cs(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Xr(e,n)}catch{return!0}}function ws(e){var t=Ii(e,2);t!==null&&Pd(t,e,2)}function Ts(e){var t=us();if(typeof e==`function`){var n=e;if(e=n(),Zo){lt(!0);try{n()}finally{lt(!1)}}}return t.memoizedState=t.baseState=e,t.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:hs,lastRenderedState:e},t}function Es(e,t,n,r){return e.baseState=n,_s(e,I,typeof r==`function`?r:hs)}function Ds(e,t,n,r,i){if(mc(e))throw Error(o(485));if(e=t.action,e!==null){var a={payload:i,action:e,next:null,isTransition:!0,status:`pending`,value:null,reason:null,listeners:[],then:function(e){a.listeners.push(e)}};w.T===null?a.isTransition=!1:n(!0),r(a),n=t.pending,n===null?(a.next=t.pending=a,Os(t,a)):(a.next=n.next,t.pending=n.next=a)}}function Os(e,t){var n=t.action,r=t.payload,i=e.state;if(t.isTransition){var a=w.T,o={};o.types=a===null?null:a.types,w.T=o;try{var s=n(i,r),c=w.S;c!==null&&c(o,s),ks(e,t,s)}catch(n){js(e,t,n)}finally{a!==null&&o.types!==null&&(a.types=o.types),w.T=a}}else try{a=n(i,r),ks(e,t,a)}catch(n){js(e,t,n)}}function ks(e,t,n){typeof n==`object`&&n&&typeof n.then==`function`?n.then(function(n){As(e,t,n)},function(n){return js(e,t,n)}):As(e,t,n)}function As(e,t,n){t.status=`fulfilled`,t.value=n,Ms(t),e.state=n,t=e.pending,t!==null&&(n=t.next,n===t?e.pending=null:(n=n.next,t.next=n,Os(e,n)))}function js(e,t,n){var r=e.pending;if(e.pending=null,r!==null){r=r.next;do t.status=`rejected`,t.reason=n,Ms(t),t=t.next;while(t!==r)}e.action=null}function Ms(e){e=e.listeners;for(var t=0;t<e.length;t++)(0,e[t])()}function Ns(e,t){return t}function Ps(e,t){if(M){var n=G.formState;if(n!==null){a:{var r=F;if(M){if(j){b:{for(var i=j,a=ma;i.nodeType!==8;){if(!a){i=null;break b}if(i=lm(i.nextSibling),i===null){i=null;break b}}a=i.data,i=a===`F!`||a===`F`?i:null}if(i){j=lm(i.nextSibling),r=i.data===`F!`;break a}}ga(r)}r=!1}r&&(t=n[0])}}return n=us(),n.memoizedState=n.baseState=t,r={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Ns,lastRenderedState:t},n.queue=r,n=dc.bind(null,F,r),r.dispatch=n,r=Ts(!1),a=pc.bind(null,F,!1,r.queue),r=us(),i={state:t,dispatch:null,action:e,pending:null},r.queue=i,n=Ds.bind(null,F,i,a,n),i.dispatch=n,r.memoizedState=e,[t,n,!1]}function Fs(e){return Is(z(),I,e)}function Is(e,t,n){if(t=_s(e,t,Ns)[0],e=gs(hs)[0],typeof t==`object`&&t&&typeof t.then==`function`)try{var r=fs(t)}catch(e){throw e===no?io:e}else r=t;t=z();var i=t.queue,a=i.dispatch;return n!==t.memoizedState&&(F.flags|=2048,zs(9,{destroy:void 0},Ls.bind(null,i,n),null)),[r,a,e]}function Ls(e,t){e.action=t}function Rs(e){var t=z(),n=I;if(n!==null)return Is(t,n,e);z(),t=t.memoizedState,n=z();var r=n.queue.dispatch;return n.memoizedState=e,[t,r,!1]}function zs(e,t,n,r){return e={tag:e,create:n,deps:r,inst:t,next:null},t=F.updateQueue,t===null&&(t=ds(),F.updateQueue=t),n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e),e}function Bs(){return z().memoizedState}function Vs(e,t,n,r){var i=us();F.flags|=e,i.memoizedState=zs(1|t,{destroy:void 0},n,r===void 0?null:r)}function Hs(e,t,n,r){var i=z();r=r===void 0?null:r;var a=i.memoizedState.inst;I!==null&&r!==null&&ns(r,I.memoizedState.deps)?i.memoizedState=zs(t,a,n,r):(F.flags|=e,i.memoizedState=zs(1|t,a,n,r))}function Us(e,t){Vs(8390656,8,e,t)}function Ws(e,t){Hs(2048,8,e,t)}function Gs(e){F.flags|=4;var t=F.updateQueue;if(t===null)t=ds(),F.updateQueue=t,t.events=[e];else{var n=t.events;n===null?t.events=[e]:n.push(e)}}function Ks(e){var t=z().memoizedState;return Gs({ref:t,nextImpl:e}),function(){if(W&2)throw Error(o(440));return t.impl.apply(void 0,arguments)}}function qs(e,t){return Hs(4,2,e,t)}function Js(e,t){return Hs(4,4,e,t)}function B(e,t){if(typeof t==`function`){e=e();var n=t(e);return function(){typeof n==`function`?n():t(null)}}if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Ys(e,t,n){n=n==null?null:n.concat([e]),Hs(4,4,B.bind(null,t,e),n)}function Xs(){}function Zs(e,t){var n=z();t=t===void 0?null:t;var r=n.memoizedState;return t!==null&&ns(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Qs(e,t){var n=z();t=t===void 0?null:t;var r=n.memoizedState;if(t!==null&&ns(t,r[1]))return r[0];if(r=e(),Zo){lt(!0);try{e()}finally{lt(!1)}}return n.memoizedState=[r,t],r}function $s(e,t,n){return n===void 0||Jo&1073741824&&!(q&261930)?e.memoizedState=t:(e.memoizedState=n,e=Md(),F.lanes|=e,sd|=e,n)}function ec(e,t,n,r){return Xr(n,t)?n:No.current===null?!(Jo&106)||Jo&1073741824&&!(q&261930)?(Ic=!0,e.memoizedState=n):(e=Md(),F.lanes|=e,sd|=e,t):(e=$s(e,n,r),Xr(e,t)||(Ic=!0),e)}function tc(e,t,n,r,i){var a=T.p;T.p=a!==0&&8>a?a:8;var o=w.T,s={};s.types=o===null?null:o.types,w.T=s,pc(e,!1,t,n);try{var c=i(),l=w.S;l!==null&&l(s,c),typeof c==`object`&&c&&typeof c.then==`function`?fc(e,t,Xa(c,r),jd(e)):fc(e,t,r,jd(e))}catch(n){fc(e,t,{then:function(){},status:`rejected`,reason:n},jd())}finally{T.p=a,o!==null&&s.types!==null&&(o.types=s.types),w.T=o}}function nc(){}function rc(e,t,n,r){if(e.tag!==5)throw Error(o(476));var i=ic(e).queue;tc(e,i,t,De,n===null?nc:function(){return ac(e),n(r)})}function ic(e){var t=e.memoizedState;if(t!==null)return t;t={memoizedState:De,baseState:De,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:hs,lastRenderedState:De},next:null};var n={};return t.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:hs,lastRenderedState:n},next:null},e.memoizedState=t,e=e.alternate,e!==null&&(e.memoizedState=t),t}function ac(e){var t=ic(e);t.next===null&&(t=e.alternate.memoizedState),fc(e,t.next.queue,{},jd())}function oc(){return Ma(sh)}function sc(){return z().memoizedState}function cc(){return z().memoizedState}function lc(e){for(var t=e.return;t!==null;){switch(t.tag){case 24:case 3:var n=jd();e=wo(n);var r=To(t,e,n);r!==null&&(Pd(r,t,n),Eo(r,t,n)),t={cache:za()},e.payload=t;return}t=t.return}}function uc(e,t,n){var r=jd();n={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},mc(e)?hc(t,n):(n=Fi(e,t,n,r),n!==null&&(Pd(n,e,r),gc(n,t,r)))}function dc(e,t,n){fc(e,t,n,jd())}function fc(e,t,n,r){var i={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(mc(e))hc(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Xr(s,o))return Pi(e,t,i,0),G===null&&Ni(),!1}catch{}if(n=Fi(e,t,i,r),n!==null)return Pd(n,e,r),gc(n,t,r),!0}return!1}function pc(e,t,n,r){if(r={lane:2,revertLane:Pf(),gesture:null,action:r,hasEagerState:!1,eagerState:null,next:null},mc(e)){if(t)throw Error(o(479))}else t=Fi(e,n,r,2),t!==null&&Pd(t,e,2)}function mc(e){var t=e.alternate;return e===F||t!==null&&t===F}function hc(e,t){Xo=Yo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function gc(e,t,n){if(n&4194048){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Et(e,n)}}var _c={readContext:Ma,use:ps,useCallback:R,useContext:R,useEffect:R,useImperativeHandle:R,useLayoutEffect:R,useInsertionEffect:R,useMemo:R,useReducer:R,useRef:R,useState:R,useDebugValue:R,useDeferredValue:R,useTransition:R,useSyncExternalStore:R,useId:R,useHostTransitionStatus:R,useFormState:R,useActionState:R,useOptimistic:R,useMemoCache:R,useCacheRefresh:R,useEffectEvent:R},vc={readContext:Ma,use:ps,useCallback:function(e,t){return us().memoizedState=[e,t===void 0?null:t],e},useContext:Ma,useEffect:Us,useImperativeHandle:function(e,t,n){n=n==null?null:n.concat([e]),Vs(4194308,4,B.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Vs(4194308,4,e,t)},useInsertionEffect:function(e,t){Vs(4,2,e,t)},useMemo:function(e,t){var n=us();t=t===void 0?null:t;var r=e();if(Zo){lt(!0);try{e()}finally{lt(!1)}}return n.memoizedState=[r,t],r},useReducer:function(e,t,n){var r=us();if(n!==void 0){var i=n(t);if(Zo){lt(!0);try{n(t)}finally{lt(!1)}}}else i=t;return r.memoizedState=r.baseState=i,e={pending:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:i},r.queue=e,e=e.dispatch=uc.bind(null,F,e),[r.memoizedState,e]},useRef:function(e){var t=us();return e={current:e},t.memoizedState=e},useState:function(e){e=Ts(e);var t=e.queue,n=dc.bind(null,F,t);return t.dispatch=n,[e.memoizedState,n]},useDebugValue:Xs,useDeferredValue:function(e,t){return $s(us(),e,t)},useTransition:function(){var e=Ts(!1);return e=tc.bind(null,F,e.queue,!0,!1),us().memoizedState=e,[!1,e]},useSyncExternalStore:function(e,t,n){var r=F,i=us();if(M){if(n===void 0)throw Error(o(407));n=n()}else{if(n=t(),G===null)throw Error(o(349));q&127||bs(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,Us(Ss.bind(null,r,a,e),[e]),r.flags|=2048,zs(9,{destroy:void 0},xs.bind(null,r,a,n,t),null),n},useId:function(){var e=us(),t=G.identifierPrefix;if(M){var n=oa,r=aa;n=(r&~(1<<32-D(r)-1)).toString(32)+n,t=`_`+t+`R_`+n,n=Qo++,0<n&&(t+=`H`+n.toString(32)),t+=`_`}else n=ts++,t=`_`+t+`r_`+n.toString(32)+`_`;return e.memoizedState=t},useHostTransitionStatus:oc,useFormState:Ps,useActionState:Ps,useOptimistic:function(e){var t=us();t.memoizedState=t.baseState=e;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return t.queue=n,t=pc.bind(null,F,!0,n),n.dispatch=t,[e,t]},useMemoCache:ms,useCacheRefresh:function(){return us().memoizedState=lc.bind(null,F)},useEffectEvent:function(e){var t=us(),n={impl:e};return t.memoizedState=n,function(){if(W&2)throw Error(o(440));return n.impl.apply(void 0,arguments)}}},yc={readContext:Ma,use:ps,useCallback:Zs,useContext:Ma,useEffect:Ws,useImperativeHandle:Ys,useInsertionEffect:qs,useLayoutEffect:Js,useMemo:Qs,useReducer:gs,useRef:Bs,useState:function(){return gs(hs)},useDebugValue:Xs,useDeferredValue:function(e,t){return ec(z(),I.memoizedState,e,t)},useTransition:function(){var e=gs(hs)[0],t=z().memoizedState;return[typeof e==`boolean`?e:fs(e),t]},useSyncExternalStore:ys,useId:sc,useHostTransitionStatus:oc,useFormState:Fs,useActionState:Fs,useOptimistic:function(e,t){return Es(z(),I,e,t)},useMemoCache:ms,useCacheRefresh:cc,useEffectEvent:Ks},bc={readContext:Ma,use:ps,useCallback:Zs,useContext:Ma,useEffect:Ws,useImperativeHandle:Ys,useInsertionEffect:qs,useLayoutEffect:Js,useMemo:Qs,useReducer:vs,useRef:Bs,useState:function(){return vs(hs)},useDebugValue:Xs,useDeferredValue:function(e,t){var n=z();return I===null?$s(n,e,t):ec(n,I.memoizedState,e,t)},useTransition:function(){var e=vs(hs)[0],t=z().memoizedState;return[typeof e==`boolean`?e:fs(e),t]},useSyncExternalStore:ys,useId:sc,useHostTransitionStatus:oc,useFormState:Rs,useActionState:Rs,useOptimistic:function(e,t){var n=z();return I===null?(n.baseState=e,[e,n.queue.dispatch]):Es(n,I,e,t)},useMemoCache:ms,useCacheRefresh:cc,useEffectEvent:Ks};function xc(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:x({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Sc={enqueueSetState:function(e,t,n){e=e._reactInternals;var r=jd(),i=wo(r);i.payload=t,n!=null&&(i.callback=n),t=To(e,i,r),t!==null&&(Pd(t,e,r),Eo(t,e,r))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=jd(),i=wo(r);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=To(e,i,r),t!==null&&(Pd(t,e,r),Eo(t,e,r))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=jd(),r=wo(n);r.tag=2,t!=null&&(r.callback=t),t=To(e,r,n),t!==null&&(Pd(t,e,n),Eo(t,e,n))}};function Cc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate==`function`?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Zr(n,r)||!Zr(i,a):!0}function wc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps==`function`&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps==`function`&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Sc.enqueueReplaceState(t,t.state,null)}function Tc(e,t){var n=t;if(`ref`in t)for(var r in n={},t)r!==`ref`&&(n[r]=t[r]);if(e=e.defaultProps)for(var i in n===t&&(n=x({},n)),e)n[i]===void 0&&(n[i]=e[i]);return n}function Ec(e){ki(e)}function Dc(e){console.error(e)}function Oc(e){ki(e)}function kc(e,t){try{var n=e.onUncaughtError;n(t.value,{componentStack:t.stack})}catch(e){setTimeout(function(){throw e})}}function Ac(e,t,n){try{var r=e.onCaughtError;r(n.value,{componentStack:n.stack,errorBoundary:t.tag===1?t.stateNode:null})}catch(e){setTimeout(function(){throw e})}}function jc(e,t,n){return n=wo(n),n.tag=3,n.payload={element:null},n.callback=function(){kc(e,t)},n}function Mc(e){return e=wo(e),e.tag=3,e}function Nc(e,t,n,r){var i=n.type.getDerivedStateFromError;if(typeof i==`function`){var a=r.value;e.payload=function(){return i(a)},e.callback=function(){Ac(t,n,r)}}var o=n.stateNode;o!==null&&typeof o.componentDidCatch==`function`&&(e.callback=function(){Ac(t,n,r),typeof i!=`function`&&(yd===null?yd=new Set([this]):yd.add(this));var e=r.stack;this.componentDidCatch(r.value,{componentStack:e===null?``:e})})}function Pc(e,t,n,r,i){if(n.flags|=32768,typeof r==`object`&&r&&typeof r.then==`function`){if(t=n.alternate,t!==null&&Aa(t,n,i,!0),n=Lo.current,n!==null){switch(n.tag){case 31:case 13:case 19:return Ro===null?Kd():n.alternate===null&&Y===0&&(Y=3),n.flags&=-257,n.flags|=65536,n.lanes=i,r===ao?n.flags|=16384:(t=n.updateQueue,t===null?n.updateQueue=new Set([r]):t.add(r),mf(e,r,i)),!1;case 22:return n.flags|=65536,r===ao?n.flags|=16384:(t=n.updateQueue,t===null?(t={transitions:null,markerInstances:null,retryQueue:new Set([r])},n.updateQueue=t):(n=t.retryQueue,n===null?t.retryQueue=new Set([r]):n.add(r)),mf(e,r,i)),!1}throw Error(o(435,n.tag))}return mf(e,r,i),Kd(),!1}if(M)return t=Lo.current,t===null?(r!==ha&&(t=Error(o(423),{cause:r}),Sa(Zi(t,n))),e=e.current.alternate,e.flags|=65536,i&=-i,e.lanes|=i,r=Zi(r,n),i=jc(e.stateNode,r,i),Do(e,i),Y!==4&&(Y=2)):(!(t.flags&65536)&&(t.flags|=256),t.flags|=65536,t.lanes=i,r!==ha&&(e=Error(o(422),{cause:r}),Sa(Zi(e,n)))),!1;var a=Error(o(520),{cause:r});if(a=Zi(a,n),fd===null?fd=[a]:fd.push(a),Y!==4&&(Y=2),t===null)return!0;r=Zi(r,n),n=t;do{switch(n.tag){case 3:return n.flags|=65536,e=i&-i,n.lanes|=e,e=jc(n.stateNode,r,e),Do(n,e),!1;case 1:if(t=n.type,a=n.stateNode,!(n.flags&128)&&(typeof t.getDerivedStateFromError==`function`||a!==null&&typeof a.componentDidCatch==`function`&&(yd===null||!yd.has(a))))return n.flags|=65536,i&=-i,n.lanes|=i,i=Mc(i),Nc(i,e,n,r),Do(n,i),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var Fc=Error(o(461)),Ic=!1;function Lc(e,t,n,r){t.child=e===null?bo(t,null,n,r):yo(t,e.child,n,r)}function Rc(e,t,n,r,i){n=n.render;var a=t.ref;if(`ref`in r){var o={};for(var s in r)s!==`ref`&&(o[s]=r[s])}else o=r;return ja(t),r=rs(e,t,n,o,a,i),s=ss(),e!==null&&!Ic?(cs(e,t,i),fl(e,t,i)):(M&&s&&la(t),t.flags|=1,Lc(e,t,r,i),t.child)}function zc(e,t,n,r,i){if(e===null){var a=n.type;return typeof a==`function`&&!Hi(a)&&a.defaultProps===void 0&&n.compare===null?(t.tag=15,t.type=a,Bc(e,t,a,r,i)):(e=Gi(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!pl(e,i)){var o=a.memoizedProps;if(n=n.compare,n=n===null?Zr:n,n(o,r)&&e.ref===t.ref)return fl(e,t,i)}return t.flags|=1,e=Ui(a,r),e.ref=t.ref,e.return=t,t.child=e}function Bc(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Zr(a,r)&&e.ref===t.ref){if(Ic=!1,t.pendingProps=r=a,pl(e,i))e.flags&131072&&(Ic=!0);else return t.lanes=e.lanes,fl(e,t,i)}}return Jc(e,t,n,r,i)}function Vc(e,t,n,r){var i=r.children,a=e===null?null:e.memoizedState;if(e===null&&t.stateNode===null&&(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),r.mode===`hidden`){if(t.flags&128){if(a=a===null?n:a.baseLanes|n,e!==null){for(r=t.child=e.child,i=0;r!==null;)i=i|r.lanes|r.childLanes,r=r.sibling;r=i&~a}else r=0,t.child=null;return Uc(e,t,a,n,r)}if(n&536870912)t.memoizedState={baseLanes:0,cachePool:null},e!==null&&eo(t,a===null?null:a.cachePool),a===null?P():Fo(t,a),Vo(t);else return r=t.lanes=536870912,Uc(e,t,a===null?n:a.baseLanes|n,n,r)}else a===null?(e!==null&&eo(t,null),P(),Ho()):(eo(t,a.cachePool),Fo(t,a),Ho(),t.memoizedState=null);return Lc(e,t,i,n),t.child}function Hc(e,t){return e!==null&&e.tag===22||t.stateNode!==null||(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),t.sibling}function Uc(e,t,n,r,i){var a=$a();return a=a===null?null:{parent:Ra._currentValue,pool:a},t.memoizedState={baseLanes:n,cachePool:a},e!==null&&eo(t,null),P(),Vo(t),e!==null&&Aa(e,t,r,!0),t.childLanes=i,null}function Wc(e,t){return t=rl({mode:t.mode,children:t.children},e.mode),t.ref=e.ref,e.child=t,t.return=e,t}function Gc(e,t,n){return yo(t,e.child,null,n),e=Wc(t,t.pendingProps),e.flags|=2,Uo(t),t.memoizedState=null,e}function Kc(e,t,n){var r=t.pendingProps,i=!!(t.flags&128);if(t.flags&=-129,e===null){if(M){if(r.mode===`hidden`)return e=Wc(t,r),t.lanes=536870912,e.memoizedState={baseLanes:0,cachePool:null},Hc(null,e);if(Bo(t),(e=j)?(e=am(e,ma),e=e!==null&&e.data===`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:ia===null?null:{id:aa,overflow:oa},retryLane:536870912,hydrationErrors:null},n=Ji(e),n.return=t,t.child=n,fa=t,j=null)):e=null,e===null)throw ga(t);return t.lanes=536870912,null}return Wc(t,r)}var a=e.memoizedState;if(a!==null){var s=a.dehydrated;if(Bo(t),i){if(t.flags&256)t.flags&=-257,t=Gc(e,t,n);else if(t.memoizedState!==null)t.child=e.child,t.flags|=128,t=null;else throw Error(o(558))}else if(Ic||Aa(e,t,n,!1),i=(n&e.childLanes)!==0,Ic||i){if(No.current===null){if(r=G,r!==null&&(s=Dt(r,n),s!==0&&s!==a.retryLane))throw a.retryLane=s,Ii(e,s),Pd(r,e,s),Fc;Kd()}t=Gc(e,t,n)}else e=a.treeContext,j=lm(s.nextSibling),fa=t,M=!0,pa=null,ma=!1,e!==null&&da(t,e),t=Wc(t,r),t.flags|=134221824;return t}return e=Ui(e.child,{mode:r.mode,children:r.children}),e.ref=t.ref,t.child=e,e.return=t,e}function qc(e,t){var n=t.ref;if(n===null)e!==null&&e.ref!==null&&(t.flags|=4194816);else{if(typeof n!=`function`&&typeof n!=`object`)throw Error(o(284));(e===null||e.ref!==n)&&(t.flags|=4194816)}}function Jc(e,t,n,r,i){return ja(t),n=rs(e,t,n,r,void 0,i),r=ss(),e!==null&&!Ic?(cs(e,t,i),fl(e,t,i)):(M&&r&&la(t),t.flags|=1,Lc(e,t,n,i),t.child)}function Yc(e,t,n,r,i,a){return ja(t),t.updateQueue=null,n=as(t,r,n,i),is(e),r=ss(),e!==null&&!Ic?(cs(e,t,a),fl(e,t,a)):(M&&r&&la(t),t.flags|=1,Lc(e,t,n,a),t.child)}function Xc(e,t,n,r,i){if(ja(t),t.stateNode===null){var a=zi,o=n.contextType;typeof o==`object`&&o&&(a=Ma(o)),a=new n(r,a),t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,a.updater=Sc,t.stateNode=a,a._reactInternals=t,a=t.stateNode,a.props=r,a.state=t.memoizedState,a.refs={},So(t),o=n.contextType,a.context=typeof o==`object`&&o?Ma(o):zi,a.state=t.memoizedState,o=n.getDerivedStateFromProps,typeof o==`function`&&(xc(t,n,o,r),a.state=t.memoizedState),typeof n.getDerivedStateFromProps==`function`||typeof a.getSnapshotBeforeUpdate==`function`||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(o=a.state,typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount(),o!==a.state&&Sc.enqueueReplaceState(a,a.state,null),Ao(t,r,a,i),ko(),a.state=t.memoizedState),typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!0}else if(e===null){a=t.stateNode;var s=t.memoizedProps,c=Tc(n,s);a.props=c;var l=a.context,u=n.contextType;o=zi,typeof u==`object`&&u&&(o=Ma(u));var d=n.getDerivedStateFromProps;u=typeof d==`function`||typeof a.getSnapshotBeforeUpdate==`function`,s=t.pendingProps!==s,u||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(s||l!==o)&&wc(t,a,r,o),xo=!1;var f=t.memoizedState;a.state=f,Ao(t,r,a,i),ko(),l=t.memoizedState,s||f!==l||xo?(typeof d==`function`&&(xc(t,n,d,r),l=t.memoizedState),(c=xo||Cc(t,n,c,r,f,l,o))?(u||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount==`function`&&(t.flags|=4194308)):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=o,r=c):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,Co(e,t),o=t.memoizedProps,u=Tc(n,o),a.props=u,d=t.pendingProps,f=a.context,l=n.contextType,c=zi,typeof l==`object`&&l&&(c=Ma(l)),s=n.getDerivedStateFromProps,(l=typeof s==`function`||typeof a.getSnapshotBeforeUpdate==`function`)||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(o!==d||f!==c)&&wc(t,a,r,c),xo=!1,f=t.memoizedState,a.state=f,Ao(t,r,a,i),ko();var p=t.memoizedState;o!==d||f!==p||xo||e!==null&&e.dependencies!==null&&N(e.dependencies)?(typeof s==`function`&&(xc(t,n,s,r),p=t.memoizedState),(u=xo||Cc(t,n,u,r,f,p,c)||e!==null&&e.dependencies!==null&&N(e.dependencies))?(l||typeof a.UNSAFE_componentWillUpdate!=`function`&&typeof a.componentWillUpdate!=`function`||(typeof a.componentWillUpdate==`function`&&a.componentWillUpdate(r,p,c),typeof a.UNSAFE_componentWillUpdate==`function`&&a.UNSAFE_componentWillUpdate(r,p,c)),typeof a.componentDidUpdate==`function`&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate==`function`&&(t.flags|=1024)):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=p),a.props=r,a.state=p,a.context=c,r=u):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return a=r,qc(e,t),r=!!(t.flags&128),a||r?(a=t.stateNode,n=r&&typeof n.getDerivedStateFromError!=`function`?null:a.render(),t.flags|=1,e!==null&&r?(t.child=yo(t,e.child,null,i),t.child=yo(t,null,n,i)):Lc(e,t,n,i),t.memoizedState=a.state,e=t.child):e=fl(e,t,i),e}function Zc(e,t,n,r){return ba(),t.flags|=256,Lc(e,t,n,r),t.child}var Qc={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function $c(e){return{baseLanes:e,cachePool:to()}}function el(e,t,n){return e=e===null?0:e.childLanes&~n,t&&(e|=ud),e}function tl(e,t,n){var r=t.pendingProps,i=!1,a=!!(t.flags&128),o;if((o=a)||(o=e!==null&&e.memoizedState===null?!1:!!(Wo.current&2)),o&&(i=!0,t.flags&=-129),o=!!(t.flags&32),t.flags&=-33,e===null){if(M){if(i?zo(t):Ho(),(e=j)?(e=am(e,ma),e=e!==null&&e.data!==`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:ia===null?null:{id:aa,overflow:oa},retryLane:536870912,hydrationErrors:null},n=Ji(e),n.return=t,t.child=n,fa=t,j=null)):e=null,e===null)throw ga(t);return t.lanes=sm(e)?32:536870912,null}return a=r.children,r=r.fallback,i?(Ho(),i=t.mode,a=rl({mode:`hidden`,children:a},i),r=Ki(r,i,n,null),a.return=t,r.return=t,a.sibling=r,t.child=a,r=t.child,r.memoizedState=$c(n),r.childLanes=el(e,o,n),t.memoizedState=Qc,Hc(null,r)):(zo(t),nl(t,a))}var s=e.memoizedState;if(s!==null){var c=s.dehydrated;if(c!==null)return al(e,t,a,o,r,c,s,n)}return i?(Ho(),i=r.fallback,a=t.mode,s=e.child,c=s.sibling,r=Ui(s,{mode:`hidden`,children:r.children}),r.subtreeFlags=s.subtreeFlags&1206910976,c===null?(i=Ki(i,a,n,null),i.flags|=2):i=Ui(c,i),i.return=t,r.return=t,r.sibling=i,t.child=r,Hc(null,r),r=t.child,i=e.child.memoizedState,i===null?i=$c(n):(a=i.cachePool,a===null?a=to():(s=Ra._currentValue,a=a.parent===s?a:{parent:s,pool:s}),i={baseLanes:i.baseLanes|n,cachePool:a}),r.memoizedState=i,r.childLanes=el(e,o,n),t.memoizedState=Qc,Hc(e.child,r)):(zo(t),n=e.child,e=n.sibling,n=Ui(n,{mode:`visible`,children:r.children}),n.return=t,n.sibling=null,e!==null&&(o=t.deletions,o===null?(t.deletions=[e],t.flags|=16):o.push(e)),t.child=n,t.memoizedState=null,n)}function nl(e,t){return t=rl({mode:`visible`,children:t},e.mode),t.return=e,e.child=t}function rl(e,t){return e=Vi(22,e,null,t),e.lanes=0,e}function il(e,t,n){return yo(t,e.child,null,n),e=nl(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function al(e,t,n,r,i,a,s,c){if(n)return t.flags&256?(zo(t),t.flags&=-257,il(e,t,c)):t.memoizedState===null?(Ho(),a=i.fallback,s=t.mode,i=rl({mode:`visible`,children:i.children},s),a=Ki(a,s,c,null),a.flags|=2,i.return=t,a.return=t,i.sibling=a,t.child=i,yo(t,e.child,null,c),i=t.child,i.memoizedState=$c(c),i.childLanes=el(e,r,c),t.memoizedState=Qc,Hc(null,i)):(Ho(),t.child=e.child,t.flags|=128,null);if(zo(t),sm(a)){if(r=a.nextSibling&&a.nextSibling.dataset,r)var l=r.dgst;return r=l,r!==``&&(i=Error(o(419)),i.stack=``,i.digest=r,Sa({value:i,source:null,stack:null})),il(e,t,c)}if(Ic||Aa(e,t,c,!1),r=(c&e.childLanes)!==0,Ic||r){if(No.current!==null)return il(e,t,c);if(r=G,r!==null&&(i=Dt(r,c),i!==0&&i!==s.retryLane))throw s.retryLane=i,Ii(e,i),Pd(r,e,i),Fc;return om(a)||Kd(),il(e,t,c)}return om(a)?(t.flags|=192,t.child=e.child,null):(e=s.treeContext,j=lm(a.nextSibling),fa=t,M=!0,pa=null,ma=!1,e!==null&&da(t,e),t=nl(t,i.children),t.flags|=134221824,t)}function ol(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Oa(e.return,t,n)}function sl(e){for(var t=null;e!==null;){var n=e.alternate;n!==null&&qo(n)===null&&(t=e),e=e.sibling}return t}function cl(e,t,n,r,i,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i,treeForkCount:a}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i,o.treeForkCount=a)}function ll(e){var t=e.child;for(e.child=null;t!==null;){var n=t.sibling;t.sibling=e.child,e.child=t,t=n}}function ul(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;r=r.children;var o=Wo.current;if(t.flags&128)return Go(t,o),null;var s=!!(o&2);if(s?(o=o&1|2,t.flags|=128):o&=1,Go(t,o),i===`backwards`&&e!==null?(ll(e),Lc(e,t,r,n),ll(e)):Lc(e,t,r,n),r=M?ta:0,!s&&e!==null&&e.flags&128)a:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&ol(e,n,t);else if(e.tag===19)ol(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break a;for(;e.sibling===null;){if(e.return===null||e.return===t)break a;e=e.return}e.sibling.return=e.return,e=e.sibling}switch(i){case`backwards`:n=sl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null,ll(t)),cl(t,!0,i,null,a,r);break;case`unstable_legacy-backwards`:for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&qo(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}cl(t,!0,n,null,a,r);break;case`together`:cl(t,!1,null,null,void 0,r);break;case`independent`:t.memoizedState=null;break;default:n=sl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),cl(t,!1,i,n,a,r)}return t.child}function dl(e,t,n){var r=t.pendingProps;return Ea(t,t.type,r.value),Lc(e,t,r.children,n),t.child}function fl(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),sd|=t.lanes,(n&t.childLanes)===0){if(e!==null){if(Aa(e,t,n,!1),(n&t.childLanes)===0)return null}else return null}if(e!==null&&t.child!==e.child)throw Error(o(153));if(t.child!==null){for(e=t.child,n=Ui(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=Ui(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function pl(e,t){return(e.lanes&t)!==0||(e=e.dependencies,!!(e!==null&&N(e)))}function ml(e,t,n){switch(t.tag){case 3:Ie(t,t.stateNode.containerInfo),Ea(t,Ra,e.memoizedState.cache),ba();break;case 27:case 5:Re(t);break;case 4:Ie(t,t.stateNode.containerInfo);break;case 10:Ea(t,t.type,t.memoizedProps.value);break;case 31:if(t.memoizedState!==null)return t.flags|=128,Bo(t),null;break;case 13:var r=t.memoizedState;if(r!==null){if(r.dehydrated!==null)return zo(t),t.flags|=128,null;r=Aa(e,t,n,!1);var i=t.child.childLanes;return r||(n&i)!==0?tl(e,t,n):(zo(t),e=fl(e,t,n),e===null?null:e.sibling)}zo(t);break;case 19:if(t.flags&128)return ul(e,t,n);if(i=!!(e.flags&128),r=(n&t.childLanes)!==0,r||=(Aa(e,t,n,!1),(n&t.childLanes)!==0),i){if(r)return ul(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),Go(t,Wo.current),r)break;return null;case 22:return t.lanes=0,Vc(e,t,n,t.pendingProps);case 24:Ea(t,Ra,e.memoizedState.cache)}return fl(e,t,n)}function hl(e,t,n){if(e!==null){if(e.memoizedProps!==t.pendingProps)Ic=!0;else{if(!pl(e,n)&&!(t.flags&128))return Ic=!1,ml(e,t,n);Ic=!!(e.flags&131072)}}else Ic=!1,M&&t.flags&1048576&&ca(t,ta,t.index);switch(t.lanes=0,t.tag){case 16:a:{var r=t.pendingProps;if(e=co(t.elementType),t.type=e,typeof e==`function`)Hi(e)?(r=Tc(e,r),t.tag=1,t=Xc(null,t,e,r,n)):(t.tag=0,t=Jc(null,t,e,r,n));else{if(e!=null){var i=e.$$typeof;if(i===pe){t.tag=11,t=Rc(null,t,e,r,n);break a}if(i===ge){t.tag=14,t=zc(null,t,e,r,n);break a}if(i===S){t.tag=10,t.type=e,t=dl(null,t,n);break a}}throw t=Te(e)||e,Error(o(306,t,``))}}return t;case 0:return Jc(e,t,t.type,t.pendingProps,n);case 1:return r=t.type,i=Tc(r,t.pendingProps),Xc(e,t,r,i,n);case 3:a:{if(Ie(t,t.stateNode.containerInfo),e===null)throw Error(o(387));r=t.pendingProps;var a=t.memoizedState;i=a.element,Co(e,t),Ao(t,r,null,n);var s=t.memoizedState;if(r=s.cache,Ea(t,Ra,r),r!==a.cache&&ka(t,[Ra],n,!0),ko(),r=s.element,a.isDehydrated){if(a={element:r,isDehydrated:!1,cache:s.cache},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){t=Zc(e,t,r,n);break a}if(r!==i){i=Zi(Error(o(424)),t),Sa(i),t=Zc(e,t,r,n);break a}switch(e=t.stateNode.containerInfo,e.nodeType){case 9:e=e.body;break;default:e=e.nodeName===`HTML`?e.ownerDocument.body:e}for(j=lm(e.firstChild),fa=t,M=!0,pa=null,ma=!0,n=bo(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling}else{if(ba(),r===i){t=fl(e,t,n);break a}Lc(e,t,r,n)}t=t.child}return t;case 26:return qc(e,t),e===null?(n=Nm(t.type,null,t.pendingProps,null))?t.memoizedState=n:M||(t.stateNode=fp(t.type,t.pendingProps,Pe.current,t)):t.memoizedState=Nm(t.type,e.memoizedProps,t.pendingProps,e.memoizedState),null;case 27:return Re(t),e===null&&M&&(r=t.stateNode=hm(t.type,t.pendingProps,Pe.current),fa=t,ma=!0,i=j,Sp(t.type)?(um=i,j=lm(r.firstChild)):j=i),Lc(e,t,t.pendingProps.children,n),qc(e,t),e===null&&(t.flags|=4194304),t.child;case 5:return e===null&&M&&((i=r=j)&&(r=rm(r,t.type,t.pendingProps,ma),r===null?i=!1:(t.stateNode=r,fa=t,j=lm(r.firstChild),ma=!1,i=!0)),i||ga(t)),Re(t),i=t.type,a=t.pendingProps,s=e===null?null:e.memoizedProps,r=a.children,pp(i,a)?r=null:s!==null&&pp(i,s)&&(t.flags|=32),t.memoizedState!==null&&(i=rs(e,t,os,null,null,n),sh._currentValue=i),qc(e,t),Lc(e,t,r,n),t.child;case 6:return e===null&&M&&((e=n=j)&&(n=im(n,t.pendingProps,ma),n===null?e=!1:(t.stateNode=n,fa=t,j=null,e=!0)),e||ga(t)),null;case 13:return tl(e,t,n);case 4:return Ie(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=yo(t,null,r,n):Lc(e,t,r,n),t.child;case 11:return Rc(e,t,t.type,t.pendingProps,n);case 7:return r=t.pendingProps,qc(e,t),Lc(e,t,r,n),t.child;case 8:return Lc(e,t,t.pendingProps.children,n),t.child;case 12:return Lc(e,t,t.pendingProps.children,n),t.child;case 10:return dl(e,t,n);case 9:return i=t.type._context,r=t.pendingProps.children,ja(t),i=Ma(i),r=r(i),t.flags|=1,Lc(e,t,r,n),t.child;case 14:return zc(e,t,t.type,t.pendingProps,n);case 15:return Bc(e,t,t.type,t.pendingProps,n);case 19:return ul(e,t,n);case 31:return Kc(e,t,n);case 22:return Vc(e,t,n,t.pendingProps);case 24:return ja(t),r=Ma(Ra),e===null?(i=$a(),i===null&&(i=G,a=za(),i.pooledCache=a,a.refCount++,a!==null&&(i.pooledCacheLanes|=n),i=a),t.memoizedState={parent:r,cache:i},So(t),Ea(t,Ra,i)):((e.lanes&n)!==0&&(Co(e,t),Ao(t,null,null,n),ko()),i=e.memoizedState,a=t.memoizedState,i.parent===r?(r=a.cache,Ea(t,Ra,r),r!==i.cache&&ka(t,[Ra],n,!0)):(i={parent:r,cache:r},t.memoizedState=i,t.lanes===0&&(t.memoizedState=t.updateQueue.baseState=i),Ea(t,Ra,r))),Lc(e,t,t.pendingProps.children,n),t.child;case 30:return t.stateNode===null&&(t.stateNode={autoName:null,paired:null,clones:null,ref:null}),r=t.pendingProps,r.name!=null&&r.name!==`auto`?t.flags|=e===null?18882560:18874368:M&&la(t),e!==null&&e.memoizedProps.name!==r.name?t.flags|=4194816:qc(e,t),Lc(e,t,r.children,n),t.child;case 29:throw t.pendingProps}throw Error(o(156,t.tag))}function gl(e){e.flags|=4}function _l(e,t,n,r,i){var a;if((a=!!(e.mode&32))&&(a=n===null?Jm(t,r):Jm(t,r)&&(r.src!==n.src||r.srcSet!==n.srcSet)),a){if(e.flags|=16777216,(i&335544128)===i){if(e.stateNode.complete)e.flags|=8192;else if(Ud())e.flags|=8192;else throw lo=ao,ro}}else e.flags&=-16777217}function vl(e,t){if(t.type!==`stylesheet`||t.state.loading&4)e.flags&=-16777217;else if(e.flags|=16777216,!Ym(t)){if(Ud())e.flags|=8192;else throw lo=ao,ro}}function yl(e,t){t!==null&&(e.flags|=4),e.flags&16384&&(t=e.tag===22?536870912:xt(),e.lanes|=t,dd|=t)}function bl(e,t){if(!M)switch(e.tailMode){case`visible`:break;case`collapsed`:for(var n=e.tail,r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null;break;default:for(t=e.tail,n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null}}function V(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&1206910976,r|=i.flags&1206910976,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function xl(e,t,n){var r=t.pendingProps;switch(ua(t),t.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return V(t),null;case 1:return V(t),null;case 3:return n=t.stateNode,r=null,e!==null&&(r=e.memoizedState.cache),t.memoizedState.cache!==r&&(t.flags|=2048),Da(Ra),Le(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(e===null||e.child===null)&&(ya(t)?gl(t):e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,xa())),V(t),null;case 26:var i=t.type,a=t.memoizedState;return e===null?(gl(t),a===null?(V(t),_l(t,i,null,r,n)):(V(t),vl(t,a))):a?a===e.memoizedState?(V(t),t.flags&=-16777217):(gl(t),V(t),vl(t,a)):(e=e.memoizedProps,e!==r&&gl(t),V(t),_l(t,i,e,r,n)),null;case 27:if(ze(t),n=Pe.current,i=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(!r){if(t.stateNode===null)throw Error(o(166));return V(t),t.subtreeFlags&=-33554433,null}e=Me.current,ya(t)?_a(t,e):(e=hm(i,r,n),t.stateNode=e,gl(t))}return V(t),t.subtreeFlags&=-33554433,null;case 5:if(ze(t),i=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(!r){if(t.stateNode===null)throw Error(o(166));return V(t),t.subtreeFlags&=-33554433,null}if(a=Me.current,ya(t))_a(t,a);else{var s=lp(Pe.current);switch(a){case 1:a=s.createElementNS(`http://www.w3.org/2000/svg`,i);break;case 2:a=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,i);break;default:switch(i){case`svg`:a=s.createElementNS(`http://www.w3.org/2000/svg`,i);break;case`math`:a=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,i);break;case`script`:a=s.createElement(`div`),a.innerHTML=`<script><\/script>`,a=a.removeChild(a.firstChild);break;case`select`:a=typeof r.is==`string`?s.createElement(`select`,{is:r.is}):s.createElement(`select`),r.multiple?a.multiple=!0:r.size&&(a.size=r.size);break;default:a=typeof r.is==`string`?s.createElement(i,{is:r.is}):s.createElement(i)}}a[Nt]=t,a[Pt]=r;a:for(s=t.child;s!==null;){if(s.tag===5||s.tag===6)a.appendChild(s.stateNode);else if(s.tag!==4&&s.tag!==27&&s.child!==null){s.child.return=s,s=s.child;continue}if(s===t)break a;for(;s.sibling===null;){if(s.return===null||s.return===t)break a;s=s.return}s.sibling.return=s.return,s=s.sibling}t.stateNode=a;a:switch(np(a,i,r),i){case`button`:case`input`:case`select`:case`textarea`:r=!!r.autoFocus;break a;case`img`:r=!0;break a;default:r=!1}r&&gl(t)}}return V(t),t.subtreeFlags&=-33554433,_l(t,t.type,e===null?null:e.memoizedProps,t.pendingProps,n),null;case 6:if(e&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(typeof r!=`string`&&t.stateNode===null)throw Error(o(166));if(e=Pe.current,ya(t)){if(e=t.stateNode,n=t.memoizedProps,r=null,i=fa,i!==null)switch(i.tag){case 27:case 5:r=i.memoizedProps}e[Nt]=t,e=!!(e.nodeValue===n||r!==null&&!0===r.suppressHydrationWarning||ep(e.nodeValue,n)),e||ga(t,!0)}else e=lp(e).createTextNode(r),e[Nt]=t,t.stateNode=e}return V(t),null;case 31:if(n=t.memoizedState,e===null||e.memoizedState!==null){if(r=ya(t),n!==null){if(e===null){if(!r)throw Error(o(318));if(e=t.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(557));e[Nt]=t}else ba(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;V(t),e=!1}else n=xa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=n),e=!0;if(!e)return t.flags&256?(Uo(t),t):(Uo(t),null);if(t.flags&128)throw Error(o(558))}return V(t),null;case 13:if(r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(i=ya(t),r!==null&&r.dehydrated!==null){if(e===null){if(!i)throw Error(o(318));if(i=t.memoizedState,i=i===null?null:i.dehydrated,!i)throw Error(o(317));i[Nt]=t}else ba(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;V(t),i=!1}else i=xa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=i),i=!0;if(!i)return t.flags&256?(Uo(t),t):(Uo(t),null)}return Uo(t),t.flags&128?(t.lanes=n,t):(n=r!==null,e=e!==null&&e.memoizedState!==null,n&&(r=t.child,i=null,r.alternate!==null&&r.alternate.memoizedState!==null&&r.alternate.memoizedState.cachePool!==null&&(i=r.alternate.memoizedState.cachePool.pool),a=null,r.memoizedState!==null&&r.memoizedState.cachePool!==null&&(a=r.memoizedState.cachePool.pool),a!==i&&(r.flags|=2048)),n!==e&&n&&(t.child.flags|=8192),yl(t,t.updateQueue),V(t),null);case 4:return Le(),e===null&&Wf(t.stateNode.containerInfo),t.flags|=67108864,V(t),null;case 10:return Da(t.type),V(t),null;case 19:if(Ko(t),r=t.memoizedState,r===null)return V(t),null;if(i=!!(t.flags&128),a=r.rendering,a===null){if(i)bl(r,!1);else{if(Y!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(a=qo(e),a!==null){for(t.flags|=128,bl(r,!1),e=a.updateQueue,t.updateQueue=e,yl(t,e),t.subtreeFlags=0,e=n,n=t.child;n!==null;)Wi(n,e),n=n.sibling;return Go(t,Wo.current&1|2),M&&sa(t,r.treeForkCount),t.child}e=e.sibling}r.tail!==null&&Qe()>_d&&(t.flags|=128,i=!0,bl(r,!1),t.lanes=4194304)}}else{if(!i){if(e=qo(a),e!==null){if(t.flags|=128,i=!0,e=e.updateQueue,t.updateQueue=e,yl(t,e),bl(r,!0),r.tail===null&&r.tailMode!==`collapsed`&&r.tailMode!==`visible`&&!a.alternate&&!M)return V(t),null}else 2*Qe()-r.renderingStartTime>_d&&n!==536870912&&(t.flags|=128,i=!0,bl(r,!1),t.lanes=4194304)}r.isBackwards?(a.sibling=t.child,t.child=a):(e=r.last,e===null?t.child=a:e.sibling=a,r.last=a)}if(r.tail!==null){e=r.tail;a:{for(n=e;n!==null;){if(n.alternate!==null){n=!1;break a}n=n.sibling}n=!0}return r.rendering=e,r.tail=e.sibling,r.renderingStartTime=Qe(),e.sibling=null,a=Wo.current,a=i?a&1|2:a&1,r.tailMode===`visible`||r.tailMode===`collapsed`||!n||M?Go(t,a):(n=a,E(Lo,t),E(Wo,n),Ro===null&&(Ro=t)),M&&sa(t,r.treeForkCount),e}return V(t),null;case 22:case 23:return Uo(t),Io(),r=t.memoizedState!==null,e===null?r&&(t.flags|=8192):e.memoizedState!==null!==r&&(t.flags|=8192),r?n&536870912&&!(t.flags&128)&&(V(t),t.subtreeFlags&6&&(t.flags|=8192)):V(t),n=t.updateQueue,n!==null&&yl(t,n.retryQueue),n=null,e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),r=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(r=t.memoizedState.cachePool.pool),r!==n&&(t.flags|=2048),e!==null&&je(Qa),null;case 24:return n=null,e!==null&&(n=e.memoizedState.cache),t.memoizedState.cache!==n&&(t.flags|=2048),Da(Ra),V(t),null;case 25:return null;case 30:return t.flags|=33554432,V(t),null}throw Error(o(156,t.tag))}function Sl(e,t){switch(ua(t),t.tag){case 1:return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return Da(Ra),Le(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 26:case 27:case 5:return ze(t),null;case 31:if(t.memoizedState!==null){if(Uo(t),t.alternate===null)throw Error(o(340));ba()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 13:if(Uo(t),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(o(340));ba()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return Ko(t),e=t.flags,e&65536?(t.flags=e&-65537|128,e=t.memoizedState,e!==null&&(e.rendering=null,e.tail=null),t.flags|=4,t):null;case 4:return Le(),null;case 10:return Da(t.type),null;case 22:case 23:return Uo(t),Io(),e!==null&&je(Qa),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 24:return Da(Ra),null;case 25:return null;default:return null}}function Cl(e,t){switch(ua(t),t.tag){case 3:Da(Ra),Le();break;case 26:case 27:case 5:ze(t);break;case 4:Le();break;case 31:t.memoizedState!==null&&Uo(t);break;case 13:Uo(t);break;case 19:Ko(t);break;case 10:Da(t.type);break;case 22:case 23:Uo(t),Io(),e!==null&&je(Qa);break;case 24:Da(Ra)}}function wl(e,t){try{var n=t.updateQueue,r=n===null?null:n.lastEffect;if(r!==null){var i=r.next;n=i;do{if((n.tag&e)===e){r=void 0;var a=n.create,o=n.inst;r=a(),o.destroy=r}n=n.next}while(n!==i)}}catch(e){Z(t,t.return,e)}}function Tl(e,t,n){try{var r=t.updateQueue,i=r===null?null:r.lastEffect;if(i!==null){var a=i.next;r=a;do{if((r.tag&e)===e){var o=r.inst,s=o.destroy;if(s!==void 0){o.destroy=void 0,i=t;var c=n,l=s;try{l()}catch(e){Z(i,c,e)}}}r=r.next}while(r!==a)}}catch(e){Z(t,t.return,e)}}function El(e){var t=e.updateQueue;if(t!==null){var n=e.stateNode;try{Mo(t,n)}catch(t){Z(e,e.return,t)}}}function Dl(e,t,n){n.props=Tc(e.type,e.memoizedProps),n.state=e.memoizedState;try{n.componentWillUnmount()}catch(n){Z(e,t,n)}}function Ol(e,t){try{var n=e.ref;if(n!==null){switch(e.tag){case 26:case 27:case 5:var r=e.stateNode;break;case 30:var i=e.stateNode,a=Ei(e.memoizedProps,i);(i.ref===null||i.ref.name!==a)&&(i.ref=Pp(a)),r=i.ref;break;case 7:if(e.stateNode===null){var o=new Fp(e);m(e.child,!1,Qp,o,void 0,void 0),e.stateNode=o}r=e.stateNode;break;default:r=e.stateNode}typeof n==`function`?e.refCleanup=n(r):n.current=r}}catch(n){Z(e,t,n)}}function kl(e,t){var n=e.ref,r=e.refCleanup;if(n!==null){if(typeof r==`function`)try{r()}catch(n){Z(e,t,n)}finally{e.refCleanup=null,e=e.alternate,e!=null&&(e.refCleanup=null)}else if(typeof n==`function`)try{n(null)}catch(n){Z(e,t,n)}else n.current=null}}function Al(e,t){if((e.tag===5||e.tag===27||e.tag===6)&&e.alternate===null&&t!==null)for(var n=0;n<t.length;n++)em(e.stateNode,t[n])}function jl(e){for(var t=e.return;t!==null&&(Pl(t)&&em(e.stateNode,t.stateNode),!Nl(t));)t=t.return}function Ml(e){for(var t=e.return;t!==null&&(Pl(t)&&tm(e.stateNode,t.stateNode),!Nl(t));)t=t.return}function Nl(e){return e.tag===5||e.tag===3||e.tag===27}function Pl(e){return e&&e.tag===7&&e.stateNode!==null}function Fl(e){var t=e.type,n=e.memoizedProps,r=e.stateNode;try{a:switch(t){case`button`:case`input`:case`select`:case`textarea`:n.autoFocus&&r.focus();break a;case`img`:n.src?r.src=n.src:n.srcSet&&(r.srcset=n.srcSet)}}catch(t){Z(e,e.return,t)}}function Il(e,t,n){try{var r=e.stateNode;ip(r,e.type,n,t),r[Pt]=t}catch(t){Z(e,e.return,t)}}function Ll(e){return e.tag===5||e.tag===3||e.tag===26||e.tag===27&&Sp(e.type)||e.tag===4}function Rl(e){a:for(;;){for(;e.sibling===null;){if(e.return===null||Ll(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.tag===27&&Sp(e.type)||e.flags&2||e.child===null||e.tag===4)continue a;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function zl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?(n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n).insertBefore(i,t):(t=n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n,t.appendChild(i),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=On)),Al(e,r),k=!0;else if(i!==4&&(i===27&&(Al(e,r),r=null,Sp(e.type)&&(n=e.stateNode,t=null)),e=e.child,e!==null))for(zl(e,t,n,r),e=e.sibling;e!==null;)zl(e,t,n,r),e=e.sibling}function Bl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?n.insertBefore(i,t):n.appendChild(i),Al(e,r),k=!0;else if(i!==4&&(i===27&&(Al(e,r),r=null,Sp(e.type)&&(n=e.stateNode)),e=e.child,e!==null))for(Bl(e,t,n,r),e=e.sibling;e!==null;)Bl(e,t,n,r),e=e.sibling}function Vl(e){var t=e.stateNode,n=e.memoizedProps;try{for(var r=e.type,i=t.attributes;i.length;)t.removeAttributeNode(i[0]);np(t,r,n),t[Nt]=e,t[Pt]=n}catch(t){Z(e,e.return,t)}}var Hl=!1,Ul=null;function Wl(e){(e.tag===30||e.subtreeFlags&33554432)&&(Hl=!0)}var Gl=null;function Kl(){var e=Gl;return Gl=null,e}var ql=0;function Jl(e,t,n,r,i){return ql=0,Yl(e.child,t,n,r,i)}function Yl(e,t,n,r,i){for(var a=!1;e!==null;){if(e.tag===5){var o=e.stateNode;if(r!==null){var s=Op(o);r.push(s),s.view&&(a=!0)}else a||Op(o).view&&(a=!0);Hl=!0,Tp(o,ql===0?t:t+`_`+ql,n),ql++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&i||Yl(e.child,t,n,r,i)&&(a=!0));e=e.sibling}return a}function Xl(e,t){for(;e!==null;)e.tag===5?Ep(e.stateNode,e.memoizedProps):(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&t||Xl(e.child,t)),e=e.sibling}function Zl(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if((e.tag!==22||e.memoizedState===null)&&(Zl(e),e.tag===30&&e.flags&18874368&&e.stateNode.paired)){var t=e.memoizedProps;if(t.name==null||t.name===`auto`)throw Error(o(544));var n=t.name;t=Oi(t.default,t.share),t!==`none`&&(Jl(e,n,t,null,!1)||Xl(e.child,!1))}e=e.sibling}}function Ql(e,t){if(e.tag===30){var n=e.stateNode,r=e.memoizedProps,i=Ei(r,n),a=Oi(r.default,n.paired?r.share:r.enter);a===`none`?Zl(e):Jl(e,i,a,null,!1)?(Zl(e),n.paired||t||Nd(e,r.onEnter)):Xl(e.child,!1)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Ql(e,t),e=e.sibling;else Zl(e)}function $l(e){if(Ul!==null&&Ul.size!==0){var t=Ul;if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var n=e.memoizedProps,r=n.name;if(r!=null&&r!==`auto`){var i=t.get(r);if(i!==void 0){var a=Oi(n.default,n.share);if(a!==`none`&&(Jl(e,r,a,null,!1)?(a=e.stateNode,i.paired=a,a.paired=i,Nd(e,n.onShare)):Xl(e.child,!1)),t.delete(r),t.size===0)break}}}$l(e)}e=e.sibling}}}function eu(e){if(e.tag===30){var t=e.memoizedProps,n=Ei(t,e.stateNode),r=Ul===null?void 0:Ul.get(n),i=Oi(t.default,r===void 0?t.exit:t.share);i!==`none`&&(Jl(e,n,i,null,!1)?r===void 0?Nd(e,t.onExit):(i=e.stateNode,r.paired=i,i.paired=r,Ul.delete(n),Nd(e,t.onShare)):Xl(e.child,!1)),Ul!==null&&$l(e)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)eu(e),e=e.sibling;else Ul!==null&&$l(e)}function tu(e){for(e=e.child;e!==null;){if(e.tag===30){var t=e.memoizedProps,n=Ei(t,e.stateNode);t=Oi(t.default,t.update),e.flags&=-5,t!==`none`&&Jl(e,n,t,e.memoizedState=[],!1)}else e.subtreeFlags&33554432&&tu(e);e=e.sibling}}function nu(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var t=e.stateNode;t.paired!==null&&(t.paired=null,Xl(e.child,!1))}nu(e)}e=e.sibling}}function ru(e){if(e.tag===30)e.stateNode.paired=null,Xl(e.child,!1),nu(e);else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)ru(e),e=e.sibling;else nu(e)}function iu(e){for(e=e.child;e!==null;)e.tag===30?Xl(e.child,!1):e.subtreeFlags&33554432&&iu(e),e=e.sibling}function au(e,t,n,r,i,a,o){for(var s=!1;t!==null;){if(t.tag===5){var c=t.stateNode;if(a!==null&&ql<a.length){var l=a[ql],u=Op(c);(l.view||u.view)&&(s=!0);var d;if(d=!(e.flags&4)){if(u.clip)d=!0;else{d=l.rect;var f=u.rect;d=d.y!==f.y||d.x!==f.x||d.height!==f.height||d.width!==f.width}}d&&(e.flags|=4),u.abs?u=!l.abs:(l=l.rect,u=u.rect,u=l.height!==u.height||l.width!==u.width),u&&(e.flags|=32)}else e.flags|=32;e.flags&4&&Tp(c,ql===0?n:n+`_`+ql,i),s&&e.flags&4||(Gl===null&&(Gl=[]),Gl.push(c,ql===0?r:r+`_`+ql,t.memoizedProps)),ql++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&o?e.flags|=t.flags&32:au(e,t.child,n,r,i,a,o)&&(s=!0));t=t.sibling}return s}function ou(e,t){for(e=e.child;e!==null;){if(e.tag===30){var n=e.memoizedProps,r=e.stateNode,i=Ei(n,r),a=Oi(n.default,n.update);if(t){r=r.clones;var o=r===null?null:r.map(kp)}else o=e.memoizedState,e.memoizedState=null;r=e;var s=e.child;ql=0,i=au(r,s,i,i,a,o,!1),e.flags&4&&i&&(t||Nd(e,n.onUpdate))}else e.subtreeFlags&33554432&&ou(e,t);e=e.sibling}}var su=!1,H=!1,cu=!1,lu=!1,uu=typeof WeakSet==`function`?WeakSet:Set,du=null,fu=!1,pu=!1,mu=!1,hu=!1;function gu(e,t,n){if(e=e.containerInfo,sp=gh,e=ni(e),ri(e)){if(`selectionStart`in e)var r={start:e.selectionStart,end:e.selectionEnd};else a:{r=(r=e.ownerDocument)&&r.defaultView||window;var i=r.getSelection&&r.getSelection();if(i&&i.rangeCount!==0){r=i.anchorNode;var a=i.anchorOffset,o=i.focusNode;i=i.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break a}var s=0,c=-1,l=-1,u=0,d=0,f=e,p=null;b:for(;;){for(var m;f!==r||a!==0&&f.nodeType!==3||(c=s+a),f!==o||i!==0&&f.nodeType!==3||(l=s+i),f.nodeType===3&&(s+=f.nodeValue.length),(m=f.firstChild)!==null;)p=f,f=m;for(;;){if(f===e)break b;if(p===r&&++u===a&&(c=s),p===o&&++d===i&&(l=s),(m=f.nextSibling)!==null)break;f=p,p=f.parentNode}f=m}r=c===-1||l===-1?null:{start:c,end:l}}else r=null}r||={start:0,end:0}}else r=null;for(cp={focusedElem:e,selectionRange:r},gh=!1,n=(n&335544064)===n,du=t,t=n?9270:1024;du!==null;){if(e=du,n&&(r=e.deletions,r!==null))for(a=0;a<r.length;a++)n&&eu(r[a]);if(e.alternate===null&&e.flags&2)n&&Wl(e),_u(n);else{if(e.tag===22){if(r=e.alternate,e.memoizedState!==null){r!==null&&r.memoizedState===null&&n&&eu(r),_u(n);continue}if(r!==null&&r.memoizedState!==null){n&&Wl(e),_u(n);continue}}r=e.child,(e.subtreeFlags&t)!==0&&r!==null?(r.return=e,du=r):(n&&tu(e),_u(n))}}Ul=null}function _u(e){for(;du!==null;){var t=du,n=e,r=t.alternate,i=t.flags;switch(t.tag){case 0:case 11:case 15:break;case 1:if(i&1024&&r!==null){n=void 0,i=r.memoizedProps,r=r.memoizedState;var a=t.stateNode;try{var s=Tc(t.type,i);n=a.getSnapshotBeforeUpdate(s,r),a.__reactInternalSnapshotBeforeUpdate=n}catch(e){Z(t,t.return,e)}}break;case 3:if(i&1024){if(r=t.stateNode.containerInfo,n=r.nodeType,n===9)nm(r);else if(n===1)switch(r.nodeName){case`HEAD`:case`HTML`:case`BODY`:nm(r);break;default:r.textContent=``}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&r!==null&&(n=Ei(r.memoizedProps,r.stateNode),i=t.memoizedProps,i=Oi(i.default,i.update),i!==`none`&&Jl(r,n,i,r.memoizedState=[],!0));break;default:if(i&1024)throw Error(o(163))}if(r=t.sibling,r!==null){r.return=t.return,du=r;break}du=t.return}}function vu(e,t,n){var r=n.flags;switch(n.tag){case 0:case 11:case 15:Lu(e,n),r&4&&wl(5,n);break;case 1:if(Lu(e,n),r&4){if(e=n.stateNode,t===null)try{e.componentDidMount()}catch(e){Z(n,n.return,e)}else{var i=Tc(n.type,t.memoizedProps);t=t.memoizedState;try{e.componentDidUpdate(i,t,e.__reactInternalSnapshotBeforeUpdate)}catch(e){Z(n,n.return,e)}}}r&64&&El(n),r&512&&Ol(n,n.return);break;case 3:if(Lu(e,n),r&64&&(e=n.updateQueue,e!==null)){if(t=null,n.child!==null)switch(n.child.tag){case 27:case 5:t=n.child.stateNode;break;case 1:t=n.child.stateNode}try{Mo(e,t)}catch(e){Z(n,n.return,e)}}break;case 27:t===null&&r&4&&Vl(n);case 26:case 5:Lu(e,n),t===null&&r&4&&Fl(n),r&512&&Ol(n,n.return);break;case 12:Lu(e,n);break;case 31:Lu(e,n),r&4&&Eu(e,n);break;case 13:Lu(e,n),r&4&&Du(e,n),r&64&&(e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null&&(n=_f.bind(null,n),cm(e,n))));break;case 22:if(r=n.memoizedState!==null||su,!r){var a=t!==null&&t.memoizedState!==null||H;t=su,i=H,su=r,(H=a)&&!i?(r=2,n.subtreeFlags&8772&&(r|=1),zu(e,n,r)):Lu(e,n),su=t,H=i}break;case 30:Lu(e,n),r&512&&Ol(n,n.return);break;case 7:r&512&&Ol(n,n.return);default:Lu(e,n)}}function yu(e,t){for(e=e.child;e!==null;)bu(e,t),e=e.sibling}function bu(e,t){switch(e.tag){case 5:case 26:try{var n=e.stateNode;if(t){var r=n.style;typeof r.setProperty==`function`?r.setProperty(`display`,`none`,`important`):r.display=`none`}else{var i=e.stateNode,a=e.memoizedProps.style,o=a!=null&&a.hasOwnProperty(`display`)?a.display:null;i.style.display=o==null||typeof o==`boolean`?``:(``+o).trim()}}catch(t){Z(e,e.return,t)}xu(e,t);break;case 6:try{e.stateNode.nodeValue=t?``:e.memoizedProps,k=!0}catch(t){Z(e,e.return,t)}break;case 18:try{var s=e.stateNode;t?wp(s,!0):wp(e.stateNode,!1)}catch(t){Z(e,e.return,t)}break;case 22:case 23:e.memoizedState===null&&yu(e,t);break;default:yu(e,t)}}function xu(e,t){if(e.subtreeFlags&67108864)for(e=e.child;e!==null;){a:{var n=e,r=t;switch(n.tag){case 4:bu(n,r);break a;case 22:n.memoizedState===null&&xu(n,r);break a;default:xu(n,r)}}e=e.sibling}}function Su(e){var t=e.alternate;t!==null&&(e.alternate=null,Su(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&Ht(t)),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}var U=null,Cu=!1;function wu(e,t,n){for(n=n.child;n!==null;)Tu(e,t,n),n=n.sibling}function Tu(e,t,n){if(ct&&typeof ct.onCommitFiberUnmount==`function`)try{ct.onCommitFiberUnmount(st,n)}catch{}switch(n.tag){case 26:H||kl(n,t),wu(e,t,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!H&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:H||kl(n,t),Ml(n);var r=U,i=Cu;Sp(n.type)&&(U=n.stateNode,Cu=!1),wu(e,t,n),gm(n.stateNode,n.type,n.memoizedProps),U=r,Cu=i;break;case 5:H||kl(n,t),Ml(n);case 6:if(n.tag===6&&Ml(n),r=U,i=Cu,U=null,wu(e,t,n),U=r,Cu=i,U!==null){if(Cu)try{(U.nodeType===9?U.body:U.nodeName===`HTML`?U.ownerDocument.body:U).removeChild(n.stateNode),k=!0}catch(e){Z(n,t,e)}else try{U.removeChild(n.stateNode),k=!0}catch(e){Z(n,t,e)}}break;case 18:U!==null&&(Cu?(e=U,Cp(e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,n.stateNode),Hh(e)):Cp(U,n.stateNode));break;case 4:r=U,i=Cu,U=n.stateNode.containerInfo,Cu=!0,wu(e,t,n),U=r,Cu=i;break;case 0:case 11:case 14:case 15:Tl(2,n,t),H||Tl(4,n,t),wu(e,t,n);break;case 1:H||(kl(n,t),r=n.stateNode,typeof r.componentWillUnmount==`function`&&Dl(n,t,r)),wu(e,t,n);break;case 21:wu(e,t,n);break;case 22:H=(r=H)||n.memoizedState!==null,wu(e,t,n),H=r;break;case 30:kl(n,t),wu(e,t,n);break;case 7:H||kl(n,t),wu(e,t,n);break;default:wu(e,t,n)}}function Eu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null))){e=e.dehydrated;try{Hh(e)}catch(e){Z(t,t.return,e)}}}function Du(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null&&(e=e.dehydrated,e!==null))))try{Hh(e)}catch(e){Z(t,t.return,e)}}function Ou(e){switch(e.tag){case 31:case 13:case 19:var t=e.stateNode;return t===null&&(t=e.stateNode=new uu),t;case 22:return e=e.stateNode,t=e._retryCache,t===null&&(t=e._retryCache=new uu),t;default:throw Error(o(435,e.tag))}}function ku(e,t){var n=Ou(e);t.forEach(function(t){if(!n.has(t)){n.add(t);var r=vf.bind(null,e,t);t.then(r,r)}})}function Au(e,t,n){var r=t.deletions;if(r!==null)for(var i=0;i<r.length;i++){var a=r[i],s=e,c=t,l=c;a:for(;l!==null;){switch(l.tag){case 27:if(Sp(l.type)){U=l.stateNode,Cu=!1;break a}break;case 5:U=l.stateNode,Cu=!1;break a;case 3:case 4:U=l.stateNode.containerInfo,Cu=!0;break a}l=l.return}if(U===null)throw Error(o(160));Tu(s,c,a),U=null,Cu=!1,s=a.alternate,s!==null&&(s.return=null),a.return=null}if(t.subtreeFlags&13886)for(t=t.child;t!==null;)Mu(t,e,n),t=t.sibling}var ju=null;function Mu(e,t,n){var r=e.alternate,i=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(i&4&&(r=e.updateQueue,r=r===null?null:r.events,r!==null))for(var a=0;a<r.length;a++){var s=r[a];s.ref.impl=s.nextImpl}Au(t,e,n),Nu(e),i&4&&(Tl(3,e,e.return),wl(3,e),Tl(5,e,e.return));break;case 1:Au(t,e,n),Nu(e),i&512&&(H||r===null||kl(r,r.return)),i&64&&su&&(e=e.updateQueue,e!==null&&(t=e.callbacks,t!==null&&(n=e.shared.hiddenCallbacks,e.shared.hiddenCallbacks=n===null?t:n.concat(t))));break;case 26:if(a=ju,Au(t,e,n),Nu(e),i&512&&(H||r===null||kl(r,r.return)),i&4){if(i=r===null?null:r.memoizedState,n=e.memoizedState,r===null){if(n===null){if(e.stateNode===null){if(su)e.stateNode=fp(e.type,e.memoizedProps,t.containerInfo,e);else{a:{t=e.type,n=e.memoizedProps,i=a.ownerDocument||a;b:switch(t){case`title`:r=i.getElementsByTagName(`title`)[0],(!r||r[Bt]||r[Nt]||r.namespaceURI===`http://www.w3.org/2000/svg`||r.hasAttribute(`itemprop`))&&(r=i.createElement(t),i.head.insertBefore(r,i.querySelector(`head > title`))),np(r,t,n),r[Nt]=e,O(r),t=r;break a;case`link`:if(a=Gm(`link`,`href`,i).get(t+(n.href||``))){for(s=0;s<a.length;s++)if(r=a[s],r.getAttribute(`href`)===(n.href==null||n.href===``?null:n.href)&&r.getAttribute(`rel`)===(n.rel==null?null:n.rel)&&r.getAttribute(`title`)===(n.title==null?null:n.title)&&r.getAttribute(`crossorigin`)===(n.crossOrigin==null?null:n.crossOrigin)){a.splice(s,1);break b}}r=i.createElement(t),np(r,t,n),i.head.appendChild(r);break;case`meta`:if(a=Gm(`meta`,`content`,i).get(t+(n.content||``))){for(s=0;s<a.length;s++)if(r=a[s],r.getAttribute(`content`)===(n.content==null?null:``+n.content)&&r.getAttribute(`name`)===(n.name==null?null:n.name)&&r.getAttribute(`property`)===(n.property==null?null:n.property)&&r.getAttribute(`http-equiv`)===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute(`charset`)===(n.charSet==null?null:n.charSet)){a.splice(s,1);break b}}r=i.createElement(t),np(r,t,n),i.head.appendChild(r);break;default:throw Error(o(468,t))}r[Nt]=e,O(r),t=r}e.stateNode=t}}else su||Km(a,e.type,e.stateNode)}else e.stateNode=Bm(a,n,e.memoizedProps)}else i===n?n===null&&e.stateNode!==null&&Il(e,e.memoizedProps,r.memoizedProps):(i===null?(t=r.stateNode,t===null||H||t.parentNode.removeChild(t)):i.count--,n===null?su||Km(a,e.type,e.stateNode):Bm(a,n,e.memoizedProps))}break;case 27:Au(t,e,n),Nu(e),i&512&&(H||r===null||kl(r,r.return)),r!==null&&i&4&&Il(e,e.memoizedProps,r.memoizedProps);break;case 5:if(a=cu,cu=!1,Au(t,e,n),cu=a,Nu(e),i&512&&(H||r===null||kl(r,r.return)),e.flags&32){t=e.stateNode;try{bn(t,``),k=!0}catch(t){Z(e,e.return,t)}}i&4&&e.stateNode!=null&&(t=e.memoizedProps,Il(e,t,r===null?t:r.memoizedProps)),i&1024&&(lu=!0);break;case 6:if(Au(t,e,n),Nu(e),i&4){if(e.stateNode===null)throw Error(o(162));t=e.memoizedProps,n=e.stateNode;try{n.nodeValue=t,k=!0}catch(t){Z(e,e.return,t)}}break;case 3:if(k=!1,Wm=null,a=ju,ju=bm(t.containerInfo),Au(t,e,n),ju=a,Nu(e),i&4&&r!==null&&r.memoizedState.isDehydrated)try{Hh(t.containerInfo)}catch(t){Z(e,e.return,t)}lu&&(lu=!1,Pu(e)),k=!1;break;case 4:i=cu,cu=su,r=nn(),a=ju,ju=bm(e.stateNode.containerInfo),Au(t,e,n),Nu(e),ju=a,k&&pu&&(mu=!0),k=r,cu=i;break;case 12:Au(t,e,n),Nu(e);break;case 31:Au(t,e,n),Nu(e),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 13:Au(t,e,n),Nu(e),e.child.flags&8192&&e.memoizedState!==null!=(r!==null&&r.memoizedState!==null)&&(hd=Qe()),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 22:a=e.memoizedState!==null,s=r!==null&&r.memoizedState!==null;var c=su,l=H,u=cu;su=c||a,cu=u||a,H=l||s,Au(t,e,n),H=l,cu=u,su=c,Nu(e),i&8192&&(t=e.stateNode,t._visibility=a?t._visibility&-2:t._visibility|1,!a||r===null||s||su||H||(t=s||H,n=su,r=H,su=a||su,H=t,Ru(e,2),su=n,H=r),!a&&cu||yu(e,a)),i&4&&(t=e.updateQueue,t!==null&&(n=t.retryQueue,n!==null&&(t.retryQueue=null,ku(e,n))));break;case 19:Au(t,e,n),Nu(e),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 30:i&512&&(H||r===null||kl(r,r.return)),i=nn(),a=pu,s=(n&335544064)===n,c=e.memoizedProps,pu=s&&Oi(c.default,c.update)!==`none`,Au(t,e,n),Nu(e),s&&r!==null&&k&&(e.flags|=4),pu=a,k=i;break;case 21:break;case 7:i&512&&(H||r===null||kl(r,r.return)),r&&r.stateNode!==null&&(r.stateNode._fragmentFiber=e);default:Au(t,e,n),Nu(e)}}function Nu(e){var t=e.flags;if(t&2){try{for(var n,r=e.return;r!==null;){if(Ll(r)){n=r;break}r=r.return}r=null;for(var i=e.return;i!==null;){if(Pl(i)){var a=i.stateNode;r===null?r=[a]:r.push(a)}if(Nl(i))break;i=i.return}var s=r;if(n==null)throw Error(o(160));switch(n.tag){case 27:var c=n.stateNode;Bl(e,Rl(e),c,s);break;case 5:var l=n.stateNode;n.flags&32&&(bn(l,``),n.flags&=-33),Bl(e,Rl(e),l,s);break;case 3:case 4:var u=n.stateNode.containerInfo;zl(e,Rl(e),u,s);break;default:throw Error(o(161))}}catch(t){Z(e,e.return,t)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Pu(e){if(e.subtreeFlags&1024)for(e=e.child;e!==null;){var t=e;Pu(t),t.tag===5&&t.flags&1024&&(t=t.stateNode,gh=!0,t.reset(),gh=!1),e=e.sibling}}function Fu(e,t){if(t.subtreeFlags&9270)for(t=t.child;t!==null;)Iu(t,e),t=t.sibling;else ou(t,!1)}function Iu(e,t){var n=e.alternate;if(n===null)Ql(e,!1);else switch(e.tag){case 3:if(hu=fu=!1,Kl(),Fu(t,e),!fu&&!mu){if(e=Gl,e!==null)for(var r=0;r<e.length;r+=3){n=e[r];var i=e[r+1];Ep(n,e[r+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(`+i+`)`})}e=t.containerInfo,e=e.nodeType===9?e.documentElement:e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===``&&(e.style.viewTransitionName=`none`,e.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(root)`}),e.animate({width:[0,0],height:[0,0]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition`})),hu=!0}Gl=null;break;case 5:Fu(t,e);break;case 4:r=fu,fu=!1,Fu(t,e),fu&&(mu=!0),fu=r;break;case 22:e.memoizedState===null&&(n.memoizedState===null?Fu(t,e):Ql(e,!1));break;case 30:r=fu,i=Kl(),fu=!1,Fu(t,e),fu&&(e.flags|=4);var a=e.memoizedProps,o=e.stateNode;t=Ei(a,o),o=Ei(n.memoizedProps,o);var s=Oi(a.default,a.update);s===`none`?t=!1:(a=n.memoizedState,n.memoizedState=null,n=e.child,ql=0,t=au(e,n,t,o,s,a,!0),ql!==(a===null?0:a.length)&&(e.flags|=32)),e.flags&4&&t?(Nd(e,e.memoizedProps.onUpdate),Gl=i):i!==null&&(i.push.apply(i,Gl),Gl=i),fu=e.flags&32?!0:r;break;default:Fu(t,e)}}function Lu(e,t){if(t.subtreeFlags&8772)for(t=t.child;t!==null;)vu(e,t.alternate,t),t=t.sibling}function Ru(e,t){for(e=e.child;e!==null;){var n=e,r=t;switch(n.tag){case 0:case 11:case 14:case 15:Tl(4,n,n.return),Ru(n,r);break;case 1:kl(n,n.return);var i=n.stateNode;typeof i.componentWillUnmount==`function`&&Dl(n,n.return,i),Ru(n,r);break;case 27:r&2&&gm(n.stateNode,n.type,n.memoizedProps);case 5:kl(n,n.return),n.tag!==5&&n.tag!==27||Ml(n),Ru(n,r);break;case 6:Ml(n);break;case 26:kl(n,n.return),i=n.stateNode,n.memoizedState!==null||i===null||H||i.parentNode.removeChild(i),Ru(n,r);break;case 22:n.memoizedState===null&&Ru(n,r);break;case 30:kl(n,n.return),Ru(n,r);break;case 7:kl(n,n.return);default:Ru(n,r)}e=e.sibling}}function zu(e,t,n){for(n=t.subtreeFlags&8772?n:n&-2,t=t.child;t!==null;){var r=t.alternate,i=e,a=t,o=a.flags,s=!!(n&1);switch(a.tag){case 0:case 11:case 15:zu(i,a,n),wl(4,a);break;case 1:if(zu(i,a,n),r=a,i=r.stateNode,typeof i.componentDidMount==`function`)try{i.componentDidMount()}catch(e){Z(r,r.return,e)}if(r=a,i=r.updateQueue,i!==null){var c=r.stateNode;try{var l=i.shared.hiddenCallbacks;if(l!==null)for(i.shared.hiddenCallbacks=null,i=0;i<l.length;i++)jo(l[i],c)}catch(e){Z(r,r.return,e)}}s&&o&64&&El(a),Ol(a,a.return);break;case 27:n&2&&Vl(a);case 5:a.tag!==5&&a.tag!==27||jl(a),zu(i,a,n),s&&r===null&&o&4&&Fl(a),Ol(a,a.return);break;case 6:jl(a);break;case 26:c=a.stateNode,a.memoizedState!==null||c===null||su||Km(bm(c.ownerDocument),a.type,c),zu(i,a,n),s&&r===null&&o&4&&Fl(a),Ol(a,a.return);break;case 12:zu(i,a,n);break;case 31:zu(i,a,n),s&&o&4&&Eu(i,a);break;case 13:zu(i,a,n),s&&o&4&&Du(i,a);break;case 22:a.memoizedState===null&&zu(i,a,n),Ol(a,a.return);break;case 30:zu(i,a,n),Ol(a,a.return);break;case 7:Ol(a,a.return);default:zu(i,a,n)}t=t.sibling}}function Bu(e,t){var n=null;e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),e=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(e=t.memoizedState.cachePool.pool),e!==n&&(e!=null&&e.refCount++,n!=null&&Ba(n))}function Vu(e,t){e=null,t.alternate!==null&&(e=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==e&&(t.refCount++,e!=null&&Ba(e))}function Hu(e,t,n,r){var i=(n&335544064)===n;if(t.subtreeFlags&(i?10262:10256))for(t=t.child;t!==null;)Uu(e,t,n,r),t=t.sibling;else i&&iu(t)}function Uu(e,t,n,r){var i=(n&335544064)===n;i&&t.alternate===null&&t.return!==null&&t.return.alternate!==null&&ru(t);var a=t.flags;switch(t.tag){case 0:case 11:case 15:Hu(e,t,n,r),a&2048&&wl(9,t);break;case 1:Hu(e,t,n,r);break;case 3:Hu(e,t,n,r),i&&hu&&(e=e.containerInfo,e=e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,e.style.viewTransitionName===`root`&&(e.style.viewTransitionName=``),e=e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===`none`&&(e.style.viewTransitionName=``)),a&2048&&(a=null,t.alternate!==null&&(a=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==a&&(t.refCount++,a!=null&&Ba(a)));break;case 12:if(a&2048){Hu(e,t,n,r),a=t.stateNode;try{var o=t.memoizedProps,s=o.id,c=o.onPostCommit;typeof c==`function`&&c(s,t.alternate===null?`mount`:`update`,a.passiveEffectDuration,-0)}catch(e){Z(t,t.return,e)}}else Hu(e,t,n,r);break;case 31:Hu(e,t,n,r);break;case 13:Hu(e,t,n,r);break;case 23:break;case 22:o=t.stateNode,s=t.alternate,t.memoizedState===null?(i&&s!==null&&s.memoizedState!==null&&ru(t),o._visibility&2?Hu(e,t,n,r):(o._visibility|=2,Wu(e,t,n,r,!!(t.subtreeFlags&10256)||!1))):(i&&s!==null&&s.memoizedState===null&&ru(s),o._visibility&2?Hu(e,t,n,r):Gu(e,t)),a&2048&&Bu(s,t);break;case 24:Hu(e,t,n,r),a&2048&&Vu(t.alternate,t);break;case 30:i&&(a=t.alternate,a!==null&&(Xl(a.child,!0),Xl(t.child,!0))),Hu(e,t,n,r);break;default:Hu(e,t,n,r)}}function Wu(e,t,n,r,i){for(i&&=!!(t.subtreeFlags&10256)||!1,t=t.child;t!==null;){var a=e,o=t,s=n,c=r,l=o.flags;switch(o.tag){case 0:case 11:case 15:Wu(a,o,s,c,i),wl(8,o);break;case 23:break;case 22:var u=o.stateNode;o.memoizedState===null?(u._visibility|=2,Wu(a,o,s,c,i)):u._visibility&2?Wu(a,o,s,c,i):Gu(a,o),i&&l&2048&&Bu(o.alternate,o);break;case 24:Wu(a,o,s,c,i),i&&l&2048&&Vu(o.alternate,o);break;default:Wu(a,o,s,c,i)}t=t.sibling}}function Gu(e,t){if(t.subtreeFlags&10256)for(t=t.child;t!==null;){var n=e,r=t,i=r.flags;switch(r.tag){case 22:Gu(n,r),i&2048&&Bu(r.alternate,r);break;case 24:Gu(n,r),i&2048&&Vu(r.alternate,r);break;default:Gu(n,r)}t=t.sibling}}var Ku=8192;function qu(e,t,n){if(e.subtreeFlags&Ku)for(e=e.child;e!==null;)Ju(e,t,n),e=e.sibling}function Ju(e,t,n){switch(e.tag){case 26:qu(e,t,n),e.flags&Ku&&(e.memoizedState===null?(e=e.stateNode,(t&335544128)===t&&Zm(n,e)):Qm(n,ju,e.memoizedState,e.memoizedProps));break;case 5:qu(e,t,n),e.flags&Ku&&(e=e.stateNode,(t&335544128)===t&&Zm(n,e));break;case 3:case 4:var r=ju;ju=bm(e.stateNode.containerInfo),qu(e,t,n),ju=r;break;case 22:e.memoizedState===null&&(r=e.alternate,r!==null&&r.memoizedState!==null?(r=Ku,Ku=16777216,qu(e,t,n),Ku=r):qu(e,t,n));break;case 30:if((e.flags&Ku)!==0&&(r=e.memoizedProps.name,r!=null&&r!==`auto`)){var i=e.stateNode;i.paired=null,Ul===null&&(Ul=new Map),Ul.set(r,i)}qu(e,t,n);break;default:qu(e,t,n)}}function Yu(e){var t=e.alternate;if(t!==null&&(e=t.child,e!==null)){t.child=null;do t=e.sibling,e.sibling=null,e=t;while(e!==null)}}function Xu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];du=r,$u(r,e)}Yu(e)}if(e.subtreeFlags&10256)for(e=e.child;e!==null;)Zu(e),e=e.sibling}function Zu(e){switch(e.tag){case 0:case 11:case 15:Xu(e),e.flags&2048&&Tl(9,e,e.return);break;case 3:Xu(e);break;case 12:Xu(e);break;case 22:var t=e.stateNode;e.memoizedState!==null&&t._visibility&2&&(e.return===null||e.return.tag!==13)?(t._visibility&=-3,Qu(e)):Xu(e);break;default:Xu(e)}}function Qu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];du=r,$u(r,e)}Yu(e)}for(e=e.child;e!==null;){switch(t=e,t.tag){case 0:case 11:case 15:Tl(8,t,t.return),Qu(t);break;case 22:n=t.stateNode,n._visibility&2&&(n._visibility&=-3,Qu(t));break;default:Qu(t)}e=e.sibling}}function $u(e,t){for(;du!==null;){var n=du;switch(n.tag){case 0:case 11:case 15:Tl(8,n,t);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var r=n.memoizedState.cachePool.pool;r!=null&&r.refCount++}break;case 24:Ba(n.memoizedState.cache)}if(r=n.child,r!==null)r.return=n,du=r;else a:for(n=e;du!==null;){r=du;var i=r.sibling,a=r.return;if(Su(r),r===n){du=null;break a}if(i!==null){i.return=a,du=i;break a}du=a}}}var ed={getCacheForType:function(e){var t=Ma(Ra),n=t.data.get(e);return n===void 0&&(n=e(),t.data.set(e,n)),n},cacheSignal:function(){return Ma(Ra).controller.signal}},td=typeof WeakMap==`function`?WeakMap:Map,W=0,G=null,K=null,q=0,J=0,nd=null,rd=!1,id=!1,ad=!1,od=0,Y=0,sd=0,cd=0,ld=0,ud=0,dd=0,fd=null,pd=null,md=!1,hd=0,gd=0,_d=1/0,vd=null,yd=null,X=0,bd=null,xd=null,Sd=0,Cd=0,wd=null,Td=null,Ed=null,Dd=null,Od=null,kd=0,Ad=null;function jd(){return W&2&&q!==0?q&-q:w.T===null?At():Pf()}function Md(){if(ud===0){if(!(q&536870912)||M){var e=mt;mt<<=1,!(mt&3932160)&&(mt=262144),ud=e}else ud=536870912}return e=Lo.current,e!==null&&(e.flags|=32),ud}function Nd(e,t){if(t!=null){var n=e.stateNode,r=n.ref;r===null&&(r=n.ref=Pp(Ei(e.memoizedProps,n))),Dd===null&&(Dd=[]),Dd.push(t.bind(null,r))}}function Pd(e,t,n){(e===G&&(J===2||J===9)||e.cancelPendingCommit!==null)&&(Vd(e,0),Rd(e,q,ud,!1)),Ct(e,n),(!(W&2)||e!==G)&&(e===G&&(!(W&2)&&(cd|=n),Y===4&&Rd(e,q,ud,!1)),Ef(e))}function Fd(e,t,n){if(W&6)throw Error(o(327));var r=!n&&!(t&127)&&(t&e.expiredLanes)===0||vt(e,t),i=r?Yd(e,t):qd(e,t,!0),a=r;do{if(i===0){id&&!r&&Rd(e,t,0,!1);break}if(n=e.current.alternate,a&&!Ld(n)){i=qd(e,t,!1),a=!1;continue}if(i===2){if(a=t,e.errorRecoveryDisabledLanes&a)var s=0;else s=e.pendingLanes&-536870913,s=s===0?s&536870912?536870912:0:s;if(s!==0){t=s;a:{var c=e;i=fd;var l=c.current.memoizedState.isDehydrated;if(l&&(Vd(c,s).flags|=256),s=qd(c,s,!1),s!==2&&s!==6){if(ad&&!l){c.errorRecoveryDisabledLanes|=a,cd|=a,i=4;break a}a=pd,pd=i,a!==null&&(pd===null?pd=a:pd.push.apply(pd,a))}i=s}if(a=!1,i!==2)continue}}if(i===1){Vd(e,0),Rd(e,t,0,!0);break}a:{switch(r=e,a=i,a){case 0:case 1:throw Error(o(345));case 4:if((t&4194048)!==t&&(t&62914560)!==t)break;case 6:Rd(r,t,ud,!rd);break a;case 2:pd=null;break;case 3:case 5:break;default:throw Error(o(329))}if((t&62914560)===t&&(i=hd+300-Qe(),10<i)){if(Rd(r,t,ud,!rd),_t(r,0,!0)!==0)break a;Sd=t,r.timeoutHandle=gp(Id.bind(null,r,n,pd,vd,md,t,ud,cd,dd,rd,a,`Throttled`,-0,0),i);break a}Id(r,n,pd,vd,md,t,ud,cd,dd,rd,a,null,-0,0)}break}while(1);Ef(e)}function Id(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.timeoutHandle=-1;var m=t.subtreeFlags,h=(a&335544064)===a;if(d=null,(h||m&8192||(m&16785408)==16785408)&&(d={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:On},Ul=null,Ju(t,a,d),h&&(m=d,h=e.containerInfo,h=(h.nodeType===9?h:h.ownerDocument).__reactViewTransition,h!=null&&(m.count++,m.waitingForViewTransition=!0,m=nh.bind(m),h.finished.then(m,m))),m=(a&62914560)===a?hd-Qe():(a&4194048)===a?gd-Qe():0,m=eh(d,m),m!==null)){Sd=a,e.cancelPendingCommit=m(nf.bind(null,e,t,a,n,r,i,o,s,c,l,u,d,null,f,p)),Rd(e,a,o,!l);return}nf(e,t,a,n,r,i,o,s,c,l,u,d)}function Ld(e){for(var t=e;;){var n=t.tag;if((n===0||n===11||n===15)&&t.flags&16384&&(n=t.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Xr(a(),i))return!1}catch{return!1}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Rd(e,t,n,r){t=yt(e,t),t&=~ld,t&=~cd,e.suspendedLanes|=t,e.pingedLanes&=~t,r&&(e.warmLanes|=t),r=e.expirationTimes;for(var i=t;0<i;){var a=31-D(i),o=1<<a;r[a]=-1,i&=~o}n!==0&&Tt(e,n,t)}function zd(){return W&6?!0:(Df(0,!1),!1)}function Bd(){if(K!==null){if(J===0)var e=K.return;else e=K,Ta=wa=null,ls(e),po=null,mo=0,e=K;for(;e!==null;)Cl(e.alternate,e),e=e.return;K=null}}function Vd(e,t){var n=e.timeoutHandle;return n!==-1&&(e.timeoutHandle=-1,_p(n)),n=e.cancelPendingCommit,n!==null&&(e.cancelPendingCommit=null,n()),Sd=0,Bd(),G=e,K=n=Ui(e.current,null),q=t,J=0,nd=null,rd=!1,id=vt(e,t),ad=!1,dd=ud=ld=cd=sd=Y=0,pd=fd=null,md=!1,od=yt(e,t),Ni(),n}function Hd(e,t){F=null,w.H=_c,t===no||t===io?(t=uo(),J=3):t===ro?(t=uo(),J=4):J=t===Fc?8:typeof t==`object`&&t&&typeof t.then==`function`?6:1,nd=t,K===null&&(Y=1,kc(e,Zi(t,e.current)))}function Ud(){var e=Lo.current;return e===null?!0:(q&4194048)===q?Ro===null:(q&62914560)===q||q&536870912?e===Ro:!1}function Wd(){var e=w.H;return w.H=_c,e===null?_c:e}function Gd(){var e=w.A;return w.A=ed,e}function Kd(){Y=4,rd||(q&4194048)!==q&&Lo.current!==null||(id=!0),!(sd&134217727)&&!(cd&134217727)||G===null||Rd(G,q,ud,!1)}function qd(e,t,n){var r=W;W|=2;var i=Wd(),a=Gd();(G!==e||q!==t)&&(vd=null,Vd(e,t)),t=!1;var o=Y;a:do try{if(J!==0&&K!==null){var s=K,c=nd;switch(J){case 8:Bd(),o=6;break a;case 3:case 2:case 9:case 6:Lo.current===null&&(t=!0);var l=J;if(J=0,nd=null,$d(e,s,c,l),n&&id){o=0;break a}break;default:l=J,J=0,nd=null,$d(e,s,c,l)}}Jd(),o=Y;break}catch(t){Hd(e,t)}while(1);return t&&e.shellSuspendCounter++,Ta=wa=null,W=r,w.H=i,w.A=a,K===null&&(G=null,q=0,Ni()),o}function Jd(){for(;K!==null;)Zd(K)}function Yd(e,t){var n=W;W|=2;var r=Wd(),i=Gd();G!==e||q!==t?(vd=null,_d=Qe()+500,Vd(e,t)):id=vt(e,t);a:do try{if(J!==0&&K!==null){t=K;var a=nd;b:switch(J){case 1:J=0,nd=null,$d(e,t,a,1);break;case 2:case 9:if(oo(a)){J=0,nd=null,Qd(t);break}t=function(){J!==2&&J!==9||G!==e||(J=7),Ef(e)},a.then(t,t);break a;case 3:J=7;break a;case 4:J=5;break a;case 7:oo(a)?(J=0,nd=null,Qd(t)):(J=0,nd=null,$d(e,t,a,7));break;case 5:var s=null;switch(K.tag){case 26:s=K.memoizedState;case 5:case 27:var c=K;if(s?Ym(s):c.stateNode.complete){J=0,nd=null;var l=c.sibling;if(l!==null)K=l;else{var u=c.return;u===null?K=null:(K=u,ef(u))}break b}}J=0,nd=null,$d(e,t,a,5);break;case 6:J=0,nd=null,$d(e,t,a,6);break;case 8:Bd(),Y=6;break a;default:throw Error(o(462))}}Xd();break}catch(t){Hd(e,t)}while(1);return Ta=wa=null,w.H=r,w.A=i,W=n,K===null?(G=null,q=0,Ni(),Y):0}function Xd(){for(;K!==null&&!Xe();)Zd(K)}function Zd(e){var t=hl(e.alternate,e,od);e.memoizedProps=e.pendingProps,t===null?ef(e):K=t}function Qd(e){var t=e,n=t.alternate;switch(t.tag){case 15:case 0:t=Yc(n,t,t.pendingProps,t.type,void 0,q);break;case 11:t=Yc(n,t,t.pendingProps,t.type.render,t.ref,q);break;case 5:ls(t);var r=t;r===fa&&(M?(va(r),r.tag===5&&r.stateNode!=null&&(j=r.stateNode)):(va(r),M=!0));default:Cl(n,t),t=K=Wi(t,od),t=hl(n,t,od)}e.memoizedProps=e.pendingProps,t===null?ef(e):K=t}function $d(e,t,n,r){Ta=wa=null,ls(t),po=null,mo=0;var i=t.return;try{if(Pc(e,i,t,n,q)){Y=1,kc(e,Zi(n,e.current)),K=null;return}}catch(t){if(i!==null)throw K=i,t;Y=1,kc(e,Zi(n,e.current)),K=null;return}t.flags&32768?(M||r===1?e=!0:id||q&536870912?e=!1:(rd=e=!0,(r===2||r===9||r===3||r===6)&&(r=Lo.current,r!==null&&r.tag===13&&(r.flags|=16384))),tf(t,e)):ef(t)}function ef(e){var t=e;do{if(t.flags&32768){tf(t,rd);return}e=t.return;var n=xl(t.alternate,t,od);if(n!==null){K=n;return}if(t=t.sibling,t!==null){K=t;return}K=t=e}while(t!==null);Y===0&&(Y=5)}function tf(e,t){do{var n=Sl(e.alternate,e);if(n!==null){n.flags&=32767,K=n;return}if(n=e.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!t&&(e=e.sibling,e!==null)){K=e;return}K=e=n}while(e!==null);Y=6,K=null}function nf(e,t,n,r,i,a,s,c,l,u,d,f){e.cancelPendingCommit=null;do df();while(X!==0);if(W&6)throw Error(o(327));if(t!==null){if(t===e.current)throw Error(o(177));e===G&&(K=G=null,q=0),xd=t,bd=e,Sd=n,wd=i,Td=r,rf(e,t,n,s,c,l,f)}}function rf(e,t,n,r,i,a,o){var s=t.lanes|t.childLanes;if(Cd=s,s|=Mi,wt(e,n,s,r,i,a),Dd=null,(n&335544064)===n?(Od=Ua(e),r=10262):(Od=null,r=10256),(t.subtreeFlags&r)!==0||(t.flags&r)!==0?(e.callbackNode=null,e.callbackPriority=0,yf(nt,function(){return ff(),null})):(e.callbackNode=null,e.callbackPriority=0),Hl=!1,r=!!(t.flags&13878),t.subtreeFlags&13878||r){r=w.T,w.T=null,i=T.p,T.p=2,a=W,W|=4;try{gu(e,t,n)}finally{W=a,T.p=i,w.T=r}}X=1,Hl?Ed=Mp(o,e.containerInfo,Od,sf,cf,of,lf,ff,af,null,null):(sf(),cf(),lf())}function af(e){if(X!==0){var t=bd.onRecoverableError;t(e,{componentStack:null})}}function of(){X===3&&(X=0,Iu(xd,bd),X=4)}function sf(){if(X===1){X=0;var e=bd,t=xd,n=Sd,r=!!(t.flags&13878);if(t.subtreeFlags&13878||r){r=w.T,w.T=null;var i=T.p;T.p=2;var a=W;W|=4;try{pu=mu=!1,Mu(t,e,n),n=cp;var o=ni(e.containerInfo),s=n.focusedElem,c=n.selectionRange;if(o!==s&&s&&s.ownerDocument&&ti(s.ownerDocument.documentElement,s)){if(c!==null&&ri(s)){var l=c.start,u=c.end;if(u===void 0&&(u=l),`selectionStart`in s)s.selectionStart=l,s.selectionEnd=Math.min(u,s.value.length);else{var d=s.ownerDocument||document,f=d&&d.defaultView||window;if(f.getSelection){var p=f.getSelection(),m=s.textContent.length,h=Math.min(c.start,m),g=c.end===void 0?h:Math.min(c.end,m);!p.extend&&h>g&&(o=g,g=h,h=o);var _=ei(s,h),v=ei(s,g);if(_&&v&&(p.rangeCount!==1||p.anchorNode!==_.node||p.anchorOffset!==_.offset||p.focusNode!==v.node||p.focusOffset!==v.offset)){var y=d.createRange();y.setStart(_.node,_.offset),p.removeAllRanges(),h>g?(p.addRange(y),p.extend(v.node,v.offset)):(y.setEnd(v.node,v.offset),p.addRange(y))}}}}for(d=[],p=s;p=p.parentNode;)p.nodeType===1&&d.push({element:p,left:p.scrollLeft,top:p.scrollTop});for(typeof s.focus==`function`&&s.focus(),s=0;s<d.length;s++){var b=d[s];b.element.scrollLeft=b.left,b.element.scrollTop=b.top}}gh=!!sp,cp=sp=null}finally{W=a,T.p=i,w.T=r}}e.current=t,X=2}}function cf(){if(X===2){X=0;var e=bd,t=xd,n=!!(t.flags&8772);if(t.subtreeFlags&8772||n){n=w.T,w.T=null;var r=T.p;T.p=2;var i=W;W|=4;try{vu(e,t.alternate,t)}finally{W=i,T.p=r,w.T=n}}X=3}}function lf(){if(X===4||X===3){X=0;var e=Ed;Ed=null,Ze();var t=bd,n=xd,r=Sd,i=Td,a=(r&335544064)===r?10262:10256;if((n.subtreeFlags&a)!==0||(n.flags&a)!==0?X=5:(X=0,xd=bd=null,uf(t,t.pendingLanes)),a=t.pendingLanes,a===0&&(yd=null),kt(r),n=n.stateNode,ct&&typeof ct.onCommitFiberRoot==`function`)try{ct.onCommitFiberRoot(st,n,void 0,(n.current.flags&128)==128)}catch{}if(i!==null){n=w.T,a=T.p,T.p=2,w.T=null;try{for(var o=t.onRecoverableError,s=0;s<i.length;s++){var c=i[s];o(c.value,{componentStack:c.stack})}}finally{w.T=n,T.p=a}}if(i=Dd,o=Od,Od=null,i!==null&&(Dd=null,o===null&&(o=[]),e!==null))for(c=0;c<i.length;c++)n=(0,i[c])(o),n!==void 0&&e.finished.finally(n);Sd&3&&df(),Ef(t),a=t.pendingLanes,r&261930&&a&42?t===Ad?kd++:(kd=0,Ad=t):(kd=0,Ad=null),Df(0,!1)}}function uf(e,t){(e.pooledCacheLanes&=t)===0&&(t=e.pooledCache,t!=null&&(e.pooledCache=null,Ba(t)))}function df(){return Ed!==null&&(Ed.skipTransition(),Ed=null),sf(),cf(),lf(),ff()}function ff(){if(X!==5)return!1;var e=bd,t=Cd;Cd=0;var n=kt(Sd),r=w.T,i=T.p;try{T.p=32>n?32:n,w.T=null,n=wd,wd=null;var a=bd,s=Sd;if(X=0,xd=bd=null,Sd=0,W&6)throw Error(o(331));var c=W;if(W|=4,Zu(a.current),Uu(a,a.current,s,n),W=c,Df(0,!1),ct&&typeof ct.onPostCommitFiberRoot==`function`)try{ct.onPostCommitFiberRoot(st,a)}catch{}return!0}finally{T.p=i,w.T=r,uf(e,t)}}function pf(e,t,n){t=Zi(n,t),t=jc(e.stateNode,t,2),e=To(e,t,2),e!==null&&(Ct(e,2),Ef(e))}function Z(e,t,n){if(e.tag===3)pf(e,e,n);else for(;t!==null;){if(t.tag===3){pf(t,e,n);break}if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError==`function`||typeof r.componentDidCatch==`function`&&(yd===null||!yd.has(r))){e=Zi(n,e),n=Mc(2),r=To(t,n,2),r!==null&&(Nc(n,r,t,e),Ct(r,2),Ef(r));break}}t=t.return}}function mf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new td;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(ad=!0,i.add(n),e=hf.bind(null,e,t,n),t.then(e,e))}function hf(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),e.pingedLanes|=e.suspendedLanes&n,e.warmLanes&=~n,G===e&&(q&n)===n&&(Y===4||Y===3&&(q&62914560)===q&&300>Qe()-hd?W&2?ld|=n:Vd(e,0):ld|=n,dd===q&&(dd=0)),Ef(e)}function gf(e,t){t===0&&(t=xt()),e=Ii(e,t),e!==null&&(Ct(e,t),Ef(e))}function _f(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),gf(e,n)}function vf(e,t){var n=0;switch(e.tag){case 31:case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;case 22:r=e.stateNode._retryCache;break;default:throw Error(o(314))}r!==null&&r.delete(t),gf(e,n)}function yf(e,t){return Je(e,t)}var bf=null,xf=null,Sf=!1,Cf=!1,wf=!1,Tf=0;function Ef(e){e!==xf&&e.next===null&&(xf===null?bf=xf=e:xf=xf.next=e),Cf=!0,Sf||(Sf=!0,Nf())}function Df(e,t){if(!wf&&Cf){wf=!0;do for(var n=!1,r=bf;r!==null;){if(!t){if(e!==0){var i=r.pendingLanes;if(i===0)var a=0;else{var o=r.suspendedLanes,s=r.pingedLanes;a=(1<<31-D(42|e)+1)-1,a&=i&~(o&~s),a=a&201326741?a&201326741|1:a?a|2:0}a!==0&&(n=!0,Mf(r,a))}else a=q,a=_t(r,r===G?a:0,r.cancelPendingCommit!==null||r.timeoutHandle!==-1),!(a&3)||vt(r,a)||(n=!0,Mf(r,a))}r=r.next}while(n);wf=!1}}function Of(){kf()}function kf(){Cf=Sf=!1;var e=0;Tf!==0&&hp()&&(e=Tf);for(var t=Qe(),n=null,r=bf;r!==null;){var i=r.next,a=Af(r,t);a===0?(r.next=null,n===null?bf=i:n.next=i,i===null&&(xf=n)):(n=r,(e!==0||a&3)&&(Cf=!0)),r=i}X!==0&&X!==5||Df(e,!1),Tf!==0&&(Tf=0)}function Af(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes&-62914561;0<a;){var o=31-D(a),s=1<<o,c=i[o];c===-1?((s&n)===0||(s&r)!==0)&&(i[o]=bt(s,t)):c<=t&&(e.expiredLanes|=s),a&=~s}if(t=G,n=q,n=_t(e,e===t?n:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r=e.callbackNode,n===0||e===t&&(J===2||J===9)||e.cancelPendingCommit!==null)return r!==null&&r!==null&&Ye(r),e.callbackNode=null,e.callbackPriority=0;if(!(n&3)||vt(e,n)){if(t=n&-n,t===e.callbackPriority)return t;switch(r!==null&&Ye(r),kt(n)){case 2:case 8:n=tt;break;case 32:n=nt;break;case 268435456:n=it;break;default:n=nt}return r=jf.bind(null,e),n=Je(n,r),e.callbackPriority=t,e.callbackNode=n,t}return r!==null&&r!==null&&Ye(r),e.callbackPriority=2,e.callbackNode=null,2}function jf(e,t){if(X!==0&&X!==5)return e.callbackNode=null,e.callbackPriority=0,null;var n=e.callbackNode;if(df()&&e.callbackNode!==n)return null;var r=q;return r=_t(e,e===G?r:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r===0?null:(Fd(e,r,t),Af(e,Qe()),e.callbackNode!=null&&e.callbackNode===n?jf.bind(null,e):null)}function Mf(e,t){if(df())return null;Fd(e,t,!0)}function Nf(){bp(function(){W&6?Je(et,Of):kf()})}function Pf(){if(Tf===0){var e=Ka;e===0&&(e=pt,pt<<=1,!(pt&261888)&&(pt=256)),Tf=e}return Tf}function Ff(e){return e==null||typeof e==`symbol`||typeof e==`boolean`?null:typeof e==`function`?e:Dn(e)}function If(e,t,n,r,i){if(t===`submit`&&n&&n.stateNode===i){var a=Ff((i[Pt]||null).action),o=r.submitter;o&&(t=(t=o[Pt]||null)?Ff(t.formAction):o.getAttribute(`formAction`),t!==null&&(a=t,o=null));var s=new Yn(`action`,`action`,null,r,i);e.push({event:s,listeners:[{instance:null,listener:function(){if(r.defaultPrevented){if(Tf!==0){var e=new FormData(i,o);rc(n,{pending:!0,data:e,method:i.method,action:a},null,e)}}else typeof a==`function`&&(s.preventDefault(),e=new FormData(i,o),rc(n,{pending:!0,data:e,method:i.method,action:a},a,e))},currentTarget:i}]})}}for(var Lf=0;Lf<Ci.length;Lf++){var Rf=Ci[Lf];wi(Rf.toLowerCase(),`on`+(Rf[0].toUpperCase()+Rf.slice(1)))}wi(hi,`onAnimationEnd`),wi(gi,`onAnimationIteration`),wi(_i,`onAnimationStart`),wi(`dblclick`,`onDoubleClick`),wi(`focusin`,`onFocus`),wi(`focusout`,`onBlur`),wi(vi,`onTransitionRun`),wi(yi,`onTransitionStart`),wi(bi,`onTransitionCancel`),wi(xi,`onTransitionEnd`),Zt(`onMouseEnter`,[`mouseout`,`mouseover`]),Zt(`onMouseLeave`,[`mouseout`,`mouseover`]),Zt(`onPointerEnter`,[`pointerout`,`pointerover`]),Zt(`onPointerLeave`,[`pointerout`,`pointerover`]),Xt(`onChange`,`change click focusin focusout input keydown keyup selectionchange`.split(` `)),Xt(`onSelect`,`focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange`.split(` `)),Xt(`onBeforeInput`,[`compositionend`,`keypress`,`textInput`,`paste`]),Xt(`onCompositionEnd`,`compositionend focusout keydown keypress keyup mousedown`.split(` `)),Xt(`onCompositionStart`,`compositionstart focusout keydown keypress keyup mousedown`.split(` `)),Xt(`onCompositionUpdate`,`compositionupdate focusout keydown keypress keyup mousedown`.split(` `));var zf=`abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting`.split(` `),Bf=new Set(`beforetoggle cancel close invalid load scroll scrollend toggle`.split(` `).concat(zf));function Vf(e,t){t=!!(t&4);for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;a:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],c=s.instance,l=s.currentTarget;if(s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){ki(e)}i.currentTarget=null,a=c}else for(o=0;o<r.length;o++){if(s=r[o],c=s.instance,l=s.currentTarget,s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){ki(e)}i.currentTarget=null,a=c}}}}function Q(e,t){var n=t[It];n===void 0&&(n=t[It]=new Set);var r=e+`__bubble`;n.has(r)||(Gf(t,e,2,!1),n.add(r))}function Hf(e,t,n){var r=0;t&&(r|=4),Gf(n,e,r,t)}var Uf=`_reactListening`+Math.random().toString(36).slice(2);function Wf(e){if(!e[Uf]){e[Uf]=!0,Jt.forEach(function(t){t!==`selectionchange`&&(Bf.has(t)||Hf(t,!1,e),Hf(t,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Uf]||(t[Uf]=!0,Hf(`selectionchange`,!1,t))}}function Gf(e,t,n,r){switch(Ch(t)){case 2:var i=_h;break;case 8:i=vh;break;default:i=yh}n=i.bind(null,t,n,e),i=void 0,!Rn||t!==`touchstart`&&t!==`touchmove`&&t!==`wheel`||(i=!0),r?i===void 0?e.addEventListener(t,n,!0):e.addEventListener(t,n,{capture:!0,passive:i}):i===void 0?e.addEventListener(t,n,!1):e.addEventListener(t,n,{passive:i})}function Kf(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)a:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&o.stateNode.containerInfo===i)return;o=o.return}for(;s!==null;){if(o=Ut(s),o===null)return;if(l=o.tag,l===5||l===6||l===26||l===27){r=a=o;continue a}s=s.parentNode}}r=r.return}Fn(function(){var r=a,i=An(n),o=[];a:{var s=Si.get(e);if(s!==void 0){var l=Yn,u=e;switch(e){case`keypress`:if(Wn(n)===0)break a;case`keydown`:case`keyup`:l=pr;break;case`focusin`:u=`focus`,l=ir;break;case`focusout`:u=`blur`,l=ir;break;case`beforeblur`:case`afterblur`:l=ir;break;case`click`:if(n.button===2)break a;case`auxclick`:case`dblclick`:case`mousedown`:case`mousemove`:case`mouseup`:case`mouseout`:case`mouseover`:case`contextmenu`:l=nr;break;case`drag`:case`dragend`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`dragstart`:case`drop`:l=rr;break;case`touchcancel`:case`touchend`:case`touchmove`:case`touchstart`:l=gr;break;case hi:case gi:case _i:l=ar;break;case xi:l=_r;break;case`scroll`:case`scrollend`:l=Zn;break;case`wheel`:l=vr;break;case`copy`:case`cut`:case`paste`:l=or;break;case`gotpointercapture`:case`lostpointercapture`:case`pointercancel`:case`pointerdown`:case`pointermove`:case`pointerout`:case`pointerover`:case`pointerup`:l=mr;break;case`submit`:l=hr;break;case`toggle`:case`beforetoggle`:l=yr}var d=!!(t&4),f=!d&&(e===`scroll`||e===`scrollend`),p=d?s===null?null:s+`Capture`:s;d=[];for(var m=r,h;m!==null;){var g=m;if(h=g.stateNode,g=g.tag,g!==5&&g!==26&&g!==27||h===null||p===null||(g=In(m,p),g!=null&&d.push(qf(m,g,h))),f)break;m=m.return}0<d.length&&(s=new l(s,u,null,n,i),o.push({event:s,listeners:d}))}}if(!(t&7)){a:{if(l=e===`mouseover`||e===`pointerover`,s=e===`mouseout`||e===`pointerout`,l&&n!==kn&&(u=n.relatedTarget||n.fromElement)&&(Ut(u)||u[Ft]))break a;(s||l)&&(u=i.window===i?i:(l=i.ownerDocument)?l.defaultView||l.parentWindow:window,s?(l=n.relatedTarget||n.toElement,s=r,l=l?Ut(l):null,l!==null&&(f=c(l),d=l.tag,l!==f||d!==5&&d!==27&&d!==6)&&(l=null)):(s=null,l=r),s!==l&&(d=nr,g=`onMouseLeave`,p=`onMouseEnter`,m=`mouse`,(e===`pointerout`||e===`pointerover`)&&(d=mr,g=`onPointerLeave`,p=`onPointerEnter`,m=`pointer`),f=s==null?u:Gt(s),h=l==null?u:Gt(l),u=new d(g,m+`leave`,s,n,i),u.target=f,u.relatedTarget=h,g=null,Ut(i)===r&&(d=new d(p,m+`enter`,l,n,i),d.target=h,d.relatedTarget=f,g=d),f=g,d=s&&l?ie(s,l,Yf):null,s!==null&&Xf(o,u,s,d,!1),l!==null&&f!==null&&Xf(o,f,l,d,!0)))}a:{if(s=r?Gt(r):window,l=s.nodeName&&s.nodeName.toLowerCase(),l===`select`||l===`input`&&s.type===`file`)var _=Rr;else if(Nr(s)){if(zr)_=Jr;else{_=Kr;var v=Gr}}else l=s.nodeName,!l||l.toLowerCase()!==`input`||s.type!==`checkbox`&&s.type!==`radio`?r&&wn(r.elementType)&&(_=Rr):_=qr;if(_&&=_(e,r)){Pr(o,_,n,i);break a}v&&v(e,s,r)}switch(v=r?Gt(r):window,e){case`focusin`:(Nr(v)||v.contentEditable===`true`)&&(ai=v,oi=r,si=null);break;case`focusout`:si=oi=ai=null;break;case`mousedown`:ci=!0;break;case`contextmenu`:case`mouseup`:case`dragend`:ci=!1,li(o,n,i);break;case`selectionchange`:if(ii)break;case`keydown`:case`keyup`:li(o,n,i)}var y;if(xr)b:{switch(e){case`compositionstart`:var b=`onCompositionStart`;break b;case`compositionend`:b=`onCompositionEnd`;break b;case`compositionupdate`:b=`onCompositionUpdate`;break b}b=void 0}else kr?Dr(e,n)&&(b=`onCompositionEnd`):e===`keydown`&&n.keyCode===229&&(b=`onCompositionStart`);b&&(wr&&n.locale!==`ko`&&(kr||b!==`onCompositionStart`?b===`onCompositionEnd`&&kr&&(y=Un()):(Bn=i,Vn=`value`in Bn?Bn.value:Bn.textContent,kr=!0)),v=Jf(r,b),0<v.length&&(b=new sr(b,e,null,n,i),o.push({event:b,listeners:v}),y?b.data=y:(y=Or(n),y!==null&&(b.data=y)))),(y=Cr?Ar(e,n):jr(e,n))&&(b=Jf(r,`onBeforeInput`),0<b.length&&(v=new sr(`onBeforeInput`,`beforeinput`,null,n,i),o.push({event:v,listeners:b}),v.data=y)),If(o,e,r,n,i)}Vf(o,t)})}function qf(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Jf(e,t){for(var n=t+`Capture`,r=[];e!==null;){var i=e,a=i.stateNode;if(i=i.tag,i!==5&&i!==26&&i!==27||a===null||(i=In(e,n),i!=null&&r.unshift(qf(e,i,a)),i=In(e,t),i!=null&&r.push(qf(e,i,a))),e.tag===3)return r;e=e.return}return[]}function Yf(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5&&e.tag!==27);return e||null}function Xf(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,c=s.alternate,l=s.stateNode;if(s=s.tag,c!==null&&c===r)break;s!==5&&s!==26&&s!==27||l===null||(c=l,i?(l=In(n,a),l!=null&&o.unshift(qf(n,l,c))):i||(l=In(n,a),l!=null&&o.push(qf(n,l,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Zf=/\r\n?/g,Qf=/\u0000|\uFFFD/g;function $f(e){return(typeof e==`string`?e:``+e).replace(Zf,`
`).replace(Qf,``)}function ep(e,t){return t=$f(t),$f(e)===t}function $(e,t,n,r,i,a){switch(n){case`children`:if(typeof r==`string`)t===`body`||t===`textarea`&&r===``||bn(e,r);else if(typeof r==`number`||typeof r==`bigint`)t!==`body`&&bn(e,``+r);else return;break;case`className`:an(e,`class`,r);break;case`tabIndex`:an(e,`tabindex`,r);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:an(e,n,r);break;case`style`:Cn(e,r,a);return;case`data`:if(t!==`object`){an(e,`data`,r);break}case`src`:case`href`:if(r===``&&(t!==`a`||n!==`href`)){e.removeAttribute(n);break}if(r==null||typeof r==`function`||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=Dn(r),e.setAttribute(n,r);break;case`action`:case`formAction`:if(typeof r==`function`){e.setAttribute(n,`javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')`);break}if(typeof a==`function`&&(n===`formAction`?(t!==`input`&&$(e,t,`name`,i.name,i,null),$(e,t,`formEncType`,i.formEncType,i,null),$(e,t,`formMethod`,i.formMethod,i,null),$(e,t,`formTarget`,i.formTarget,i,null)):($(e,t,`encType`,i.encType,i,null),$(e,t,`method`,i.method,i,null),$(e,t,`target`,i.target,i,null))),r==null||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=Dn(r),e.setAttribute(n,r);break;case`onClick`:r!=null&&(e.onclick=On);return;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(o(61));if(n=r.__html,n!=null){if(i.children!=null)throw Error(o(60));a?.__html!==n&&(e.innerHTML=n)}}break;case`multiple`:e.multiple=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`muted`:e.muted=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`ref`:break;case`autoFocus`:break;case`xlinkHref`:if(r==null||typeof r==`function`||typeof r==`boolean`||typeof r==`symbol`){e.removeAttribute(`xlink:href`);break}n=Dn(r),e.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,n);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`credentialless`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:r&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,``):e.removeAttribute(n);break;case`capture`:case`download`:!0===r?e.setAttribute(n,``):!1!==r&&r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`cols`:case`rows`:case`size`:case`span`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`&&!isNaN(r)&&1<=r?e.setAttribute(n,r):e.removeAttribute(n);break;case`rowSpan`:case`start`:r==null||typeof r==`function`||typeof r==`symbol`||isNaN(r)?e.removeAttribute(n):e.setAttribute(n,r);break;case`popover`:Q(`beforetoggle`,e),Q(`toggle`,e),rn(e,`popover`,r);break;case`xlinkActuate`:on(e,`http://www.w3.org/1999/xlink`,`xlink:actuate`,r);break;case`xlinkArcrole`:on(e,`http://www.w3.org/1999/xlink`,`xlink:arcrole`,r);break;case`xlinkRole`:on(e,`http://www.w3.org/1999/xlink`,`xlink:role`,r);break;case`xlinkShow`:on(e,`http://www.w3.org/1999/xlink`,`xlink:show`,r);break;case`xlinkTitle`:on(e,`http://www.w3.org/1999/xlink`,`xlink:title`,r);break;case`xlinkType`:on(e,`http://www.w3.org/1999/xlink`,`xlink:type`,r);break;case`xmlBase`:on(e,`http://www.w3.org/XML/1998/namespace`,`xml:base`,r);break;case`xmlLang`:on(e,`http://www.w3.org/XML/1998/namespace`,`xml:lang`,r);break;case`xmlSpace`:on(e,`http://www.w3.org/XML/1998/namespace`,`xml:space`,r);break;case`is`:rn(e,`is`,r);break;case`innerText`:case`textContent`:return;default:if(!(2<n.length)||n[0]!==`o`&&n[0]!==`O`||n[1]!==`n`&&n[1]!==`N`)n=Tn.get(n)||n,rn(e,n,r);else return}k=!0}function tp(e,t,n,r,i,a){switch(n){case`style`:Cn(e,r,a);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(o(61));if(n=r.__html,n!=null){if(i.children!=null)throw Error(o(60));a?.__html!==n&&(e.innerHTML=n)}}break;case`children`:if(typeof r==`string`)bn(e,r);else if(typeof r==`number`||typeof r==`bigint`)bn(e,``+r);else return;break;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`onClick`:r!=null&&(e.onclick=On);return;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`innerHTML`:case`ref`:return;case`innerText`:case`textContent`:return;default:if(!Yt.hasOwnProperty(n))a:{if(n[0]===`o`&&n[1]===`n`&&(i=n.endsWith(`Capture`),a=n.slice(2,i?n.length-7:void 0),t=e[Pt]||null,t=t==null?null:t[n],typeof t==`function`&&e.removeEventListener(a,t,i),typeof r==`function`)){typeof t!=`function`&&t!==null&&(n in e?e[n]=null:e.hasAttribute(n)&&e.removeAttribute(n)),e.addEventListener(a,r,i);break a}k=!0,n in e?e[n]=r:!0===r?e.setAttribute(n,``):rn(e,n,r)}return}k=!0}function np(e,t,n){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`img`:Q(`error`,e),Q(`load`,e);var r=!1,i=!1,a;for(a in n)if(n.hasOwnProperty(a)){var s=n[a];if(s!=null)switch(a){case`src`:r=!0;break;case`srcSet`:i=!0;break;case`children`:case`dangerouslySetInnerHTML`:throw Error(o(137,t));default:$(e,t,a,s,n,null)}}i&&$(e,t,`srcSet`,n.srcSet,n,null),r&&$(e,t,`src`,n.src,n,null);return;case`input`:Q(`invalid`,e);var c=a=s=i=null,l=null,u=null;for(r in n)if(n.hasOwnProperty(r)){var d=n[r];if(d!=null)switch(r){case`name`:i=d;break;case`type`:s=d;break;case`checked`:l=d;break;case`defaultChecked`:u=d;break;case`value`:a=d;break;case`defaultValue`:c=d;break;case`children`:case`dangerouslySetInnerHTML`:if(d!=null)throw Error(o(137,t));break;default:$(e,t,r,d,n,null)}}hn(e,a,c,l,u,s,i,!1);return;case`select`:for(i in Q(`invalid`,e),r=s=a=null,n)if(n.hasOwnProperty(i)&&(c=n[i],c!=null))switch(i){case`value`:a=c;break;case`defaultValue`:s=c;break;case`multiple`:r=c;default:$(e,t,i,c,n,null)}t=a,n=s,e.multiple=!!r,t==null?n!=null&&_n(e,!!r,n,!0):_n(e,!!r,t,!1);return;case`textarea`:for(s in Q(`invalid`,e),a=i=r=null,n)if(n.hasOwnProperty(s)&&(c=n[s],c!=null))switch(s){case`value`:r=c;break;case`defaultValue`:i=c;break;case`children`:a=c;break;case`dangerouslySetInnerHTML`:if(c!=null)throw Error(o(91));break;default:$(e,t,s,c,n,null)}yn(e,r,i,a);return;case`option`:for(l in n)if(n.hasOwnProperty(l)&&(r=n[l],r!=null))switch(l){case`selected`:e.selected=r&&typeof r!=`function`&&typeof r!=`symbol`;break;default:$(e,t,l,r,n,null)}return;case`dialog`:Q(`beforetoggle`,e),Q(`toggle`,e),Q(`cancel`,e),Q(`close`,e);break;case`iframe`:case`object`:Q(`load`,e);break;case`video`:case`audio`:for(r=0;r<zf.length;r++)Q(zf[r],e);break;case`image`:Q(`error`,e),Q(`load`,e);break;case`details`:Q(`toggle`,e);break;case`embed`:case`source`:case`link`:Q(`error`,e),Q(`load`,e);case`area`:case`base`:case`br`:case`col`:case`hr`:case`keygen`:case`meta`:case`param`:case`track`:case`wbr`:case`menuitem`:for(u in n)if(n.hasOwnProperty(u)&&(r=n[u],r!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:throw Error(o(137,t));default:$(e,t,u,r,n,null)}return;default:if(wn(t)){for(d in n)n.hasOwnProperty(d)&&(r=n[d],r!==void 0&&tp(e,t,d,r,n,void 0));return}}for(c in n)n.hasOwnProperty(c)&&(r=n[c],r!=null&&$(e,t,c,r,n,null))}var rp={};function ip(e,t,n,r){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`input`:var i=null,a=null,s=null,c=null,l=null,u=null,d=null;for(m in n){var f=n[m];if(n.hasOwnProperty(m)&&f!=null)switch(m){case`checked`:break;case`value`:break;case`defaultValue`:l=f;default:r.hasOwnProperty(m)||$(e,t,m,null,r,f)}}for(var p in r){var m=r[p];if(f=n[p],r.hasOwnProperty(p)&&(m!=null||f!=null))switch(p){case`type`:m!==f&&(k=!0),a=m;break;case`name`:m!==f&&(k=!0),i=m;break;case`checked`:m!==f&&(k=!0),u=m;break;case`defaultChecked`:m!==f&&(k=!0),d=m;break;case`value`:m!==f&&(k=!0),s=m;break;case`defaultValue`:m!==f&&(k=!0),c=m;break;case`children`:case`dangerouslySetInnerHTML`:if(m!=null)throw Error(o(137,t));break;default:m!==f&&$(e,t,p,m,r,f)}}mn(e,s,c,l,u,d,a,i);return;case`select`:for(a in m=s=c=p=null,n)if(l=n[a],n.hasOwnProperty(a)&&l!=null)switch(a){case`value`:break;case`multiple`:m=l;default:r.hasOwnProperty(a)||$(e,t,a,null,r,l)}for(i in r)if(a=r[i],l=n[i],r.hasOwnProperty(i)&&(a!=null||l!=null))switch(i){case`value`:a!==l&&(k=!0),p=a;break;case`defaultValue`:a!==l&&(k=!0),c=a;break;case`multiple`:a!==l&&(k=!0),s=a;default:a!==l&&$(e,t,i,a,r,l)}t=c,n=s,r=m,p==null?!!r!=!!n&&(t==null?_n(e,!!n,n?[]:``,!1):_n(e,!!n,t,!0)):_n(e,!!n,p,!1);return;case`textarea`:for(c in m=p=null,n)if(i=n[c],n.hasOwnProperty(c)&&i!=null&&!r.hasOwnProperty(c))switch(c){case`value`:break;case`children`:break;default:$(e,t,c,null,r,i)}for(s in r)if(i=r[s],a=n[s],r.hasOwnProperty(s)&&(i!=null||a!=null))switch(s){case`value`:i!==a&&(k=!0),p=i;break;case`defaultValue`:i!==a&&(k=!0),m=i;break;case`children`:break;case`dangerouslySetInnerHTML`:if(i!=null)throw Error(o(91));break;default:i!==a&&$(e,t,s,i,r,a)}vn(e,p,m);return;case`option`:for(var h in n)if(p=n[h],n.hasOwnProperty(h)&&p!=null&&!r.hasOwnProperty(h))switch(h){case`selected`:e.selected=!1;break;default:$(e,t,h,null,r,p)}for(l in r)if(p=r[l],m=n[l],r.hasOwnProperty(l)&&p!==m&&(p!=null||m!=null))switch(l){case`selected`:p!==m&&(k=!0),e.selected=p&&typeof p!=`function`&&typeof p!=`symbol`;break;default:$(e,t,l,p,r,m)}return;case`img`:case`link`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:case`menuitem`:for(var g in n)p=n[g],n.hasOwnProperty(g)&&p!=null&&!r.hasOwnProperty(g)&&$(e,t,g,null,r,p);for(u in r)if(p=r[u],m=n[u],r.hasOwnProperty(u)&&p!==m&&(p!=null||m!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:if(p!=null)throw Error(o(137,t));break;default:$(e,t,u,p,r,m)}return;default:if(wn(t)){for(var _ in n)p=n[_],n.hasOwnProperty(_)&&p!==void 0&&!r.hasOwnProperty(_)&&tp(e,t,_,void 0,r,p);for(d in r)p=r[d],m=n[d],!r.hasOwnProperty(d)||p===m||p===void 0&&m===void 0||tp(e,t,d,p,r,m);return}}for(var v in n)p=n[v],n.hasOwnProperty(v)&&p!=null&&!r.hasOwnProperty(v)&&$(e,t,v,null,r,p);for(f in r)p=r[f],m=n[f],!r.hasOwnProperty(f)||p===m||p==null&&m==null||$(e,t,f,p,r,m)}function ap(e){switch(e){case`css`:case`script`:case`font`:case`img`:case`image`:case`input`:case`link`:return!0;default:return!1}}function op(){if(typeof performance.getEntriesByType==`function`){for(var e=0,t=0,n=performance.getEntriesByType(`resource`),r=0;r<n.length;r++){var i=n[r],a=i.transferSize,o=i.initiatorType,s=i.duration;if(a&&s&&ap(o)){for(o=0,s=i.responseEnd,r+=1;r<n.length;r++){var c=n[r],l=c.startTime;if(l>s)break;var u=c.transferSize,d=c.initiatorType;u&&ap(d)&&(c=c.responseEnd,o+=u*(c<s?1:(s-l)/(c-l)))}if(--r,t+=8*(a+o)/(i.duration/1e3),e++,10<e)break}}if(0<e)return t/e/1e6}return navigator.connection&&(e=navigator.connection.downlink,typeof e==`number`)?e:5}var sp=null,cp=null;function lp(e){return e.nodeType===9?e:e.ownerDocument}function up(e){switch(e){case`http://www.w3.org/2000/svg`:return 1;case`http://www.w3.org/1998/Math/MathML`:return 2;default:return 0}}function dp(e,t){if(e===0)switch(t){case`svg`:return 1;case`math`:return 2;default:return 0}return e===1&&t===`foreignObject`?0:e}function fp(e,t,n,r){return n=lp(n).createElement(e),n[Nt]=r,n[Pt]=t,np(n,e,t),O(n),n}function pp(e,t){return e===`textarea`||e===`noscript`||typeof t.children==`string`||typeof t.children==`number`||typeof t.children==`bigint`||typeof t.dangerouslySetInnerHTML==`object`&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var mp=null;function hp(){var e=window.event;return e&&e.type===`popstate`?e!==mp&&(mp=e,!0):(mp=null,!1)}var gp=typeof setTimeout==`function`?setTimeout:void 0,_p=typeof clearTimeout==`function`?clearTimeout:void 0,vp=typeof Promise==`function`?Promise:void 0,yp=typeof requestAnimationFrame==`function`?requestAnimationFrame:gp,bp=typeof queueMicrotask==`function`?queueMicrotask:vp===void 0?gp:function(e){return vp.resolve(null).then(e).catch(xp)};function xp(e){setTimeout(function(){throw e})}function Sp(e){return e===`head`}function Cp(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8){if(n=i.data,n===`/$`||n===`/&`){if(r===0){e.removeChild(i),Hh(t);return}r--}else if(n===`$`||n===`$?`||n===`$~`||n===`$!`||n===`&`)r++;else if(n===`html`)_m(e.ownerDocument.documentElement);else if(n===`head`){n=e.ownerDocument.head,_m(n);for(var a=n.firstChild;a;){var o=a.nextSibling,s=a.nodeName;a[Bt]||s===`SCRIPT`||s===`STYLE`||s===`LINK`&&a.rel.toLowerCase()===`stylesheet`||n.removeChild(a),a=o}}else n===`body`&&_m(e.ownerDocument.body)}n=i}while(n);Hh(t)}function wp(e,t){var n=e;e=0;do{var r=n.nextSibling;if(n.nodeType===1?t?(n._stashedDisplay=n.style.display,n.style.display=`none`):(n.style.display=n._stashedDisplay||``,n.getAttribute(`style`)===``&&n.removeAttribute(`style`)):n.nodeType===3&&(t?(n._stashedText=n.nodeValue,n.nodeValue=``):n.nodeValue=n._stashedText||``),r&&r.nodeType===8){if(n=r.data,n===`/$`){if(e===0)break;e--}else n!==`$`&&n!==`$?`&&n!==`$~`&&n!==`$!`||e++}n=r}while(n)}function Tp(e,t,n){if(t=CSS.escape(t)===t?t:`r-`+btoa(t).replace(/=/g,``),e.style.viewTransitionName=t,n!=null&&(e.style.viewTransitionClass=n),n=getComputedStyle(e),n.display===`inline`){if(t=e.getClientRects(),t.length===1)var r=1;else for(var i=r=0;i<t.length;i++){var a=t[i];0<a.width&&0<a.height&&r++}r===1&&(e=e.style,e.display=t.length===1?`inline-block`:`block`,e.marginTop=`-`+n.paddingTop,e.marginBottom=`-`+n.paddingBottom)}}function Ep(e,t){e=e.style,t=t.style;var n=t==null?null:t.hasOwnProperty(`viewTransitionName`)?t.viewTransitionName:t.hasOwnProperty(`view-transition-name`)?t[`view-transition-name`]:null;e.viewTransitionName=n==null||typeof n==`boolean`?``:(``+n).trim(),n=t==null?null:t.hasOwnProperty(`viewTransitionClass`)?t.viewTransitionClass:t.hasOwnProperty(`view-transition-class`)?t[`view-transition-class`]:null,e.viewTransitionClass=n==null||typeof n==`boolean`?``:(``+n).trim(),e.display===`inline-block`&&(t==null?e.display=e.margin=``:(n=t.display,e.display=n==null||typeof n==`boolean`?``:n,n=t.margin,n==null?(n=t.hasOwnProperty(`marginTop`)?t.marginTop:t[`margin-top`],e.marginTop=n==null||typeof n==`boolean`?``:n,t=t.hasOwnProperty(`marginBottom`)?t.marginBottom:t[`margin-bottom`],e.marginBottom=t==null||typeof t==`boolean`?``:t):e.margin=n))}function Dp(e,t,n){return n=n.ownerDocument.defaultView,{rect:e,abs:t.position===`absolute`||t.position===`fixed`,clip:t.clipPath!==`none`||t.overflow!==`visible`||t.filter!==`none`||t.mask!==`none`||t.mask!==`none`||t.borderRadius!==`0px`,view:0<=e.bottom&&0<=e.right&&e.top<=n.innerHeight&&e.left<=n.innerWidth}}function Op(e){return Dp(e.getBoundingClientRect(),getComputedStyle(e),e)}function kp(e){var t=e.getBoundingClientRect();t=new DOMRect(t.x+2e4,t.y+2e4,t.width,t.height);var n=getComputedStyle(e);return Dp(t,n,e)}function Ap(e){return e.documentElement.clientHeight}function jp(e){this.addEventListener(`load`,e),this.addEventListener(`error`,e)}function Mp(e,t,n,r,i,a,o,s,c){var l=t.nodeType===9?t:t.ownerDocument;try{var u=l.startViewTransition({update:function(){var t=l.defaultView,n=t.navigation&&t.navigation.transition,o=l.fonts.status;r();var s=[];if(o===`loaded`&&(Ap(l),l.fonts.status===`loading`&&s.push(l.fonts.ready)),o=s.length,e!==null)for(var c=e.suspenseyImages,u=0,d=0;d<c.length;d++){var f=c[d];if(!f.complete){var p=f.getBoundingClientRect();if(0<p.bottom&&0<p.right&&p.top<t.innerHeight&&p.left<t.innerWidth){if(u+=Xm(f),u>$m){s.length=o;break}f=new Promise(jp.bind(f)),s.push(f)}}}if(0<s.length)return t=Promise.race([Promise.all(s),new Promise(function(e){return setTimeout(e,500)})]).then(i,i),(n?Promise.allSettled([n.finished,t]):t).then(a,a);if(i(),n)return n.finished.then(a,a);a()},types:n});l.__reactViewTransition=u;var d=[];return u.ready.then(function(){for(var e=l.documentElement.getAnimations({subtree:!0}),t=0;t<e.length;t++){var n=e[t],r=n.effect,i=r.pseudoElement;if(i!=null&&i.startsWith(`::view-transition`)){d.push(n),n=r.getKeyframes();for(var a=i=void 0,s=!0,c=0;c<n.length;c++){var u=n[c],f=u.width;if(i===void 0)i=f;else if(i!==f){s=!1;break}if(f=u.height,a===void 0)a=f;else if(a!==f){s=!1;break}delete u.width,delete u.height,u.transform===`none`&&delete u.transform}s&&i!==void 0&&a!==void 0&&(r.setKeyframes(n),s=getComputedStyle(r.target,r.pseudoElement),s.width!==i||s.height!==a)&&(s=n[0],s.width=i,s.height=a,s=n[n.length-1],s.width=i,s.height=a,r.setKeyframes(n))}}o()},function(e){l.__reactViewTransition===u&&(l.__reactViewTransition=null);try{if(typeof e==`object`&&e)switch(e.name){case`InvalidStateError`:(e.message===`View transition was skipped because document visibility state is hidden.`||e.message===`Skipping view transition because document visibility state has become hidden.`||e.message===`Skipping view transition because viewport size changed.`||e.message===`Transition was aborted because of invalid state`)&&(e=null)}e!==null&&c(e)}finally{r(),i(),o()}}),u.finished.finally(function(){for(var e=0;e<d.length;e++)d[e].cancel();l.__reactViewTransition===u&&(l.__reactViewTransition=null),s()}),u}catch{return r(),i(),o(),null}}function Np(e,t){this._scope=document.documentElement,this._selector=`::view-transition-`+e+`(`+t+`)`}Np.prototype.animate=function(e,t){return t=typeof t==`number`?{duration:t}:x({},t),t.pseudoElement=this._selector,this._scope.animate(e,t)},Np.prototype.getAnimations=function(){for(var e=this._scope,t=this._selector,n=e.getAnimations({subtree:!0}),r=[],i=0;i<n.length;i++){var a=n[i].effect;a!==null&&a.target===e&&a.pseudoElement===t&&r.push(n[i])}return r},Np.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function Pp(e){return{name:e,group:new Np(`group`,e),imagePair:new Np(`image-pair`,e),old:new Np(`old`,e),new:new Np(`new`,e)}}function Fp(e){this._fragmentFiber=e,this._observers=this._eventListeners=null}Fp.prototype.addEventListener=function(e,t,n){var r=null,i=null;if(!(n!=null&&typeof n!=`boolean`&&(r=n.signal||null,r!==null&&r.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var a=this._eventListeners;if(Bp(a,e,t,n)===-1){var o=this,s=t;n!=null&&typeof n!=`boolean`&&!0===n.once&&(s=function(r){o.removeEventListener(e,t,n),typeof t==`function`?t.call(this,r):t.handleEvent(r)}),r!==null&&(i=o.removeEventListener.bind(o,e,t,n),r.addEventListener(`abort`,i,{once:!0}),i=r.removeEventListener.bind(r,`abort`,i)),r=Rp(n),a.push({type:e,listener:t,optionsOrUseCapture:n,attachedListener:s,cleanup:i}),m(this._fragmentFiber.child,!1,Ip,e,s,r)}this._eventListeners=a}};function Ip(e,t,n,r){return y(e).addEventListener(t,n,r),!1}Fp.prototype.removeEventListener=function(e,t,n){var r=this._eventListeners;if(r!==null&&(t=Bp(r,e,t,n),t!==-1)){var i=r[t];n=i.attachedListener;var a=i.cleanup;i=Rp(i.optionsOrUseCapture),m(this._fragmentFiber.child,!1,Lp,e,n,i),r.splice(t,1),a!==null&&a()}};function Lp(e,t,n,r){return y(e).removeEventListener(t,n,r),!1}function Rp(e){return e!=null&&typeof e!=`boolean`&&(!0===e.once||e.signal instanceof AbortSignal)?{capture:e.capture,passive:e.passive}:e}function zp(e){return e==null?`c=0`:typeof e==`boolean`?`c=`+(e?`1`:`0`):`c=`+(e.capture?`1`:`0`)}function Bp(e,t,n,r){if(e.length===0)return-1;r=zp(r);for(var i=0;i<e.length;i++){var a=e[i];if(a.type===t&&a.listener===n&&zp(a.optionsOrUseCapture)===r)return i}return-1}Fp.prototype.dispatchEvent=function(e){var t=h(this._fragmentFiber);if(t===null)return!0;t=y(t);var n=this._eventListeners;if(n!==null&&0<n.length||!e.bubbles){var r=t.nodeType===9?t.createComment(``):document.createTextNode(``);if(n)for(var i=0;i<n.length;i++){var a=n[i];r.addEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture))}if(t.appendChild(r),e=r.dispatchEvent(e),n)for(i=0;i<n.length;i++)a=n[i],r.removeEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture));return t.removeChild(r),e}return t.dispatchEvent(e)},Fp.prototype.focus=function(e){m(this._fragmentFiber.child,!0,Vp,e,void 0,void 0)};function Vp(e,t){return e.tag!==6&&(e=y(e),pm(e,t))}Fp.prototype.focusLast=function(e){var t=[];m(this._fragmentFiber.child,!0,Hp,t,void 0,void 0);for(var n=t.length-1;0<=n&&!Vp(t[n],e);n--);};function Hp(e,t){return t.push(e),!1}Fp.prototype.blur=function(){var e=h(this._fragmentFiber);e!==null&&(e=y(e),e=lp(e).activeElement,e!==null&&m(this._fragmentFiber.child,!1,Up,e,void 0,void 0))};function Up(e,t){return e.tag!==6&&(e=y(e),e===t||e.contains(t)?(t.blur(),!0):!1)}Fp.prototype.observeUsing=function(e){this._observers===null&&(this._observers=new Set),this._observers.add(e),m(this._fragmentFiber.child,!1,Wp,e,void 0,void 0)};function Wp(e,t){return e.tag!==6&&(e=y(e),t.observe(e),!1)}Fp.prototype.unobserveUsing=function(e){var t=this._observers;if(t!==null&&t.has(e)){t.delete(e),m(this._fragmentFiber.child,!1,Gp,e,void 0,void 0);for(var n=t=0;n<Kp.length;n++){var r=Kp[n];r.fragmentInstance===this&&r.observer===e?e.unobserve(r.instance):Kp[t++]=r}Kp.length=t}};function Gp(e,t){return e.tag!==6&&(e=y(e),t.unobserve(e),!1)}var Kp=[],qp=!1;function Jp(e,t,n){Kp.push({fragmentInstance:e,observer:t,instance:n}),qp||(qp=!0,mm(function(){qp=!1;var e=Kp;Kp=[];for(var t=0;t<e.length;t++){var n=e[t];n.observer.unobserve(n.instance)}}))}Fp.prototype.getClientRects=function(){var e=[];return m(this._fragmentFiber.child,!1,Yp,e,void 0,void 0),e};function Yp(e,t){if(e.tag===6){e=e.stateNode;var n=e.ownerDocument.createRange();n.selectNodeContents(e),t.push.apply(t,n.getClientRects())}else e=y(e),t.push.apply(t,e.getClientRects());return!1}Fp.prototype.getRootNode=function(e){var t=h(this._fragmentFiber);return t===null?this:y(t).getRootNode(e)},Fp.prototype.compareDocumentPosition=function(e){var t=h(this._fragmentFiber);if(t===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];m(this._fragmentFiber.child,!1,Hp,n,void 0,void 0);var r=y(t);if(n.length===0){if(n=r,g(this._fragmentFiber)){a:{for(t=this._fragmentFiber.return;t!==null;){if(t.tag===4){t=t.stateNode.containerInfo;break a}if(t.tag===3||t.tag===5||t.tag===27)break;t=t.return}t=null}t!=null&&(n=t)}t=this._fragmentFiber;var i=r=n.compareDocumentPosition(e);return n===e?i=Node.DOCUMENT_POSITION_CONTAINS:r&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=_(t)[1],n===null?i=Node.DOCUMENT_POSITION_PRECEDING:(e=y(n).compareDocumentPosition(e),i=e===0||e&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),i|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}t=y(n[0]),i=y(n[n.length-1]);var a=g(this._fragmentFiber)?t.parentElement:r;if(a==null)return Node.DOCUMENT_POSITION_DISCONNECTED;r=a.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_CONTAINED_BY,a=a.compareDocumentPosition(i)&Node.DOCUMENT_POSITION_CONTAINED_BY;var o=t.compareDocumentPosition(e),s=i.compareDocumentPosition(e),c=o&Node.DOCUMENT_POSITION_CONTAINED_BY||s&Node.DOCUMENT_POSITION_CONTAINED_BY;return s=r&&a&&o&Node.DOCUMENT_POSITION_FOLLOWING&&s&Node.DOCUMENT_POSITION_PRECEDING,t=r&&t===e||a&&i===e||c||s?Node.DOCUMENT_POSITION_CONTAINED_BY:!r&&t===e||!a&&i===e?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:o,t&Node.DOCUMENT_POSITION_DISCONNECTED||t&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Xp(t,this._fragmentFiber,n[0],n[n.length-1],e)?t:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Xp(e,t,n,r,i){var a=Ut(i);if(e&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!a)a:{for(;a!==null;){if(a.tag===7&&(a===t||a.alternate===t)){n=!0;break a}a=a.return}n=!1}return n}if(e&Node.DOCUMENT_POSITION_CONTAINS){if(a===null)return a=i.ownerDocument,i===a||i===a.documentElement||i===a.body;a:{for(a=t,t=h(t);a!==null;){if(!(a.tag!==5&&a.tag!==3&&a.tag!==27||a!==t&&a.alternate!==t)){a=!0;break a}a=a.return}a=!1}return a}return e&Node.DOCUMENT_POSITION_PRECEDING?((t=!!a)&&!(t=a===n)&&(t=ie(n,a,re),t===null?t=!1:(m(t,!0,te,a,n),a=b,b=null,t=a!==null)),t):e&Node.DOCUMENT_POSITION_FOLLOWING?((t=!!a)&&!(t=a===r)&&(t=ie(r,a,re),t===null?t=!1:(m(t,!0,ne,a,r),a=b,ee=b=null,t=a!==null)),t):!1}function Zp(e,t){var n=e.ownerDocument.createRange();n.selectNodeContents(e),e=n.getBoundingClientRect(),window.scrollTo(window.scrollX+e.left,t?window.scrollY+e.top:window.scrollY+e.bottom-window.innerHeight)}Fp.prototype.scrollIntoView=function(e){if(typeof e==`object`)throw Error(o(566));var t=[];m(this._fragmentFiber.child,!1,Hp,t,void 0,void 0);var n=!1!==e;if(t.length===0){var r=_(this._fragmentFiber);if(r=n?r[1]||r[0]||h(this._fragmentFiber):r[0]||r[1],r===null)return;if(r.tag===6){e=y(r),Zp(e,n);return}if(r=y(r),r.nodeType!==9){if(r.nodeType===11){n=`host`in r?r.host:null,n!==null&&n.scrollIntoView(e);return}r.scrollIntoView(e)}}for(r=n?t.length-1:0;r!==(n?-1:t.length);){var i=t[r];i.tag===6?(i=y(i),Zp(i,n)):y(i).scrollIntoView(e),r+=n?-1:1}};function Qp(e,t){return e=y(e),$p(e,t),!1}function $p(e,t){e.reactFragments??=new Set,e.reactFragments.add(t)}function em(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.addEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){for(var r=0,i=0;i<Kp.length;i++){var a=Kp[i];(a.fragmentInstance!==t||a.observer!==n||a.instance!==e)&&(Kp[r++]=a)}Kp.length=r,n.observe(e)}),$p(e,t))}function tm(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.removeEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){typeof n.rootMargin==`string`?Jp(t,n,e):n.unobserve(e)}),e.reactFragments!=null&&e.reactFragments.delete(t))}function nm(e){var t=e.firstChild;for(t&&t.nodeType===10&&(t=t.nextSibling);t;){var n=t;switch(t=t.nextSibling,n.nodeName){case`HTML`:case`HEAD`:case`BODY`:nm(n),Ht(n);continue;case`SCRIPT`:case`STYLE`:continue;case`LINK`:if(n.rel.toLowerCase()===`stylesheet`)continue}e.removeChild(n)}}function rm(e,t,n,r){for(;e.nodeType===1;){var i=n;if(e.nodeName.toLowerCase()!==t.toLowerCase()){if(!r&&(e.nodeName!==`INPUT`||e.type!==`hidden`))break}else if(!r){if(t===`input`&&e.type===`hidden`){var a=i.name==null?null:``+i.name;if(i.type===`hidden`&&e.getAttribute(`name`)===a)return e}else return e}else if(!e[Bt])switch(t){case`meta`:if(!e.hasAttribute(`itemprop`))break;return e;case`link`:if(a=e.getAttribute(`rel`),a===`stylesheet`&&e.hasAttribute(`data-precedence`)||a!==i.rel||e.getAttribute(`href`)!==(i.href==null||i.href===``?null:i.href)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin)||e.getAttribute(`title`)!==(i.title==null?null:i.title))break;return e;case`style`:if(e.hasAttribute(`data-precedence`))break;return e;case`script`:if(a=e.getAttribute(`src`),(a!==(i.src==null?null:i.src)||e.getAttribute(`type`)!==(i.type==null?null:i.type)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin))&&a&&e.hasAttribute(`async`)&&!e.hasAttribute(`itemprop`))break;return e;default:return e}if(e=lm(e.nextSibling),e===null)break}return null}function im(e,t,n){if(t===``)return null;for(;e.nodeType!==3;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!n||(e=lm(e.nextSibling),e===null))return null;return e}function am(e,t){for(;e.nodeType!==8;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!t||(e=lm(e.nextSibling),e===null))return null;return e}function om(e){return e.data===`$?`||e.data===`$~`}function sm(e){return e.data===`$!`||e.data===`$?`&&e.ownerDocument.readyState!==`loading`}function cm(e,t){var n=e.ownerDocument;if(e.data===`$~`)e._reactRetry=t;else if(e.data!==`$?`||n.readyState!==`loading`)t();else{var r=function(){t(),n.removeEventListener(`DOMContentLoaded`,r)};n.addEventListener(`DOMContentLoaded`,r),e._reactRetry=r}}function lm(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t===`$`||t===`$!`||t===`$?`||t===`$~`||t===`&`||t===`F!`||t===`F`)break;if(t===`/$`||t===`/&`)return null}}return e}var um=null;function dm(e){e=e.nextSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`/$`||n===`/&`){if(t===0)return lm(e.nextSibling);t--}else n!==`$`&&n!==`$!`&&n!==`$?`&&n!==`$~`&&n!==`&`||t++}e=e.nextSibling}return null}function fm(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`$`||n===`$!`||n===`$?`||n===`$~`||n===`&`){if(t===0)return e;t--}else n!==`/$`&&n!==`/&`||t++}e=e.previousSibling}return null}function pm(e,t){function n(){r=!0}if(e.ownerDocument.activeElement===e)return!0;var r=!1;try{e.ownerDocument.addEventListener(`focus`,n,!0),(e.focus||HTMLElement.prototype.focus).call(e,t)}finally{e.ownerDocument.removeEventListener(`focus`,n,!0)}return r}function mm(e){yp(function(){yp(function(t){return e(t)})})}function hm(e,t,n){switch(t=lp(n),e){case`html`:if(e=t.documentElement,!e)throw Error(o(452));return e;case`head`:if(e=t.head,!e)throw Error(o(453));return e;case`body`:if(e=t.body,!e)throw Error(o(454));return e;default:throw Error(o(451))}}function gm(e,t,n){for(var r in n){var i=n[r];n.hasOwnProperty(r)&&i!=null&&$(e,t,r,null,rp,i)}n.dangerouslySetInnerHTML!=null&&(e.textContent=``),e.onclick===On&&(e.onclick=null),Ht(e)}function _m(e){for(var t=e.attributes;t.length;)e.removeAttributeNode(t[0]);Ht(e)}var vm=new Map,ym=new Set;function bm(e){if(typeof e.getRootNode==`function`){var t=e.getRootNode();if(t.nodeType===9||t.nodeType===11)return t}return e.nodeType===9?e:e.ownerDocument}var xm=T.d;T.d={f:Sm,r:Cm,D:Em,C:Dm,L:Om,m:km,X:jm,S:Am,M:Mm};function Sm(){var e=xm.f(),t=zd();return e||t}function Cm(e){var t=Wt(e);t!==null&&t.tag===5&&t.type===`form`?ac(t):xm.r(e)}var wm=typeof document>`u`?null:document;function Tm(e,t,n){var r=wm;if(r&&typeof t==`string`&&t){var i=pn(t);i=`link[rel="`+e+`"][href="`+i+`"]`,typeof n==`string`&&(i+=`[crossorigin="`+n+`"]`),ym.has(i)||(ym.add(i),e={rel:e,crossOrigin:n,href:t},r.querySelector(i)===null&&(t=r.createElement(`link`),np(t,`link`,e),O(t),r.head.appendChild(t)))}}function Em(e){xm.D(e),Tm(`dns-prefetch`,e,null)}function Dm(e,t){xm.C(e,t),Tm(`preconnect`,e,t)}function Om(e,t,n){xm.L(e,t,n);var r=wm;if(r&&e&&t){var i=`link[rel="preload"][as="`+pn(t)+`"]`;t===`image`&&n&&n.imageSrcSet?(i+=`[imagesrcset="`+pn(n.imageSrcSet)+`"]`,typeof n.imageSizes==`string`&&(i+=`[imagesizes="`+pn(n.imageSizes)+`"]`)):i+=`[href="`+pn(e)+`"]`;var a=i;switch(t){case`style`:a=Pm(e);break;case`script`:a=Rm(e)}if(!(vm.has(a)||(e=x({rel:`preload`,href:t===`image`&&n&&n.imageSrcSet?void 0:e,as:t},n),vm.set(a,e),r.querySelector(i)!==null||t===`style`&&r.querySelector(Fm(a))||t===`script`&&r.querySelector(zm(a))))){var o=r.createElement(`link`);np(o,`link`,e),t===`style`&&(o[Vt]=!0,o.onload=o.onerror=function(){qt(o)}),O(o),r.head.appendChild(o)}}}function km(e,t){xm.m(e,t);var n=wm;if(n&&e){var r=t&&typeof t.as==`string`?t.as:`script`,i=`link[rel="modulepreload"][as="`+pn(r)+`"][href="`+pn(e)+`"]`,a=i;switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:a=Rm(e)}if(!vm.has(a)&&(e=x({rel:`modulepreload`,href:e},t),vm.set(a,e),n.querySelector(i)===null)){switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:if(n.querySelector(zm(a)))return}r=n.createElement(`link`),np(r,`link`,e),O(r),n.head.appendChild(r)}}}function Am(e,t,n){xm.S(e,t,n);var r=wm;if(r&&e){var i=Kt(r).hoistableStyles,a=Pm(e);t||=`default`;var o=i.get(a);if(!o){var s={loading:0,preload:null};if(o=r.querySelector(Fm(a)))s.loading=5;else{e=x({rel:`stylesheet`,href:e,"data-precedence":t},n),(n=vm.get(a))&&Hm(e,n);var c=o=r.createElement(`link`);O(c),np(c,`link`,e),c._p=new Promise(function(e,t){c.onload=e,c.onerror=t}),c.addEventListener(`load`,function(){s.loading|=1}),c.addEventListener(`error`,function(){s.loading|=2}),s.loading|=4,Vm(o,t,r)}o={type:`stylesheet`,instance:o,count:1,state:s},i.set(a,o)}}}function jm(e,t){xm.X(e,t);var n=wm;if(n&&e){var r=Kt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=x({src:e,async:!0},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),O(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Mm(e,t){xm.M(e,t);var n=wm;if(n&&e){var r=Kt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=x({src:e,async:!0,type:`module`},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),O(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Nm(e,t,n,r){var i=(i=Pe.current)?bm(i):null;if(!i)throw Error(o(446));switch(e){case`meta`:case`title`:return null;case`style`:return typeof n.precedence==`string`&&typeof n.href==`string`?(n=Pm(n.href),t=Kt(i).hoistableStyles,r=t.get(n),r||(r={type:`style`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};case`link`:if(n.rel===`stylesheet`&&typeof n.href==`string`&&typeof n.precedence==`string`){e=Pm(n.href);var a=Kt(i).hoistableStyles,s=a.get(e);if(s||(i=i.ownerDocument||i,s={type:`stylesheet`,instance:null,count:0,state:{loading:0,preload:null}},a.set(e,s),(a=i.querySelector(Fm(e)))?a._p||(s.instance=a,s.state.loading=5):(a=vm.get(e),a||(a={rel:`preload`,as:`style`,href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},vm.set(e,a)),Lm(i,e,a,s.state))),t&&r===null)throw Error(o(528,``));return s}if(t&&r!==null)throw Error(o(529,``));return null;case`script`:return t=n.async,n=n.src,typeof n==`string`&&t&&typeof t!=`function`&&typeof t!=`symbol`?(n=Rm(n),t=Kt(i).hoistableScripts,r=t.get(n),r||(r={type:`script`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};default:throw Error(o(444,e))}}function Pm(e){return`href="`+pn(e)+`"`}function Fm(e){return`link[rel="stylesheet"][`+e+`]`}function Im(e){return x({},e,{"data-precedence":e.precedence,precedence:null})}function Lm(e,t,n,r){if(t=e.querySelector(`link[rel="preload"][as="style"][`+t+`]`)){if(!0!==t[Vt]){r.loading=1;return}}else t=e.createElement(`link`),t[Vt]=!0,t.onload=t.onerror=qt.bind(null,t),np(t,`link`,n),O(t),e.head.appendChild(t);r.preload=t,t.addEventListener(`load`,function(){return r.loading|=1}),t.addEventListener(`error`,function(){return r.loading|=2})}function Rm(e){return`[src="`+pn(e)+`"]`}function zm(e){return`script[async]`+e}function Bm(e,t,n){if(t.count++,t.instance===null)switch(t.type){case`style`:var r=e.querySelector(`style[data-href~="`+pn(n.href)+`"]`);if(r)return t.instance=r,O(r),r;var i=x({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return r=(e.ownerDocument||e).createElement(`style`),O(r),np(r,`style`,i),Vm(r,n.precedence,e),t.instance=r;case`stylesheet`:i=Pm(n.href);var a=e.querySelector(Fm(i));if(a)return t.state.loading|=4,t.instance=a,O(a),a;r=Im(n),(i=vm.get(i))&&Hm(r,i),a=(e.ownerDocument||e).createElement(`link`),O(a);var s=a;return s._p=new Promise(function(e,t){s.onload=e,s.onerror=t}),np(a,`link`,r),t.state.loading|=4,Vm(a,n.precedence,e),t.instance=a;case`script`:return a=Rm(n.src),(i=e.querySelector(zm(a)))?(t.instance=i,O(i),i):(r=n,(i=vm.get(a))&&(r=x({},n),Um(r,i)),e=e.ownerDocument||e,i=e.createElement(`script`),O(i),np(i,`link`,r),e.head.appendChild(i),t.instance=i);case`void`:return null;default:throw Error(o(443,t.type))}else t.type===`stylesheet`&&!(t.state.loading&4)&&(r=t.instance,t.state.loading|=4,Vm(r,n.precedence,e));return t.instance}function Vm(e,t,n){for(var r=n.querySelectorAll(`link[rel="stylesheet"][data-precedence],style[data-precedence]`),i=r.length?r[r.length-1]:null,a=i,o=0;o<r.length;o++){var s=r[o];if(s.dataset.precedence===t)a=s;else if(a!==i)break}a?a.parentNode.insertBefore(e,a.nextSibling):(t=n.nodeType===9?n.head:n,t.insertBefore(e,t.firstChild))}function Hm(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.title??=t.title}function Um(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.integrity??=t.integrity}var Wm=null;function Gm(e,t,n){if(Wm===null){var r=new Map,i=Wm=new Map;i.set(n,r)}else i=Wm,r=i.get(n),r||(r=new Map,i.set(n,r));if(r.has(e))return r;for(r.set(e,null),n=n.getElementsByTagName(e),i=0;i<n.length;i++){var a=n[i];if(!(a[Bt]||a[Nt]||e===`link`&&a.getAttribute(`rel`)===`stylesheet`)&&a.namespaceURI!==`http://www.w3.org/2000/svg`){var o=a.getAttribute(t)||``;o=e+o;var s=r.get(o);s?s.push(a):r.set(o,[a])}}return r}function Km(e,t,n){e=e.ownerDocument||e,e.head.insertBefore(n,t===`title`?e.querySelector(`head > title`):null)}function qm(e,t,n){if(n===1||t.itemProp!=null)return!1;switch(e){case`meta`:case`title`:return!0;case`style`:if(typeof t.precedence!=`string`||typeof t.href!=`string`||t.href===``)break;return!0;case`link`:if(typeof t.rel!=`string`||typeof t.href!=`string`||t.href===``||t.onLoad||t.onError)break;switch(t.rel){case`stylesheet`:return e=t.disabled,typeof t.precedence==`string`&&e==null;default:return!0}case`script`:if(t.async&&typeof t.async!=`function`&&typeof t.async!=`symbol`&&!t.onLoad&&!t.onError&&t.src&&typeof t.src==`string`)return!0}return!1}function Jm(e,t){return e===`img`&&t.src!=null&&t.src!==``&&t.onLoad==null&&t.loading!==`lazy`}function Ym(e){return!(e.type===`stylesheet`&&!(e.state.loading&3))}function Xm(e){return(e.width||100)*(e.height||100)*(typeof devicePixelRatio==`number`?devicePixelRatio:1)*.25}function Zm(e,t){typeof t.decode==`function`&&(e.imgCount++,t.complete||(e.imgBytes+=Xm(t),e.suspenseyImages.push(t)),e=rh.bind(e),t.decode().then(e,e))}function Qm(e,t,n,r){if(n.type===`stylesheet`&&(typeof r.media!=`string`||!1!==matchMedia(r.media).matches)&&!(n.state.loading&4)){if(n.instance===null){var i=Pm(r.href),a=t.querySelector(Fm(i));if(a){t=a._p,typeof t==`object`&&t&&typeof t.then==`function`&&(e.count++,e=nh.bind(e),t.then(e,e)),n.state.loading|=4,n.instance=a,O(a);return}a=t.ownerDocument||t,r=Im(r),(i=vm.get(i))&&Hm(r,i),a=a.createElement(`link`),O(a);var o=a;o._p=new Promise(function(e,t){o.onload=e,o.onerror=t}),np(a,`link`,r),n.instance=a}e.stylesheets===null&&(e.stylesheets=new Map),e.stylesheets.set(n,t),(t=n.state.preload)&&!(n.state.loading&3)&&(e.count++,n=nh.bind(e),t.addEventListener(`load`,n),t.addEventListener(`error`,n))}}var $m=0;function eh(e,t){return e.stylesheets&&e.count===0&&ah(e,e.stylesheets),0<e.count||0<e.imgCount?function(n){var r=setTimeout(function(){if(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}},6e4+t);0<e.imgBytes&&$m===0&&($m=62500*op());var i=setTimeout(function(){if(e.waitingForImages=!1,e.count===0&&(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend)){var t=e.unsuspend;e.unsuspend=null,t()}},(e.imgBytes>$m?50:800)+t);return e.unsuspend=n,function(){e.unsuspend=null,clearTimeout(r),clearTimeout(i)}}:null}function th(e){if(e.count===0&&(e.imgCount===0||!e.waitingForImages)){if(e.stylesheets)ah(e,e.stylesheets);else if(e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}}}function nh(){this.count--,th(this)}function rh(){this.imgCount--,th(this)}var ih=null;function ah(e,t){e.stylesheets=null,e.unsuspend!==null&&(e.count++,ih=new Map,t.forEach(oh,e),ih=null,nh.call(e))}function oh(e,t){if(!(t.state.loading&4)){var n=ih.get(e);if(n)var r=n.get(null);else{n=new Map,ih.set(e,n);for(var i=e.querySelectorAll(`link[data-precedence],style[data-precedence]`),a=0;a<i.length;a++){var o=i[a];(o.nodeName===`LINK`||o.getAttribute(`media`)!==`not all`)&&(n.set(o.dataset.precedence,o),r=o)}r&&n.set(null,r)}i=t.instance,o=i.getAttribute(`data-precedence`),a=n.get(o)||r,a===r&&n.set(null,i),n.set(o,i),this.count++,r=nh.bind(this),i.addEventListener(`load`,r),i.addEventListener(`error`,r),a?a.parentNode.insertBefore(i,a.nextSibling):(e=e.nodeType===9?e.head:e,e.insertBefore(i,e.firstChild)),t.state.loading|=4}}var sh={$$typeof:S,Provider:null,Consumer:null,_currentValue:De,_currentValue2:De,_threadCount:0};function ch(e,t,n,r,i,a,o,s,c){this.tag=1,this.containerInfo=e,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=St(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=St(0),this.hiddenUpdates=St(null),this.identifierPrefix=r,this.onUncaughtError=i,this.onCaughtError=a,this.onRecoverableError=o,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=c,this.transitionTypes=null,this.incompleteTransitions=new Map}function lh(e,t,n,r,i,a,o,s,c,l,u,d){return e=new ch(e,t,n,o,c,l,u,d,s),t=1,!0===a&&(t|=24),a=Vi(3,null,null,t),e.current=a,a.stateNode=e,t=za(),t.refCount++,e.pooledCache=t,t.refCount++,a.memoizedState={element:r,isDehydrated:n,cache:t},So(a),e}function uh(e){return e?(e=zi,e):zi}function dh(e,t,n,r,i,a){i=uh(i),r.context===null?r.context=i:r.pendingContext=i,r=wo(t),r.payload={element:n},a=a===void 0?null:a,a!==null&&(r.callback=a),n=To(e,r,t),n!==null&&(Pd(n,e,t),Eo(n,e,t))}function fh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ph(e,t){fh(e,t),(e=e.alternate)&&fh(e,t)}function mh(e){if(e.tag===13||e.tag===31){var t=Ii(e,67108864);t!==null&&Pd(t,e,67108864),ph(e,67108864)}}function hh(e){if(e.tag===13||e.tag===31){var t=jd();t=Ot(t);var n=Ii(e,t);n!==null&&Pd(n,e,t),ph(e,t)}}var gh=!0;function _h(e,t,n,r){var i=w.T;w.T=null;var a=T.p;try{T.p=2,yh(e,t,n,r)}finally{T.p=a,w.T=i}}function vh(e,t,n,r){var i=w.T;w.T=null;var a=T.p;try{T.p=8,yh(e,t,n,r)}finally{T.p=a,w.T=i}}function yh(e,t,n,r){if(gh){var i=bh(r);if(i===null)Kf(e,t,r,xh,n),Mh(e,r);else if(Ph(i,e,t,n,r))r.stopPropagation();else if(Mh(e,r),t&4&&-1<jh.indexOf(e)){for(;i!==null;){var a=Wt(i);if(a!==null)switch(a.tag){case 3:if(a=a.stateNode,a.current.memoizedState.isDehydrated){var o=gt(a.pendingLanes);if(o!==0){var s=a;for(s.pendingLanes|=2,s.entangledLanes|=2;o;){var c=1<<31-D(o);s.entanglements[1]|=c,o&=~c}Ef(a),!(W&6)&&(_d=Qe()+500,Df(0,!1))}}break;case 31:case 13:s=Ii(a,2),s!==null&&Pd(s,a,2),zd(),ph(a,2)}if(a=bh(r),a===null&&Kf(e,t,r,xh,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Kf(e,t,r,null,n)}}function bh(e){return e=An(e),Sh(e)}var xh=null;function Sh(e){if(xh=null,e=Ut(e),e!==null){var t=c(e);if(t===null)e=null;else{var n=t.tag;if(n===13){if(e=l(t),e!==null)return e;e=null}else if(n===31){if(e=u(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null)}}return xh=e,null}function Ch(e){switch(e){case`beforetoggle`:case`cancel`:case`click`:case`close`:case`contextmenu`:case`copy`:case`cut`:case`auxclick`:case`dblclick`:case`dragend`:case`dragstart`:case`drop`:case`focusin`:case`focusout`:case`input`:case`invalid`:case`keydown`:case`keypress`:case`keyup`:case`mousedown`:case`mouseup`:case`paste`:case`pause`:case`play`:case`pointercancel`:case`pointerdown`:case`pointerup`:case`ratechange`:case`reset`:case`seeked`:case`submit`:case`toggle`:case`touchcancel`:case`touchend`:case`touchstart`:case`volumechange`:case`change`:case`selectionchange`:case`textInput`:case`compositionstart`:case`compositionend`:case`compositionupdate`:case`beforeblur`:case`afterblur`:case`beforeinput`:case`blur`:case`fullscreenchange`:case`fullscreenerror`:case`focus`:case`hashchange`:case`popstate`:case`select`:case`selectstart`:return 2;case`drag`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`mousemove`:case`mouseout`:case`mouseover`:case`pointermove`:case`pointerout`:case`pointerover`:case`resize`:case`scroll`:case`touchmove`:case`wheel`:case`mouseenter`:case`mouseleave`:case`pointerenter`:case`pointerleave`:return 8;case`message`:switch($e()){case et:return 2;case tt:return 8;case nt:case rt:return 32;case it:return 268435456;default:return 32}default:return 32}}var wh=!1,Th=null,Eh=null,Dh=null,Oh=new Map,kh=new Map,Ah=[],jh=`mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset`.split(` `);function Mh(e,t){switch(e){case`focusin`:case`focusout`:Th=null;break;case`dragenter`:case`dragleave`:Eh=null;break;case`mouseover`:case`mouseout`:Dh=null;break;case`pointerover`:case`pointerout`:Oh.delete(t.pointerId);break;case`gotpointercapture`:case`lostpointercapture`:kh.delete(t.pointerId)}}function Nh(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Wt(t),t!==null&&mh(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Ph(e,t,n,r,i){switch(t){case`focusin`:return Th=Nh(Th,e,t,n,r,i),!0;case`dragenter`:return Eh=Nh(Eh,e,t,n,r,i),!0;case`mouseover`:return Dh=Nh(Dh,e,t,n,r,i),!0;case`pointerover`:var a=i.pointerId;return Oh.set(a,Nh(Oh.get(a)||null,e,t,n,r,i)),!0;case`gotpointercapture`:return a=i.pointerId,kh.set(a,Nh(kh.get(a)||null,e,t,n,r,i)),!0}return!1}function Fh(e){var t=Ut(e.target);if(t!==null){var n=c(t);if(n!==null){if(t=n.tag,t===13){if(t=l(n),t!==null){e.blockedOn=t,jt(e.priority,function(){hh(n)});return}}else if(t===31){if(t=u(n),t!==null){e.blockedOn=t,jt(e.priority,function(){hh(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ih(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=bh(e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);kn=r,n.target.dispatchEvent(r),kn=null}else return t=Wt(n),t!==null&&mh(t),e.blockedOn=n,!1;t.shift()}return!0}function Lh(e,t,n){Ih(e)&&n.delete(t)}function Rh(){wh=!1,Th!==null&&Ih(Th)&&(Th=null),Eh!==null&&Ih(Eh)&&(Eh=null),Dh!==null&&Ih(Dh)&&(Dh=null),Oh.forEach(Lh),kh.forEach(Lh)}function zh(e,t){e.blockedOn===t&&(e.blockedOn=null,wh||(wh=!0,n.unstable_scheduleCallback(n.unstable_NormalPriority,Rh)))}var Bh=null;function Vh(e){Bh!==e&&(Bh=e,n.unstable_scheduleCallback(n.unstable_NormalPriority,function(){Bh===e&&(Bh=null);for(var t=0;t<e.length;t+=3){var n=e[t],r=e[t+1],i=e[t+2];if(typeof r!=`function`){if(Sh(r||n)===null)continue;break}var a=Wt(n);a!==null&&(e.splice(t,3),t-=3,rc(a,{pending:!0,data:i,method:n.method,action:r},r,i))}}))}function Hh(e){function t(t){return zh(t,e)}Th!==null&&zh(Th,e),Eh!==null&&zh(Eh,e),Dh!==null&&zh(Dh,e),Oh.forEach(t),kh.forEach(t);for(var n=0;n<Ah.length;n++){var r=Ah[n];r.blockedOn===e&&(r.blockedOn=null)}for(;0<Ah.length&&(n=Ah[0],n.blockedOn===null);)Fh(n),n.blockedOn===null&&Ah.shift();if(n=(e.ownerDocument||e).$$reactFormReplay,n!=null)for(r=0;r<n.length;r+=3){var i=n[r],a=n[r+1],o=i[Pt]||null;if(typeof a==`function`)o||Vh(n);else if(o){var s=null;if(a&&a.hasAttribute(`formAction`)){if(i=a,o=a[Pt]||null)s=o.formAction;else if(Sh(i)!==null)continue}else s=o.action;typeof s==`function`?n[r+1]=s:(n.splice(r,3),r-=3),Vh(n)}}}function Uh(){function e(e){e.canIntercept&&e.info===`react-transition`&&e.intercept({handler:function(){return new Promise(function(e){return i=e})},focusReset:`manual`,scroll:`manual`})}function t(){i!==null&&(i(),i=null),r||setTimeout(n,20)}function n(){if(!r&&!navigation.transition){var e=navigation.currentEntry;e&&e.url!=null&&navigation.navigate(e.url,{state:e.getState(),info:`react-transition`,history:`replace`})}}if(typeof navigation==`object`){var r=!1,i=null;return navigation.addEventListener(`navigate`,e),navigation.addEventListener(`navigatesuccess`,t),navigation.addEventListener(`navigateerror`,t),setTimeout(n,100),function(){r=!0,navigation.removeEventListener(`navigate`,e),navigation.removeEventListener(`navigatesuccess`,t),navigation.removeEventListener(`navigateerror`,t),i!==null&&(i(),i=null)}}}function Wh(e){this._internalRoot=e}Gh.prototype.render=Wh.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(o(409));var n=t.current;dh(n,jd(),e,t,null,null)},Gh.prototype.unmount=Wh.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;dh(e.current,2,null,e,null,null),zd(),t[Ft]=null}};function Gh(e){this._internalRoot=e}Gh.prototype.unstable_scheduleHydration=function(e){if(e){var t=At();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Ah.length&&t!==0&&t<Ah[n].priority;n++);Ah.splice(n,0,e),n===0&&Fh(e)}};var Kh=r.version;if(Kh!==`19.3.0`)throw Error(o(527,Kh,`19.3.0`));T.findDOMNode=function(e){var t=e._reactInternals;if(t===void 0)throw typeof e.render==`function`?Error(o(188)):(e=Object.keys(e).join(`,`),Error(o(268,e)));return e=f(t),e=e===null?null:p(e),e=e===null?null:e.stateNode,e};var qh={bundleType:0,version:`19.3.0`,rendererPackageName:`react-dom`,currentDispatcherRef:w,reconcilerVersion:`19.3.0`};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`){var Jh=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Jh.isDisabled&&Jh.supportsFiber)try{st=Jh.inject(qh),ct=Jh}catch{}}e.createRoot=function(e,t){if(!s(e))throw Error(o(299));var n=!1,r=``,i=Ec,a=Dc,c=Oc;return t!=null&&(!0===t.unstable_strictMode&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onUncaughtError!==void 0&&(i=t.onUncaughtError),t.onCaughtError!==void 0&&(a=t.onCaughtError),t.onRecoverableError!==void 0&&(c=t.onRecoverableError)),t=lh(e,1,!1,null,null,n,r,null,i,a,c,Uh),e[Ft]=t.current,Wf(e),new Wh(t)}})),fe=n(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=de()})),S=e(t(),1),pe=fe();function me(e){let t=``;for(let n of e){let e=n.codePointAt(0)??0;t+=n===`"`||n===`\\`||e<32||e===127?`\\${(e===0?65533:e).toString(16)} `:n}return t}function he(e){return e===`base`?``:e.split(`+`).map(e=>{let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?e:e.slice(t+1);return`[${u(n)}="${me(r)}"]`}).join(``)}var ge={transparent:{r:0,g:0,b:0,a:0},black:{r:0,g:0,b:0,a:1},white:{r:255,g:255,b:255,a:1}},_e=(e,t,n)=>Math.max(t,Math.min(n,e));function ve(e){return e.split(``).map(e=>e+e).join(``)}function ye(e){if(typeof e!=`string`)return null;let t=e.trim().replace(/^#/,``),n=t.length===3||t.length===4?ve(t):t;return n.length!==6&&n.length!==8||!/^[0-9a-fA-F]+$/.test(n)?null:{r:parseInt(n.slice(0,2),16),g:parseInt(n.slice(2,4),16),b:parseInt(n.slice(4,6),16),a:n.length===8?parseInt(n.slice(6,8),16)/255:1}}function be(e){let t=e.indexOf(`(`);if(t===-1||!e.trim().endsWith(`)`))return null;let n=e.slice(t+1,e.lastIndexOf(`)`)).replace(/\//g,` `).split(/[\s,]+/).map(e=>e.trim()).filter(Boolean);if(n.length<3)return null;let r=e=>_e(e.endsWith(`%`)?parseFloat(e)/100*255:parseFloat(e),0,255),i=r(n[0]),a=r(n[1]),o=r(n[2]);if([i,a,o].some(Number.isNaN))return null;let s=1;if(n.length>=4){let e=n[3];if(s=e.endsWith(`%`)?parseFloat(e)/100:parseFloat(e),Number.isNaN(s))return null;s=_e(s,0,1)}return{r:i,g:a,b:o,a:s}}function xe(e){let t=e.trim(),n=ge[t.toLowerCase()];return n?{...n}:t.startsWith(`#`)?ye(t):/^rgba?\(/i.test(t)?be(t):null}function Se(e,t,n){let r=e=>_e(Math.round(e),0,255).toString(16).padStart(2,`0`).toUpperCase();return`#${r(e)}${r(t)}${r(n)}`}function Ce(e){try{let t=new Intl.Locale(e);return(typeof t.getTextInfo==`function`?t.getTextInfo():t.textInfo)?.direction===`rtl`?`rtl`:`ltr`}catch{return`ltr`}}var C=r();function we(e){return Object.values(e).some(e=>Object.values(e).some(e=>typeof e==`string`))?Object.fromEntries(Object.entries(e).map(([e,t])=>[e,Object.fromEntries(Object.entries(t).map(([e,t])=>[e,typeof t==`string`?{defaultMessage:t}:t]))])):e}function Te({locale:e,messages:t,overrides:n,dir:r,children:i}){let a=r??Ce(e),o=(0,S.useMemo)(()=>{let r=t??{};return{locale:e,direction:a,messages:we(r),overrides:n,translate:_(e,r,n)}},[e,a,t,n]);return(0,C.jsx)(y,{value:o,children:i})}Te.displayName=`InternationalizationProvider`;function Ee(e,t){if(!e&&!t)return;if(!e)return t;if(!t)return e;let n={};for(let[t,r]of Object.entries(e))n[t]={...r};for(let[e,r]of Object.entries(t))if(!n[e])n[e]={...r};else for(let[t,i]of Object.entries(r))n[e][t]={...n[e][t],...i};return n}var w={[-5]:`--font-size-4xs`,[-4]:`--font-size-3xs`,[-3]:`--font-size-2xs`,[-2]:`--font-size-xs`,[-1]:`--font-size-sm`,0:`--font-size-base`,1:`--font-size-lg`,2:`--font-size-xl`,3:`--font-size-2xl`,4:`--font-size-3xl`,5:`--font-size-4xl`,6:`--font-size-5xl`},T={1:3,2:2,3:1,4:0,5:-1,6:-2},De={body:0,large:1,label:0,code:0,supporting:-1,"display-1":6,"display-2":5,"display-3":4},Oe={1:`var(--font-weight-semibold)`,2:`var(--font-weight-semibold)`,3:`var(--font-weight-semibold)`,4:`var(--font-weight-semibold)`,5:`var(--font-weight-semibold)`,6:`var(--font-weight-semibold)`},ke={body:`var(--font-weight-normal)`,large:`var(--font-weight-semibold)`,label:`var(--font-weight-medium)`,code:`var(--font-weight-normal)`,supporting:`var(--font-weight-normal)`,"display-1":`var(--font-weight-normal)`,"display-2":`var(--font-weight-normal)`,"display-3":`var(--font-weight-normal)`};function Ae(e,t,n){return Math.round(e*t**+n)}function je(e){return`${Math.round(e/16*1e4)/1e4}rem`}function E(e){return e<20?1.5:e<32?1.4:1.25}function Me(e){let t=e*E(e),n=Math.max(Math.round(t/4)*4,Math.ceil((e+4)/4)*4);return Math.round(n/e*1e4)/1e4}function Ne(e){let{base:t,ratio:n,weights:r}=e,i={},a={...Oe,...r?.heading},o={...ke,...r?.text};for(let e=-5;e<=6;e++){let r=Ae(t,n,e);i[w[e]]=je(r)}for(let[e,r]of Object.entries(T)){let o=Number(e),s=Me(Ae(t,n,r));i[`--text-heading-${o}-size`]=`var(${w[r]})`,i[`--text-heading-${o}-weight`]=a[o],i[`--text-heading-${o}-leading`]=`${s}`}for(let[e,r]of Object.entries(De)){let a=Me(Ae(t,n,r));i[`--text-${e}-size`]=`var(${w[r]})`,i[`--text-${e}-weight`]=o[e],i[`--text-${e}-leading`]=`${a}`}return i}var Pe={body:`var(--font-family-body)`,large:`var(--font-family-body)`,label:`var(--font-family-body)`,code:`var(--font-family-code)`,supporting:`var(--font-family-body)`,"display-1":`var(--font-family-heading)`,"display-2":`var(--font-family-heading)`,"display-3":`var(--font-family-heading)`};function Fe(e){let t={},n={};for(let e of[1,2,3,4,5,6])n[`level:${e}`]={fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-${e}-size)`,fontWeight:`var(--text-heading-${e}-weight)`,lineHeight:`var(--text-heading-${e}-leading)`};for(let e of[`display-1`,`display-2`,`display-3`])n[`type:${e}`]={fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-${e}-size)`,lineHeight:`var(--text-${e}-leading)`};t.heading=n;let r={};for(let e of[`body`,`large`,`label`,`code`,`supporting`,`display-1`,`display-2`,`display-3`])r[`type:${e}`]={fontFamily:Pe[e],fontSize:`var(--text-${e}-size)`,lineHeight:`var(--text-${e}-leading)`};return t.text=r,t}function Ie(e){return Math.round(e/5)*5}function Le(e){let{fast:t,medium:n,slow:r,ratio:i,easing:a}=e,o={"--duration-fast-min":`${Ie(t*i)}ms`,"--duration-fast":`${Ie(t)}ms`,"--duration-fast-max":`${Ie(t/i)}ms`,"--duration-medium-min":`${Ie(n*i)}ms`,"--duration-medium":`${Ie(n)}ms`,"--duration-medium-max":`${Ie(n/i)}ms`};return r!=null&&(o[`--duration-slow-min`]=`${Ie(r*i)}ms`,o[`--duration-slow`]=`${Ie(r)}ms`,o[`--duration-slow-max`]=`${Ie(r/i)}ms`),a&&(o[`--ease-standard`]=a),o}function Re(e){let{base:t,multiplier:n}=e;return{"--radius-none":`0px`,"--radius-inner":`${Math.round(t*1*n)}px`,"--radius-element":`${Math.round(t*2*n)}px`,"--radius-container":`${Math.round(t*3*n)}px`,"--radius-page":`${Math.round(t*7*n)}px`,"--radius-chat":`${Math.round(t*7*n)}px`,"--radius-full":`9999px`}}function ze(e){let t=e=>{let t=e/255;return t<=.04045?t/12.92:((t+.055)/1.055)**2.4};return .2126*t(e.r)+.7152*t(e.g)+.0722*t(e.b)}function Be(e,t){let n=e.a;return{r:e.r*n+t.r*(1-n),g:e.g*n+t.g*(1-n),b:e.b*n+t.b*(1-n),a:1}}function Ve(e,t){if(typeof e!=`string`)return e;let n=xe(e);if(n===null)throw TypeError(`contrastRatio: could not parse ${t} "${e}"`);return n}function He(e,t){let n=Ve(t,`background`);if(n.a<1)throw TypeError(`contrastRatio: background must be opaque — composite it over its backdrop first`);let r=Ve(e,`foreground`);r.a<1&&(r=Be(r,n));let i=ze(r),a=ze(n),o=Math.max(i,a),s=Math.min(i,a);return(o+.05)/(s+.05)}function Ue(e){let t=e/255;return t<=.04045?t/12.92:((t+.055)/1.055)**2.4}function We(e){let t=e<=.0031308?e*12.92:1.055*e**(1/2.4)-.055;return Math.round(Math.min(255,Math.max(0,t*255)))}function Ge(e,t,n){return[.4124564*e+.3575761*t+.1804375*n,.2126729*e+.7151522*t+.072175*n,.0193339*e+.119192*t+.9503041*n]}function Ke(e,t,n){return[3.2404542*e-1.5371385*t-.4985314*n,-.969266*e+1.8760108*t+.041556*n,.0556434*e-.2040259*t+1.0572252*n]}var qe=[.95047,1,1.08883];function Je(e){let t=6/29;return e>t*t*t?Math.cbrt(e):e/(3*t*t)+4/29}function Ye(e){let t=6/29;return e>t?e*e*e:3*t*t*(e-4/29)}function Xe(e,t,n){let r=Je(e/qe[0]),i=Je(t/qe[1]),a=Je(n/qe[2]);return[116*i-16,500*(r-i),200*(i-a)]}function Ze(e,t,n){let r=(e+16)/116,i=t/500+r,a=r-n/200;return[Ye(i)*qe[0],Ye(r)*qe[1],Ye(a)*qe[2]]}function Qe(e){return Ye((e+16)/116)}function $e(e){let t=ye(e);return t===null?[0,0,0]:[t.r,t.g,t.b]}function et(e){let[t,n,r]=$e(e),[i,a,o]=Ge(Ue(t),Ue(n),Ue(r)),[s,c,l]=Xe(i,a,o),u=Math.atan2(l,c)*180/Math.PI;u<0&&(u+=360);let d=Math.sqrt(c*c+l*l);return{hue:u,chroma:d,tone:Math.max(0,Math.min(100,s))}}function tt(e){let{hue:t,chroma:n,tone:r}=e;if(r<=0)return`#000000`;if(r>=100)return`#FFFFFF`;if(n<.5){let e=nt(r);return Se(e,e,e)}let i=0,a=n,o=`#000000`;for(let e=0;e<16;e++){let e=(i+a)/2,n=rt(t,e,r);n===null?a=e:(o=n,i=e)}return o}function nt(e){return We(Qe(e))}function rt(e,t,n){let r=e*Math.PI/180,[i,a,o]=Ze(n,Math.cos(r)*t,Math.sin(r)*t),[s,c,l]=Ke(i,a,o),u=We(s),d=We(c),f=We(l),p=Ue(u),m=Ue(d),h=Ue(f),g=.02;return Math.abs(p-s)>g||Math.abs(m-c)>g||Math.abs(h-l)>g||u<0||u>255||d<0||d>255||f<0||f>255?null:Se(u,d,f)}var it=[0,5,10,20,30,40,50,60,70,80,90,95,99,100];function at(e,t){let n={};for(let r of it)n[r]=tt({hue:e,chroma:t,tone:r});return n}function ot(e,t){return e+Math.round(t*255).toString(16).padStart(2,`0`).toUpperCase()}var st={warm:7,cool:5,neutral:3},ct={warm:10,cool:8,neutral:6},lt=`#0064E0`;function D(e,t){return`light-dark(${e}, ${t})`}function ut(e){return`color-mix(in srgb, var(--color-accent) ${e*100}%, transparent)`}var dt=3;function ft(e,t,n,r,i,a){let o=n,s=tt({hue:e,chroma:t,tone:o});for(;He(s,i)<a&&o+r>=0&&o+r<=100;)o+=r,s=tt({hue:e,chroma:t,tone:o});return s}function pt(e){let{accent:t,neutralStyle:n=`cool`,contrast:r=`standard`}=e,[i,a]=Array.isArray(t)?t:[t,t],o=et(i??lt),s=a===i,c=s?o:et(a??lt),l=st[n]??5,u=ct[n]??8,d=at(o.hue,Math.max(o.chroma,48)),f=at(o.hue,l),p=at(o.hue,u),m=s?d:at(c.hue,Math.max(c.chroma,48)),h=s?f:at(c.hue,l),g=s?p:at(c.hue,u),_=r===`high`,v=_?0:10,y=_?99:90,b=_?20:30,ee=_?80:70,te=_?.2:.1,ne=_?50:70,re=_?50:30,ie=D(ft(o.hue,u,ne,-1,f[99],dt),ft(c.hue,u,re,1,h[10],dt));return{...t==null?null:{"--color-accent":D(d[40],m[80]),"--color-accent-muted":D(ut(.2),ut(.25)),"--color-on-accent":D(d[100],m[20])},"--color-neutral":D(ot(f[10],.1),ot(h[90],.2)),"--color-background-surface":D(f[99],h[10]),"--color-background-body":D(f[95],h[5]),"--color-overlay":D(ot(f[10],.4),ot(h[10],.6)),"--color-overlay-hover":D(ot(f[10],.05),ot(h[100],.05)),"--color-overlay-pressed":D(ot(f[10],.1),ot(h[100],.1)),"--color-background-muted":D(ot(f[10],.05),ot(h[10],.5)),"--color-text-primary":D(f[v],h[y]),"--color-text-secondary":D(p[b],g[ee]),"--color-text-disabled":D(p[60],g[40]),"--color-text-accent":`var(--color-accent)`,"--color-icon-accent":`var(--color-accent)`,"--color-icon-primary":D(f[v],h[y]),"--color-icon-secondary":D(p[b],g[ee]),"--color-icon-disabled":D(p[60],g[40]),"--color-background-card":D(f[99],h[10]),"--color-background-popover":D(f[99],h[20]),"--color-background-inverted":D(f[10],h[99]),"--color-border":D(ot(f[10],te),ot(h[95],te)),"--color-border-emphasized":ie,"--color-skeleton":D(p[70],g[30]),"--color-track":D(p[70],g[30]),"--color-shadow":D(ot(f[0],.1),ot(h[0],.3)),"--color-tint-hover":D(`black`,`white`)}}function mt(e,{strict:t=!1}={}){if(typeof e==`string`)return e;if(Array.isArray(e)&&e.length===2&&typeof e[0]==`string`&&typeof e[1]==`string`)return`light-dark(${e[0]}, ${e[1]})`;if(t)throw Error(`Theme token values must be CSS strings or [light, dark] string tuples.`);return Array.isArray(e)?`light-dark(${e[0]}, ${e[1]})`:e}function ht(e){return{normal:`var(--font-weight-normal)`,medium:`var(--font-weight-medium)`,semibold:`var(--font-weight-semibold)`,bold:`var(--font-weight-bold)`}[e]??e}function gt(e,t){if(!e)return;let n=e.includes(` `)?`"${e}"`:e;return t?`${n}, ${t}`:n}function _t(e){if(!e.scale)return;let t={},n=e.heading;if(n?.weights)for(let[e,r]of Object.entries(n.weights))r&&(t[Number(e)]=ht(r));let r=n?.weight?ht(n.weight):void 0;if(r)for(let e=1;e<=6;e++)e in t||(t[e]=r);let i={};return e.body?.weight&&(i.body=ht(e.body.weight)),e.code?.weight&&(i.code=ht(e.code.weight)),{base:e.scale.base,ratio:e.scale.ratio,weights:{...Object.keys(t).length>0?{heading:t}:{},...Object.keys(i).length>0?{text:i}:{}}}}function vt(e){let t={},n=gt(e.body?.family,e.body?.fallbacks),r=gt(e.heading?.family,e.heading?.fallbacks)??n,i=gt(e.code?.family,e.code?.fallbacks);return n&&(t[`--font-family-body`]=n),r&&(t[`--font-family-heading`]=r),i&&(t[`--font-family-code`]=i),t}function yt(e,t,{strictTokens:n=!1}={}){let r={...t?.tokens},i=e.typography,a=i?_t(i):void 0;if(e.color&&Object.assign(r,pt(e.color)),a&&Object.assign(r,Ne(a)),e.radius&&Object.assign(r,Re(e.radius)),e.motion&&Object.assign(r,Le(e.motion)),i&&Object.assign(r,vt(i)),e.syntax)for(let[t,n]of Object.entries(e.syntax.tokens))r[`--color-syntax-`+t]=n;if(e.tokens)for(let[t,i]of Object.entries(e.tokens))i!==void 0&&(r[t]=mt(i,{strict:n}));let o=e.components;return a&&(o=Ee(Fe(a),e.components)),t?.components&&(o=Ee(t.components,o)),{tokens:r,components:o}}var bt=String.raw`(?:[-_a-zA-Z0-9]|\P{ASCII}|\\(?:[0-9a-fA-F]{1,6}[ \t\r\n\f]?|[^\r\n\f]))`;RegExp(`^--${bt}*$`,`u`);var xt=RegExp(`var\\(\\s*(--${bt}*)(?=\\s*(?:,|\\)))`,`giu`);function St(e,t){return Object.prototype.hasOwnProperty.call(e,t)}function Ct(e,t){if(typeof e==`string`)return e;if(Array.isArray(e)&&e.length===2&&typeof e[0]==`string`&&typeof e[1]==`string`)return`light-dark(${e[0]}, ${e[1]})`;throw Error(`${t} must be a CSS string or a [light, dark] string tuple.`)}function wt(e,t){if(typeof e==`string`){xt.lastIndex=0;for(let n=xt.exec(e);n;n=xt.exec(e))t.add(n[1]);return}if(Array.isArray(e)){for(let n of e)wt(n,t);return}if(e&&typeof e==`object`)for(let n of Object.values(e))wt(n,t)}function Tt(e,t,n){let r=new Map;for(let[t,n]of Object.entries(e)){let i=new Set;wt(n,i),r.set(t,[...i].filter(t=>St(e,t)))}let i=(e,t)=>{let n=[e],i=new Set(n),a=o=>{for(let s of r.get(o)??[])if(t.has(s)){if(s===e)return n.push(e),!0;if(!i.has(s)){if(n.push(s),i.add(s),a(s))return!0;i.delete(s),n.pop()}}return!1};return a(e),n},a=0,o=new Map,s=new Map,c=[],l=new Set,u=e=>{let d=a++;o.set(e,d),s.set(e,d),c.push(e),l.add(e);for(let t of r.get(e)??[])o.has(t)?l.has(t)&&s.set(e,Math.min(s.get(e)??d,o.get(t)??0)):(u(t),s.set(e,Math.min(s.get(e)??d,s.get(t)??0)));if(s.get(e)!==d)return;let f=[],p;do p=c.pop()??e,l.delete(p),f.push(p);while(p!==e);if(!(f.length>1||(r.get(f[0])??[]).includes(f[0]))||n&&!f.some(e=>n.has(e)))return;let m=f.find(e=>n?.has(e))??f[0],h=i(m,new Set(f)).join(` -> `);throw Error(`${t?`${t}: `:``}Theme token cycle detected: ${h}.`)};for(let t of Object.keys(e))o.has(t)||u(t)}function Et(e,t,n,r){let i=`defineTheme("${e}").adaptations.rules[${t}].value.localTokens`;if(n===void 0)return;if(typeof n!=`object`||!n||Array.isArray(n))throw Error(`${i} must be a token map.`);let a={};for(let[e,t]of Object.entries(n)){if(!r||!St(r,e))throw Error(`${i}["${e}"] cannot enroll a theme-local token. Declare it in the root theme or an exact enrolled base first.`);a[e]=Ct(t,`${i}["${e}"]`)}return Object.keys(a).length>0?a:void 0}var Dt=[`sm`,`md`,`lg`,`xl`,`2xl`],Ot=Object.freeze({sm:640,md:768,lg:1024,xl:1280,"2xl":1536});function kt(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function At(e,t){if(!kt(e))throw Error(`${t} must be an object.`)}function jt(e,t,n){for(let r of Object.keys(e))if(!t.has(r))throw Error(`${n}.${r} is not supported.`)}function Mt(e){if(Array.isArray(e)){let t=[];for(let n of e)t.push(Mt(n));return t}return kt(e)?Object.fromEntries(Object.entries(e).map(([e,t])=>[e,Mt(t)])):e}var Nt=new Set(Dt),Pt=new Set([`widthBreakpoints`,`rules`]),Ft=new Set([`when`,`value`]),It=new Set([`width`,`pointer`,`contrast`,`motion`]),Lt=new Set([`from`,`below`]),Rt=new Set([`typography`,`color`,`radius`,`motion`,`tokens`,`localTokens`,`components`]);function zt(e,t){At(e,t),jt(e,Nt,t);let n={};for(let r of Dt){let i=e[r];if(i!==void 0){if(typeof i!=`number`||!Number.isFinite(i)||i<=0)throw Error(`${t}.${r} must be a finite positive number of CSS pixels.`);n[r]=i}}return n}function Bt(e,t){let n=zt(e,t);for(let e of Dt)if(n[e]===void 0)throw Error(`${t}.${e} is missing from the effective breakpoint map.`);return n}function Vt(e,t){let n;for(let r of Dt){if(n!==void 0&&e[r]<=e[n])throw Error(`${t} must be strictly increasing: ${r} (${e[r]}) is not above ${n} (${e[n]}).`);n=r}}function Ht(e,t){At(e,t),jt(e,It,t);let n=Object.fromEntries(Object.entries(e).filter(([,e])=>e!==void 0));if(Object.keys(n).length===0)throw Error(`${t} must contain at least one condition.`);let r=Mt(n);if(r.width!==void 0){if(At(r.width,`${t}.width`),jt(r.width,Lt,`${t}.width`),r.width.from===void 0&&r.width.below===void 0)throw Error(`${t}.width must contain \`from\`, \`below\`, or both.`);for(let e of[`from`,`below`]){let n=r.width[e];if(n!==void 0&&(typeof n!=`string`||!Nt.has(n)))throw Error(`${t}.width.${e} must be one of ${Dt.join(`, `)}.`)}}if(r.pointer!==void 0&&r.pointer!==`coarse`&&r.pointer!==`fine`)throw Error(`${t}.pointer must be 'coarse' or 'fine'.`);if(r.contrast!==void 0&&r.contrast!==`more`&&r.contrast!==`less`&&r.contrast!==`no-preference`)throw Error(`${t}.contrast must be 'more', 'less', or 'no-preference'.`);if(r.motion!==void 0&&r.motion!==`reduce`&&r.motion!==`no-preference`)throw Error(`${t}.motion must be 'reduce' or 'no-preference'.`);return r}function Ut(e,t){At(e,t),jt(e,Rt,t);for(let n of Rt){let r=e[n];if(r!==void 0&&!kt(r))throw Error(`${t}.${n} must be an object.`)}return Mt(e)}function Wt(e,t){if(At(e,t),jt(e,Ft,t),!Object.prototype.hasOwnProperty.call(e,`when`))throw Error(`${t}.when is required.`);if(!Object.prototype.hasOwnProperty.call(e,`value`))throw Error(`${t}.value is required.`);return{when:Ht(e.when,`${t}.when`),value:Ut(e.value,`${t}.value`)}}function Gt(e,t){if(!Array.isArray(e))throw Error(`${t} must be an array of {when, value} objects.`);for(let n=0;n<e.length;n++)if(!Object.prototype.hasOwnProperty.call(e,n))throw Error(`${t}[${n}] must be present.`);return e.map((e,n)=>Wt(e,`${t}[${n}]`))}function Kt(e,t,n){let r=`defineTheme("${e}").adaptations`,i={...Ot},a=[];t!==void 0&&(At(t,`defineTheme("${e}").extends.__adaptations`),jt(t,new Set([`widthBreakpoints`,`rules`]),`defineTheme("${e}").extends.__adaptations`),i=Bt(t.widthBreakpoints,`defineTheme("${e}").extends.__adaptations.widthBreakpoints`),a=Gt(t.rules,`defineTheme("${e}").extends.__adaptations.rules`));let o={},s=[];n!==void 0&&(At(n,r),jt(n,Pt,r),n.widthBreakpoints!==void 0&&(o=zt(n.widthBreakpoints,`${r}.widthBreakpoints`)),n.rules!==void 0&&(s=Gt(n.rules,`${r}.rules`)));let c={...i,...o};return Vt(c,`${r}.widthBreakpoints`),{widthBreakpoints:c,rules:[...a,...s]}}function O(e,t){return e?t?{...e,...t,weights:e.weights||t.weights?{...e.weights,...t.weights}:void 0}:e:t}function qt(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw Error(`${t} must be a finite number.`)}function Jt(e,t,n){let r=t.scale!==void 0||[t.body,t.heading,t.code].some(e=>e?.weight!==void 0||e?.weights!==void 0),i=r?e?.scale:void 0;if(t.scale!==void 0){let r=t.scale.base??e?.scale?.base,a=t.scale.ratio??e?.scale?.ratio;if(r===void 0||a===void 0)throw Error(`${n}.scale must supply both \`base\` and \`ratio\` unless the missing field exists on the effective root typography axis.`);qt(r,`${n}.scale.base`),qt(a,`${n}.scale.ratio`),i={base:r,ratio:a}}else if(r&&i===void 0)throw Error(`${n} sets typography weights without an effective root scale. Supply \`scale.base\` and \`scale.ratio\` in the same rule.`);return{...i?{scale:i}:{},body:O(e?.body,t.body),heading:O(e?.heading,t.heading),code:O(e?.code,t.code)}}function Yt(e,t,n){let r=t.base??e?.base,i=t.multiplier??e?.multiplier;if(r===void 0||i===void 0)throw Error(`${n} must supply both \`base\` and \`multiplier\` unless the missing field exists on the effective root radius axis.`);return qt(r,`${n}.base`),qt(i,`${n}.multiplier`),{base:r,multiplier:i}}function Xt(e,t,n){let r=t.fast??e?.fast,i=t.medium??e?.medium,a=t.ratio??e?.ratio;if(r===void 0||i===void 0||a===void 0)throw Error(`${n} must supply \`fast\`, \`medium\`, and \`ratio\` unless each missing field exists on the effective root motion axis.`);qt(r,`${n}.fast`),qt(i,`${n}.medium`),qt(a,`${n}.ratio`);let o=t.slow??e?.slow;o!==void 0&&qt(o,`${n}.slow`);let s=t.easing??e?.easing;return{fast:r,medium:i,ratio:a,...o===void 0?{}:{slow:o},...s===void 0?{}:{easing:s}}}function Zt(e,t,n,r){let i=`defineTheme("${e}").adaptations.rules[${t}].value`;return{typography:n.typography?Jt(r.typography,n.typography,`${i}.typography`):void 0,color:n.color?{...r.color,...n.color}:void 0,radius:n.radius?Yt(r.radius,n.radius,`${i}.radius`):void 0,motion:n.motion?Xt(r.motion,n.motion,`${i}.motion`):void 0,tokens:n.tokens,components:n.components}}function Qt(e,t){if(typeof e==`string`){if(/\b(?:NaN|undefined)\b/.test(e))throw Error(`${t} resolved to the invalid CSS value "${e}".`);return}if(kt(e)){for(let[n,r]of Object.entries(e))Qt(r,`${t}.${n}`);return}throw Error(`${t} must resolve to a concrete CSS string.`)}function $t(e,t,n,r){let i=`defineTheme("${e}").adaptations.rules[${t}].when`,a=[];if(n.width){let e=n.width.from,t=n.width.below;if(e!==void 0&&t!==void 0&&r[e]>=r[t])throw Error(`${i}.width must resolve to \`from < below\`; ${e} is ${r[e]}px and ${t} is ${r[t]}px.`);e!==void 0&&a.push(`(width >= ${r[e]}px)`),t!==void 0&&a.push(`(width < ${r[t]}px)`)}if(n.pointer!==void 0&&a.push(`(pointer: ${n.pointer})`),n.contrast!==void 0&&a.push(`(prefers-contrast: ${n.contrast})`),n.motion!==void 0&&a.push(`(prefers-reduced-motion: ${n.motion})`),a.length===0)throw Error(`${i} must contain at least one concrete condition.`);return a.join(` and `)}var en={pointer:[`coarse`,`fine`,`none`],contrast:[`more`,`less`,`no-preference`,`custom`],motion:[`reduce`,`no-preference`]};function tn(e,t,n){let r=e.width?.from;if(r!==void 0&&n.width<t[r])return!1;let i=e.width?.below;return!(i!==void 0&&n.width>=t[i]||e.pointer!==void 0&&e.pointer!==n.pointer||e.contrast!==void 0&&e.contrast!==n.contrast||e.motion!==void 0&&e.motion!==n.motion)}function k(e,t,n,r,i){let a=i.flatMap((e,t)=>Object.keys(e.tokens).length>0||e.localTokens&&Object.keys(e.localTokens).length>0?[t]:[]);if(a.length===0)return;let o=[0,...Dt.map(e=>t[e])],s=new Set;for(let c of o)for(let o of en.pointer)for(let l of en.contrast)for(let u of en.motion){let d={width:c,pointer:o,contrast:l,motion:u},f=a.filter(e=>tn(i[e].when,t,d));if(f.length===0)continue;let p=f.join(`,`);if(s.has(p))continue;s.add(p);let m={...n,...r},h=new Set;for(let e of f){let t=i[e];Object.assign(m,t.tokens,t.localTokens);for(let e of Object.keys(t.tokens))h.add(e);for(let e of Object.keys(t.localTokens??{}))h.add(e)}let g=f.join(`, `),_=f.map(e=>i[e].query).join(`; `);Tt(m,`defineTheme("${e}").adaptations ${f.length===1?`rule`:`overlapping rules`} [${g}] (${_})`,h)}}function nn(e,t,n){if(!e||!t)return e;let r={};for(let[i,a]of Object.entries(e))for(let[e,o]of Object.entries(a)){let a=Object.fromEntries(Object.entries(o).filter(([r])=>t[i]?.[e]?.[r]===void 0||Object.hasOwn(n?.[i]?.[e]??{},r)));Object.keys(a).length>0&&((r[i]??={})[e]=a)}return Object.keys(r).length>0?r:void 0}function rn(e,t,n,r,i,a){if(t.rules.length===0)return;let o=t.rules.map((r,o)=>{let s=Zt(e,o,r.value,n),c=yt(s,void 0,{strictTokens:!0});for(let[t,n]of Object.entries(c.tokens))Qt(n,`defineTheme("${e}").adaptations.rules[${o}].value.tokens["${t}"]`);c.components&&Qt(c.components,`defineTheme("${e}").adaptations.rules[${o}].value.components`);for(let t of Object.keys(r.value.tokens??{}))if(i&&Object.prototype.hasOwnProperty.call(i,t))throw Error(`defineTheme("${e}").adaptations.rules[${o}].value.tokens["${t}"] matches an enrolled theme-local declaration; write it through value.localTokens instead.`);let l=Et(e,o,r.value.localTokens,i);return{when:r.when,query:$t(e,o,r.when,t.widthBreakpoints),tokens:c.tokens,localTokens:l,components:s.typography?.scale?nn(c.components,a,r.value.components):c.components}});return k(e,t.widthBreakpoints,r,i,o),o}var an={avatar:[{property:`borderRadius`,vars:[`--_avatar-radius`]}],banner:[{property:`borderRadius`,vars:[`--_banner-radius`]}],"bottom-sheet":[{property:`padding`,expand:`container`}],button:[{property:`borderRadius`,vars:[`--_button-radius`]}],card:[{property:`borderRadius`,vars:[`--_card-radius`]},{property:`padding`,expand:`container`}],chat:[{property:`borderRadius`,vars:[`--_chat-composer-radius`]},{property:`padding`,vars:[`--_chat-composer-padding`]}],dialog:[{property:`borderRadius`,vars:[`--_dialog-radius`]},{property:`padding`,expand:`container`}],drawer:[{property:`padding`,expand:`container`}],"context-menu":[{property:`borderRadius`,vars:[`--_dropdown-menu-radius`]},{property:`padding`,vars:[`--_dropdown-menu-padding`]}],"dropdown-menu":[{property:`borderRadius`,vars:[`--_dropdown-menu-radius`]},{property:`padding`,vars:[`--_dropdown-menu-padding`]}],field:[{property:`borderRadius`,vars:[`--_field-radius`]}],"hover-card":[{property:`borderRadius`,vars:[`--_hovercard-radius`]}],item:[{property:`paddingInline`,vars:[`--_item-inset-inline`]}],"number-input":[{property:`padding`,expand:`container`},{property:`borderRadius`,vars:[`--_field-radius`]}],popover:[{property:`borderRadius`,vars:[`--_popover-radius`]}],"progress-bar-mark":[{property:`width`,vars:[`--_progressbar-mark-width`],replaces:!0},{property:`height`,vars:[`--_progressbar-mark-height`],replaces:!0}],section:[{property:`padding`,expand:`container`}],"segmented-control":[{property:`borderRadius`,vars:[`--_segmented-control-radius`]},{property:`padding`,vars:[`--_segmented-control-padding`]}],"text-area":[{property:`paddingInline`,vars:[`--_textarea-inline-padding`],replaces:!0}]};function on(e,t){let n=an[e];return n?n.filter(e=>e.property===t):[]}function sn(e){return e===`
`||e===`\r`||e===`\f`}function cn(e){return e===` `||e===`	`||sn(e)}function ln(e){return/^[0-9a-fA-F]$/.test(e)}function un(e){return/^[A-Za-z_]$/.test(e)||e.charCodeAt(0)>=128}function dn(e){return un(e)||/^[0-9-]$/.test(e)}function fn(e){return e.replace(/\r\n?|\f/g,`
`).replace(/\0/g,`�`)}var pn=/[\x01-\x08\x0b\x0e-\x1f\x7f]/;function mn(e,t){return e[t]===`\\`&&t+1<e.length&&!sn(e[t+1])}function hn(e,t){let n=e[t];if(n===`-`){let n=e[t+1];return n!==void 0&&(n===`-`||un(n)||mn(e,t+1))}return n===`\\`?mn(e,t):n!==void 0&&un(n)}function gn(e,t){if(ln(e[t])){let n=t;for(;n<e.length&&n-t<6&&ln(e[n]);)n++;return n<e.length&&cn(e[n])&&n++,n}return t+((e.codePointAt(t)??0)>65535?2:1)}function _n(e,t){let n=``,r=t;for(;r<e.length;){let t=e[r];if(dn(t))n+=t,r++;else if(mn(e,r)){let t=gn(e,r+1);if(ln(e[r+1])){let i=e.slice(r+1,t).trim(),a=parseInt(i,16);n+=a===0||a>1114111||a>=55296&&a<=57343?`�`:String.fromCodePoint(a)}else n+=e.slice(r+1,t);r=t}else break}return{end:r,name:n}}function vn(e,t){let n=e[t],r=t+1;for(;r<e.length;){let t=e[r];if(t===n)return{end:r+1};if(sn(t))return{reason:`a newline inside a quoted string makes a bad string`};if(t===`\\`){if(r+1>=e.length)break;r=sn(e[r+1])?r+2:gn(e,r+1);continue}r++}return{reason:`an unclosed ${n} string would swallow the rest of the rule`}}function yn(e,t){let n=t;for(;n<e.length;){let t=e[n];if(t===`)`)return{end:n+1};if(cn(t)){for(;n<e.length&&cn(e[n]);)n++;if(n>=e.length)break;return e[n]===`)`?{end:n+1}:{reason:`whitespace inside an unquoted url() makes a bad url`}}if(pn.test(t))return{reason:`a non-printable character inside url() makes a bad url`};if(t===`"`||t===`'`||t===`(`)return{reason:`"${t}" inside an unquoted url() makes a bad url; quote the URL instead`};if(t===`\\`){if(!mn(e,n))return{reason:`a stray backslash inside url() makes a bad url`};n=gn(e,n+1);continue}n++}return{reason:`an unclosed url( would swallow the rest of the rule`}}var bn={"(":`)`,"[":`]`,"{":`}`};function xn(e){let t=fn(e),n=[],r=0;for(;r<t.length;){let e=t[r];if(e===`/`&&t[r+1]===`*`){let e=t.indexOf(`*/`,r+2);if(e===-1)return`an unclosed /* comment would swallow the rest of the rule`;r=e+2;continue}if(e===`"`||e===`'`){let e=vn(t,r);if(`reason`in e)return e.reason;r=e.end;continue}if(hn(t,r)){let e=_n(t,r);if(r=e.end,t[r]===`(`&&e.name.toLowerCase()===`url`){let e=r+1;for(;e<t.length&&cn(t[e]);)e++;if(t[e]===`"`||t[e]===`'`){n.push(`)`),r+=1;continue}let i=yn(t,e);if(`reason`in i)return i.reason;r=i.end}continue}if(e===`\\`){if(r+1>=t.length)return`a trailing backslash would escape the declaration terminator`;r=sn(t[r+1])?r+1:gn(t,r+1);continue}if(e===`(`||e===`[`||e===`{`){if(e===`{`&&n.length===0)return`an unquoted "{" would open a nested block`;n.push(bn[e]),r++;continue}if(e===`)`||e===`]`||e===`}`){if(n.length===0)return e===`}`?`an unquoted "}" would close the rule`:`an unbalanced "${e}" has nothing to close`;let t=n.pop();if(t!==e)return`"${e}" does not close the open "${t===`)`?`(`:t===`]`?`[`:`{`}"`;r++;continue}if(e===`;`&&n.length===0)return`an unquoted ";" would end the declaration`;r++}if(n.length>0){let e=n[n.length-1];return`an unclosed "${e===`)`?`(`:e===`]`?`[`:`{`}" would swallow the rest of the rule`}return null}function Sn(e){let t=fn(e);if(t.length===0||!hn(t,0))return`a property name must be a CSS identifier`;let{end:n}=_n(t,0);return n===t.length?null:`a property name must be one CSS identifier; "${t[n]}" cannot appear in one`}function Cn(e){return`[data-${s}-theme="${e}"]`}var wn=`[data-${s}-theme]`;function Tn(e){return`[data-${s}-media="${e}"]`}function En(e,t){return`.${l}-${e}${t}`}function Dn(e,t,n){let r=En(e,t);return n?`:is(${Tn(n)}) :is(${r})`:r}var On=`:where(:not(:disabled,[aria-disabled="true"]))`;function kn(e,t){let n=[],r=0,i=0;for(let t=0;t<e.length;t++){let a=e[t];a===`(`?r++:a===`)`?r=Math.max(0,r-1):a===`,`&&r===0&&(n.push(e.slice(i,t).trim()),i=t+1)}n.push(e.slice(i).trim());let a=An(t);return n.map(e=>`${e}${a}`).join(`, `)}function An(e){if(!/^:hover(?![-\w])/.test(e)||e.includes(`[aria-disabled`))return e;let t=e.indexOf(`::`);return t===-1?e+On:e.slice(0,t)+On+e.slice(t)}function jn(e){return e.replace(/[A-Z]/g,e=>`-${e.toLowerCase()}`)}var Mn=new Set([`padding`,`paddingBlock`,`paddingInline`,`paddingBlockStart`,`paddingBlockEnd`,`paddingInlineStart`,`paddingInlineEnd`]),Nn={paddingTop:`paddingBlockStart`,paddingBottom:`paddingBlockEnd`},Pn=new Set([...Mn,...Object.keys(Nn)]);function Fn(e){let t={};for(let[n,r]of e)switch(Nn[n]??n){case`padding`:{let e=r.trim().split(/\s+/);e.length===1?(t.blockStart=e[0],t.blockEnd=e[0],t.inline=e[0]):e.length===2?(t.blockStart=e[0],t.blockEnd=e[0],t.inline=e[1]):e.length>=3&&(t.blockStart=e[0],t.inline=e[1],t.blockEnd=e[2]);break}case`paddingBlock`:{let e=r.trim().split(/\s+/);t.blockStart=e[0],t.blockEnd=e[1]??e[0];break}case`paddingInline`:{let e=r.trim().split(/\s+/);e.length===1?t.inline=e[0]:(t.inlineStart=e[0],t.inlineEnd=e[1]);break}case`paddingBlockStart`:t.blockStart=r;break;case`paddingBlockEnd`:t.blockEnd=r;break;case`paddingInlineStart`:t.inlineStart=r;break;case`paddingInlineEnd`:t.inlineEnd=r}return t}function In(e,t,n=!1){let r=o(`${e}-padding`),i=[],a=t.inlineStart??t.inline,s=t.inlineEnd??t.inline;if(a!=null&&s!=null&&a===s&&t.blockStart!=null&&t.blockEnd!=null&&a===t.blockStart&&t.blockStart===t.blockEnd?i.push([r,a??``]):(t.inlineStart!=null||t.inlineEnd!=null?(a!=null&&i.push([`${r}-inline-start`,a]),s!=null&&i.push([`${r}-inline-end`,s])):t.inline!=null&&i.push([`${r}-inline`,t.inline]),t.blockStart!=null&&i.push([`${r}-block-start`,t.blockStart]),t.blockEnd!=null&&i.push([`${r}-block-end`,t.blockEnd])),n){let e=new Set(i.map(([e])=>e)),t=e.has(r)?[`${r}-inline`,`${r}-inline-start`,`${r}-inline-end`,`${r}-block-start`,`${r}-block-end`]:e.has(`${r}-inline`)?[`${r}-inline-start`,`${r}-inline-end`]:[];for(let n of t)e.has(n)||i.push([n,`initial`])}return i}var Ln=e=>{console.warn(`[astryx theme] ${e}`)};function Rn(e){return e?t=>e.push(t):Ln}function zn(e){let t=JSON.stringify(e);return t.length>80?`${t.slice(0,77)}..."`:t}function Bn(e,t,n,r,i){e(`dropped "${t}" in ${r}: ${i} (value: ${zn(n)})`)}function Vn(e,t,n,r){let i=Sn(e)??xn(t);return i===null||(Bn(r,e,t,n,i),!1)}function Hn(e,t,n,r=e=>e){let i=[];for(let[a,o]of e){let e=`${o}`,s=r(a);Vn(s,e,t,n)&&i.push(`    ${s}: ${e};`)}return i.join(`
`)}function Un(e,t,n){let r=[Hn(Object.entries(e.tokens),`${t}tokens`,n),Hn(Object.entries(e.localTokens??{}),`${t}localTokens`,n)].filter(e=>e.length>0).join(`
`);return Object.keys(e.tokens).length>0||Object.keys(e.localTokens??{}).length>0?`  :scope {\n${r}\n  }`:null}function Wn(e,t,n){return`${e}components.${t}[${JSON.stringify(n)}]`}function Gn(e,t){let n=[],r=Rn(t),i=e=>`var(${e})`,a=Un(e,``,r);return a!==null&&n.push(a),e.tokens[`--font-family-body`]&&n.push(`  :scope {
    font-family: var(--font-family-body);
  }`),e.components&&Yn(e.components,n,{sink:r}),Xn(i,n),Zn(e.components||{},n),$n(e.components||{},n),qn(e.components||{},n),n}var Kn={normal:`var(--font-weight-normal)`,medium:`var(--font-weight-medium)`,semibold:`var(--font-weight-semibold)`,bold:`var(--font-weight-bold)`};function qn(e,t,{surface:n,inheritedComponents:r,forceAll:i=!1,authoredOnly:a=!1}={}){let o=e.heading,s=o&&typeof o==`object`&&!Array.isArray(o)?o:void 0;if(!s&&!i)return;let c=r?.heading,l=c&&typeof c==`object`?c:void 0;for(let[e,r]of Object.entries(Kn)){let i=`weight:${e}`,o=s?Jn(s,i):void 0;if(a&&o===void 0)continue;let c=l?Jn(l,i):void 0,u=o??c??r,d=Dn(`heading`,he(i),n);t.push(`  ${d} { font-weight: ${u}; }`)}}function Jn(e,t){let n=e[t];if(typeof n!=`object`||!n||Array.isArray(n))return;let r=n.fontWeight;return typeof r==`string`&&xn(r)===null?r:void 0}function Yn(e,t,{resetInheritedPaddingSpecificity:n=!1,surface:r,sink:i=Ln,location:a=``}={}){for(let[o,s]of Object.entries(e))for(let[e,c]of Object.entries(s)){let s=Object.entries(c);if(s.length===0)continue;let l=Dn(o,he(e),r),u=Wn(a,o,e),d=[],f=[];for(let[e,t]of s)e.startsWith(`:`)&&typeof t==`object`?f.push([e,t]):Vn(jn(e),`${t}`,u,i)&&d.push([e,t]);let p=d,m=[],h=!1,g=new Set;for(let[e,t]of d){let n=on(o,e),r=Mn.has(e)&&e!==`padding`?on(o,`padding`):[];for(let i of[...n,...r])if(i.expand===`container`&&Mn.has(e)&&(h=!0),i.replaces&&g.add(e),i.vars)for(let e of i.vars)m.push([e,t]);e in Nn&&on(o,`padding`).some(e=>e.expand===`container`)&&(h=!0)}if(h){let e=d.filter(([e])=>Pn.has(e)),t=d.filter(([e])=>!Pn.has(e)),r=In(o,Fn(e),n);p=[...t,...r]}if(g.size>0&&(p=p.filter(([e])=>!g.has(e))),m.length>0&&(p=[...p,...m]),p.length>0){let e=Hn(p,u,i,jn);t.push(`  ${l} {\n${e}\n  }`)}for(let[e,n]of f){let r=Object.entries(n);if(r.length>0){let n=Hn(r,`${u}[${JSON.stringify(e)}]`,i,jn);t.push(`  ${kn(l,e)} {\n${n}\n  }`)}}}}function Xn(e,t){t.push(`  :where(h1, h2, h3, h4, h5, h6) {
    font-family: var(--font-family-heading);
    color: var(--color-text-primary);
  }`);for(let n=1;n<=6;n++)t.push(`  :where(h${n}) {
    font-size: ${e(`--text-heading-${n}-size`)};
    font-weight: ${e(`--text-heading-${n}-weight`)};
    line-height: ${e(`--text-heading-${n}-leading`)};
  }`);t.push(`  :where(p) {
    font-family: var(--font-family-body);
    font-size: ${e(`--text-body-size`)};
    font-weight: ${e(`--text-body-weight`)};
    line-height: ${e(`--text-body-leading`)};
    color: var(--color-text-primary);
  }`),t.push(`  :where(small) {
    font-size: ${e(`--text-supporting-size`)};
    font-weight: ${e(`--text-supporting-weight`)};
    line-height: ${e(`--text-supporting-leading`)};
    color: var(--color-text-secondary);
  }`),t.push(`  :where(code, pre) {
    font-family: var(--font-family-code);
    font-size: ${e(`--text-code-size`)};
    line-height: ${e(`--text-code-leading`)};
  }`),t.push(`  :where(hr) {
    border: none;
    border-top: 1px solid var(--color-border);
  }`)}function Zn(e,t,n){let r={primary:`var(--color-text-primary)`,secondary:`var(--color-text-secondary)`,disabled:`var(--color-text-disabled)`,placeholder:`var(--color-text-secondary)`,accent:`var(--color-text-accent)`},i=`text`in e,a=`heading`in e,o=`link`in e;if(i||a||o)for(let[e,s]of Object.entries(r))i&&t.push(`  ${Dn(`text`,he(`color:${e}`),n)} { color: ${s}; }`),a&&t.push(`  ${Dn(`heading`,he(`color:${e}`),n)} { color: ${s}; }`),o&&t.push(`  ${Dn(`link`,he(`color:${e}`),n)} { color: ${s}; }`)}var Qn={"4xs":`var(--font-size-4xs)`,"3xs":`var(--font-size-3xs)`,"2xs":`var(--font-size-2xs)`,xsm:`var(--font-size-xs)`,sm:`var(--font-size-sm)`,base:`var(--font-size-base)`,lg:`var(--font-size-lg)`,xl:`var(--font-size-xl)`,"2xl":`var(--font-size-2xl)`,"3xl":`var(--font-size-3xl)`,"4xl":`var(--font-size-4xl)`};function $n(e,t,n){if(`text`in e)for(let[e,r]of Object.entries(Qn)){let i=he(`size:${e}`);t.push(`  ${Dn(`text`,i,n)} { font-size: ${r}; }`)}}function er(e,t){let n=Gn(e,t),r=[],i=[];for(let e of n)e.trimStart().startsWith(`:where(`)?r.push(e):i.push(e);return{component:i,prose:r}}function tr(e,t){let n=[],r=Cn(e.name),i=Rn(t);for(let t of[`dark`,`light`]){let r=t===`dark`?e.__onDark:e.__onLight;if(!r)continue;let a=t===`dark`?`onDark.`:`onLight.`,o=Object.entries(r.tokens);if(o.length>0){let e=Hn(o,`${a}tokens`,i);n.push(`  ${Tn(t)} {\n${e}\n  }`)}r.components&&(Yn(r.components,n,{resetInheritedPaddingSpecificity:!0,surface:t,sink:i,location:a}),Zn(r.components,n,t),$n(r.components,n,t),qn(r.components,n,{surface:t,inheritedComponents:e.components}))}return n.length===0?``:`@scope (${r}) to (${wn}) {\n${n.join(`

`)}\n}`}function nr(e,t,n){let r=[],i=Un(e,t,n);return i!==null&&r.push(i),e.components&&(Yn(e.components,r,{resetInheritedPaddingSpecificity:!0,sink:n,location:t}),Zn(e.components,r),$n(e.components,r)),r}function rr(e){let t=e?.heading;return typeof t==`object`&&!!t&&!Array.isArray(t)}function ir(e,t,n){return`@media ${e} {\n  @scope (${n}) to (${wn}) {\n${t.map(or).join(`

`)}\n  }\n}`}function ar(e,t){let n=Rn(t),r=e.__adaptations===void 0?void 0:Kt(e.name,e.__adaptations,void 0),i=e.__adaptationRules??(r?rn(e.name,r,e.__axes??{},e.tokens,e.localTokens,e.components):void 0);if(!i||i.length===0)return{prose:``,component:``};let a=Cn(e.name),o=[],s=i.some(e=>rr(e.components));for(let[e,t]of i.entries()){let r=nr(t,`adaptations[${e}].`,n);r.length!==0&&o.push(ir(t.query,r,a))}if(s){let t=[];qn(e.components??{},t,{forceAll:!0}),o.push(`@scope (${a}) to (${wn}) {\n${t.join(`

`)}\n}`);for(let e of i){let t=[];qn(e.components??{},t,{authoredOnly:!0}),t.length>0&&o.push(ir(e.query,t,a))}}return{prose:``,component:o.join(`

`)}}function or(e){return`  ${e}`}function sr(e,t){let{component:n,prose:r}=er(e,t),i=Cn(e.name),a=wn,o=``;r.length>0&&(o=`@scope (${i}) to (${a}) {\n${r.join(`

`)}\n}`);let s=``;n.length>0&&(s=`@scope (${i}) to (${a}) {\n${n.join(`

`)}\n}`);let c=ar(e,t);c.component&&(s=s?`${s}\n\n${c.component}`:c.component);let l=tr(e,t);return l&&(s=s?`${s}\n\n${l}`:l),{prose:o,component:s}}var cr={base:{k1xSpc:`xjp7ctv`,kMwMTN:`x1tgivj0`,kMv6JI:`x9ynric`,$$css:!0},light:{kQNsl9:`x19aimcq`,$$css:!0},dark:{kQNsl9:`xntwwlm`,$$css:!0},system:{kQNsl9:`x108lcm5`,$$css:!0}},lr=S.createContext(!1);lr.displayName=`ThemeNestingContext`;var ur=new Set,dr=`@layer reset, astryx-base, astryx-theme;`;function fr(e){let t=(0,S.useId)();(0,S.useInsertionEffect)(()=>{if(e.__built)return;let n=`astryx-theme-${e.name}`;if(ur.has(n))return;`${e.name}`,`${e.name}${e.name}${e.name}${e.name}`;let{prose:r,component:i}=sr(e);ur.add(n);let o=[()=>ur.delete(n)];if(r){let n=document.createElement(`style`);n.setAttribute(a(`theme-prose`),e.name),n.setAttribute(a(`id`),t),n.textContent=`${dr}\n@layer reset {\n${r}\n}`,document.head.appendChild(n)}if(i){let n=document.createElement(`style`);n.setAttribute(a(`theme`),e.name),n.setAttribute(a(`id`),t),n.textContent=`${dr}\n@layer astryx-theme {\n${i}\n}`,document.head.appendChild(n)}return(r||i)&&o.push(()=>{let n=document.querySelector(`style[${a(`theme-prose`)}="${e.name}"][${a(`id`)}="${t}"]`),r=document.querySelector(`style[${a(`theme`)}="${e.name}"][${a(`id`)}="${t}"]`);n?.remove(),r?.remove()}),()=>{for(let e of o)e()}},[e,t])}function pr(e,t,n){d(()=>{if(!e&&!(typeof document>`u`))return t===`light`||t===`dark`?document.documentElement.setAttribute(`data-theme`,t):document.documentElement.removeAttribute(`data-theme`),document.documentElement.setAttribute(a(`theme`),n),()=>{document.documentElement.removeAttribute(`data-theme`),document.documentElement.removeAttribute(a(`theme`))}},[e,t,n])}function mr({theme:e,mode:t=`system`,children:n}){let r=(0,S.use)(lr);g(e),fr(e),pr(r,t,e.name);let i=t===`dark`?cr.dark:t===`light`?cr.light:cr.system,a=(0,S.useMemo)(()=>({theme:e,mode:t}),[e,t]);return(0,C.jsx)(f,{value:a,children:(0,C.jsx)(lr,{value:!0,children:(0,C.jsx)(`div`,{...c(cr.base,i),"data-astryx-theme":e.name,"data-theme":t===`system`?void 0:t,children:n})})})}mr.displayName=`Theme`;function hr(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Malformed settings response.`);return e}function gr(e){if(typeof e!=`string`||!/^[0-9a-f]{64}$/.test(e))throw Error(`Malformed settings fingerprint.`);return e}function _r(e){let t=hr(e);if(!Array.isArray(t.schema)||t.schema.length>256)throw Error(`Malformed settings schema.`);let n=new Set;return{schema:t.schema.map(e=>{let t=hr(e);if(typeof t.name!=`string`||n.has(t.name)||typeof t.type!=`string`||![`bool`,`int`,`int_or_null`,`float`,`str`,`str_or_null`,`array`,`object`,`object_or_null`].includes(t.type)||typeof t.mutable!=`boolean`||typeof t.help!=`string`||!(t.allowed===null||Array.isArray(t.allowed)&&t.allowed.every(e=>typeof e==`string`))||t.reason!==void 0&&typeof t.reason!=`string`||!Object.hasOwn(t,`default`))throw Error(`Malformed setting specification.`);return n.add(t.name),t}),current:hr(t.current),fingerprint:gr(t.fingerprint)}}function vr(e){let t=hr(e);if(!Array.isArray(t.rejected)||!t.rejected.every(e=>{let t=hr(e);return typeof t.name==`string`&&typeof t.reason==`string`}))throw Error(`Malformed settings rejection list.`);return{applied:hr(t.applied),rejected:t.rejected,current:hr(t.current),fingerprint:gr(t.fingerprint)}}function yr(e){let t=hr(e),n=hr(t.default_generation_settings),r=e=>typeof e==`number`&&Number.isSafeInteger(e)&&e>0?e:null;return{nCtx:r(n.n_ctx),kvCacheMode:typeof t.kv_cache_mode==`string`&&t.kv_cache_mode.length<=64&&/^[a-z0-9+_-]+$/.test(t.kv_cache_mode)?t.kv_cache_mode:null,totalSlots:r(t.total_slots),geometry:t.geometry===void 0?null:hr(t.geometry)}}function br(e,t){if(!e.mutable)throw Error(e.reason??`Restart required.`);let n=t;if(![`str`,`str_or_null`].includes(e.type)||e.type===`str_or_null`&&t===`null`)try{n=JSON.parse(t)}catch{throw Error(`Enter a valid ${e.type} value.`)}let r=e.type.endsWith(`_or_null`),i=e.type.replace(`_or_null`,``);if(!(r&&n===null)&&(i===`bool`?typeof n!=`boolean`:i===`int`?typeof n!=`number`||!Number.isSafeInteger(n):i===`float`?typeof n!=`number`||!Number.isFinite(n):i===`str`?typeof n!=`string`:i===`array`?!Array.isArray(n):typeof n!=`object`||!n||Array.isArray(n)))throw Error(`Expected ${e.type}.`);if(e.allowed!==null&&(typeof n!=`string`||!e.allowed.includes(n)))throw Error(`Allowed values: ${e.allowed.join(`, `)}.`);return n}function xr(e,t){return typeof t==`string`&&[`str`,`str_or_null`].includes(e.type)?t:JSON.stringify(t)??``}var Sr=7;function Cr(e){if(!Number.isFinite(e))return String(e);let t=Math.fround(e);for(let e=1;e<Sr;e+=1){let n=Number(t.toPrecision(e));if(Math.fround(n)===t)return String(n)}return String(Number(t.toPrecision(Sr)))}function wr(e,t){if(t==null)return``;let n=e.type.replace(`_or_null`,``);return n===`float`&&typeof t==`number`?Cr(t):n===`int`&&typeof t==`number`?String(t):xr(e,t)}function Tr(e){let t=hr(e);if(!Array.isArray(t.tokens)||!t.tokens.every(e=>typeof e==`number`&&Number.isInteger(e)))throw Error(`Malformed tokenization response.`);return t.tokens.length}var Er=1048576,Dr=class{decoder=new TextDecoder(`utf-8`);maxFrameBytes;onMessage;onDone;buffered=``;eventName=``;dataLines=[];id=null;retry=null;frameBytes=0;ended=!1;get done(){return this.ended}constructor(e){this.maxFrameBytes=e.maxFrameBytes??Er,this.onMessage=e.onMessage,this.onDone=e.onDone}push(e){if(!this.ended){if(this.frameBytes+=e.byteLength,this.frameBytes>this.maxFrameBytes)throw Error(`SSE frame exceeded the configured byte limit.`);this.buffered+=this.decoder.decode(e,{stream:!0}),this.drainLines(!1)}}close(){if(this.ended)return;let e=this.decoder.decode();e.length>0&&(this.buffered+=e),this.drainLines(!0),this.buffered.length>0&&(this.consumeLine(this.buffered),this.buffered=``),this.dispatch()}drainLines(e){for(;this.buffered.length>0;){let t=this.buffered.indexOf(`
`),n=this.buffered.indexOf(`\r`),r;if(r=t===-1?n:n===-1?t:Math.min(t,n),r===-1||this.buffered[r]===`\r`&&this.buffered[r+1]===void 0&&!e)break;let i=this.buffered.slice(0,r),a=this.buffered[r]===`\r`&&this.buffered[r+1]===`
`?r+2:r+1;this.buffered=this.buffered.slice(a),this.consumeLine(i)}e&&this.buffered.length===0&&this.dispatch()}consumeLine(e){if(e.length===0){this.dispatch();return}if(e.startsWith(`:`))return;let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?``:e.slice(t+1),i=r.startsWith(` `)?r.slice(1):r;n===`event`?this.eventName=i:n===`data`?this.dataLines.push(i):n===`id`?this.id=i:n===`retry`&&/^\d+$/.test(i)&&(this.retry=Number(i))}dispatch(){if(this.dataLines.length===0){this.eventName=``,this.retry=null,this.frameBytes=0;return}let e=this.dataLines.join(`
`),t=this.eventName.length===0?`message`:this.eventName;if(this.eventName=``,this.dataLines=[],this.frameBytes=0,e===`[DONE]`){this.ended=!0,this.onDone?.();return}this.onMessage({event:t,data:e,id:this.id,retry:this.retry}),this.retry=null}};function Or(e){return Array.from(e).some(e=>{let t=e.charCodeAt(0);return t<=32||t>=127&&t<=159})}function kr(e){if(/[\\?#]/.test(e)||Or(e)||e.includes(`//`))throw Error(`WebUI API base must not contain a query, hash, backslash, whitespace, controls or empty path segments.`);for(let t of e.split(`/`)){let e;try{e=decodeURIComponent(t)}catch{throw Error(`WebUI API base contains an invalid path escape.`)}if(e===`.`||e===`..`)throw Error(`WebUI API base must not contain dot path segments.`);if(/[/\\%?#]/.test(e)||Or(e))throw Error(`WebUI API base contains an ambiguous encoded path segment.`)}}function Ar(e){let t=e??Mr();if(t===``)return``;if(!t.startsWith(`/`)||t.startsWith(`//`))throw Error(`WebUI API base must be a same-origin absolute path.`);return kr(t),new URL(t,`http://webui.invalid`).pathname.replace(/\/$/,``)}function jr(e){return/^(.*)\/webui(?:\/(?:index\.html)?)?$/.exec(e)?.[1]??null}function Mr(){if(typeof document>`u`)return``;let e=document.querySelector(`meta[name="mlxcel-ui-api-base"]`)?.content;if(e!==void 0&&e.length>0)return Ar(e);let t=document.querySelector(`base`)?.getAttribute(`href`);if(t!=null&&t.length>0){let e=t.replace(/^[a-z][a-z\d+.-]*:\/\/[^/]*/i,``);kr(e===`.`?``:e.replace(/^\.\//,``));let n=new URL(t,document.URL);if(n.origin!==window.location.origin||n.username||n.password||n.search||n.hash)throw Error(`WebUI document base must stay on the current origin without credentials, query or hash.`);return Ar(jr(n.pathname)??n.pathname)}let n=window.location.pathname,r=jr(n);return r===null?``:(kr(n),Ar(r))}function Nr(e,t,n){if(!t.startsWith(`/ui-api/v1/`)&&t!==`/v1/chat/completions`&&t!==`/v1/responses`&&t!==`/settings`&&t!==`/props`&&t!==`/tokenize`)throw Error(`WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.`);let r=new URLSearchParams;for(let[e,t]of Object.entries(n??{}))t!=null&&r.set(e,String(t));return`${e}${t}${r.size===0?``:`?${r.toString()}`}`}function Pr(e){if(e.length===0)throw Error(`Opaque path segment must not be empty.`);return encodeURIComponent(e)}var Fr=`{
  "openapi": "3.1.0",
  "info": {
    "title": "mlxcel WebUI local control API",
    "version": "1.0.0",
    "description": "Contract gate for epic #1834. This file is JSON-compatible YAML by policy so the validator has no external dependency."
  },
  "servers": [
    {
      "url": "{api_prefix}",
      "variables": {
        "api_prefix": {
          "default": "",
          "description": "Validated server API prefix; paths below include /ui-api/v1 and the prefix is never derived from browser input."
        }
      }
    }
  ],
  "security": [
    {
      "bearerAuth": []
    }
  ],
  "paths": {
    "/ui-api/v1/bootstrap": {
      "get": {
        "summary": "Return server identity, enabled features, redacted roots and hard limits without model initialization.",
        "responses": {
          "200": {
            "description": "Bootstrap response",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/BootstrapResponse"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/bootstrap.model-free.json"
                  }
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/catalog": {
      "get": {
        "summary": "List the deterministic model inventory. Query: limit 1..200 default 50, cursor, q, source, task, lifecycle, support, completeness. No downloads or loads.",
        "responses": {
          "200": {
            "description": "Catalog page",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CatalogListResponse"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/catalog.page.json"
                  }
                }
              }
            }
          },
          "400": {
            "description": "Invalid filter or cursor",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "minimum": 1,
              "maximum": 200,
              "default": 50
            },
            "description": "Page size. Values above 200 are rejected rather than clamped."
          },
          {
            "name": "cursor",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/CursorToken"
            },
            "description": "Opaque cursor returned by the previous page."
          },
          {
            "name": "q",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "maxLength": 128
            },
            "description": "Case-insensitive display/inference-id substring filter."
          },
          {
            "name": "source",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/CatalogSourceKind"
            },
            "description": "Filter by source kind."
          },
          {
            "name": "task",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/TaskKind"
            },
            "description": "Filter by declared task capability."
          },
          {
            "name": "lifecycle",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/ModelLifecycleState"
            },
            "description": "Filter by inference lifecycle state."
          },
          {
            "name": "support",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean"
            },
            "description": "true keeps supported+runnable entries, false keeps unsupported entries."
          },
          {
            "name": "completeness",
            "in": "query",
            "required": false,
            "schema": {
              "type": "boolean"
            },
            "description": "true keeps complete checkpoints, false keeps incomplete entries."
          }
        ]
      }
    },
    "/ui-api/v1/catalog/{id}": {
      "get": {
        "summary": "Read one catalog entry by opaque model id.",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "schema": {
              "$ref": "#/components/schemas/ModelId"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Catalog entry",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CatalogEntry"
                }
              }
            }
          },
          "404": {
            "description": "Unknown model id",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/catalog/refresh": {
      "post": {
        "summary": "Start a bounded background rescan; GET never mutates.",
        "responses": {
          "202": {
            "description": "Accepted",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationAccepted"
                }
              }
            }
          },
          "429": {
            "description": "Operation limit",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "422": {
            "description": "Catalog refresh is unsupported in read-only single-model mode",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/model-actions": {
      "post": {
        "summary": "Load or unload through the shared lifecycle coordinator. Duplicate idempotency keys replay the original operation inside this server instance.",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/ModelActionRequest"
              },
              "examples": {
                "fixture": {
                  "externalValue": "../../tests/fixtures/webui/examples/request.model-action.load.json"
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Accepted",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationAccepted"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/operation.accepted.json"
                  }
                }
              }
            }
          },
          "409": {
            "description": "Stale revision or conflicting action",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "422": {
            "description": "Unsupported action/profile",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "413": {
            "description": "JSON request body exceeds the WebUI contract limit",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/downloads": {
      "post": {
        "summary": "Download a public HuggingFace repository into the configured cache after explicit consent.",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/DownloadRequest"
              },
              "examples": {
                "fixture": {
                  "externalValue": "../../tests/fixtures/webui/examples/request.download.json"
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Accepted",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationAccepted"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/operation.accepted.json"
                  }
                }
              }
            }
          },
          "422": {
            "description": "Invalid or unsupported repository",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "413": {
            "description": "JSON request body exceeds the WebUI contract limit",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/model-removals": {
      "post": {
        "summary": "Remove a cache-owned model only after server checks and UI confirmation.",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/RemovalRequest"
              },
              "examples": {
                "fixture": {
                  "externalValue": "../../tests/fixtures/webui/examples/request.removal.json"
                }
              }
            }
          }
        },
        "responses": {
          "202": {
            "description": "Accepted",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationAccepted"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/operation.accepted.json"
                  }
                }
              }
            }
          },
          "409": {
            "description": "Busy/loading/downloading/draining model",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "422": {
            "description": "Non-cache model cannot be removed",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "413": {
            "description": "JSON request body exceeds the WebUI contract limit",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/operations": {
      "get": {
        "summary": "List active operations plus the bounded terminal history. Query: limit default 50 max 200, cursor, state, kind, target.",
        "responses": {
          "200": {
            "description": "Operations page",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationsListResponse"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/operations.list.json"
                  }
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "limit",
            "in": "query",
            "required": false,
            "schema": {
              "type": "integer",
              "minimum": 1,
              "maximum": 200,
              "default": 50
            },
            "description": "Page size for active plus terminal operation records."
          },
          {
            "name": "cursor",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/CursorToken"
            },
            "description": "Opaque cursor returned by the previous page."
          },
          {
            "name": "state",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/OperationState"
            },
            "description": "Filter by operation state."
          },
          {
            "name": "kind",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/OperationKind"
            },
            "description": "Filter by operation kind."
          },
          {
            "name": "target",
            "in": "query",
            "required": false,
            "schema": {
              "type": "string",
              "maxLength": 128,
              "pattern": "^[A-Za-z0-9._~-]{1,128}(?![\\\\s\\\\S])"
            },
            "description": "Opaque model id or operation target token; never a filesystem path."
          }
        ]
      }
    },
    "/ui-api/v1/operations/{id}": {
      "get": {
        "summary": "Read one operation by id.",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "schema": {
              "$ref": "#/components/schemas/OperationId"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Operation",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/Operation"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/operation.running.json"
                  }
                }
              }
            }
          },
          "404": {
            "description": "Unknown operation",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/operations/{id}/cancel": {
      "post": {
        "summary": "Request cancellation; completion is reported only after the worker stops or cancellation is rejected.",
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "schema": {
              "$ref": "#/components/schemas/OperationId"
            }
          }
        ],
        "responses": {
          "202": {
            "description": "Cancellation accepted",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/OperationAccepted"
                }
              }
            }
          },
          "422": {
            "description": "Cancellation unsupported",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    },
    "/ui-api/v1/events": {
      "get": {
        "summary": "Authenticated SSE stream. Clients normally reconnect with the paired server_instance_id and after_sequence query cursor computed from the minimum authoritative resource fence. Last-Event-ID remains supported only as an opaque legacy replay cursor and must not be sent together with the paired query cursor. Ring gaps or changed server_instance_id produce a reset event and full resnapshot.",
        "responses": {
          "200": {
            "description": "SSE event stream carrying UiEvent JSON payloads"
          },
          "409": {
            "description": "Replay gap or server restart",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        },
        "parameters": [
          {
            "name": "Last-Event-ID",
            "in": "header",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/EventId"
            },
            "description": "Opaque event id from the last applied SSE event. It is a legacy cursor and is rejected when either paired replay query parameter is present."
          },
          {
            "name": "server_instance_id",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/ServerInstanceId"
            },
            "description": "Server instance that produced after_sequence. Required exactly when after_sequence is supplied; a different instance returns a typed server_restart event."
          },
          {
            "name": "after_sequence",
            "in": "query",
            "required": false,
            "schema": {
              "$ref": "#/components/schemas/EventSequence"
            },
            "description": "Replay every retained event with sequence greater than this value. Required exactly when server_instance_id is supplied; 0 and the current sequence are valid. Non-safe integers, future sequences, duplicate parameters and malformed values are rejected."
          }
        ]
      }
    },
    "/ui-api/v1/runtime": {
      "get": {
        "summary": "Model-scoped observation snapshot from existing settings/slot/cache/worker counters. Query: model_id required. No autoload and no sampling work.",
        "parameters": [
          {
            "name": "model_id",
            "in": "query",
            "required": true,
            "schema": {
              "$ref": "#/components/schemas/ModelId"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Runtime snapshot",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/RuntimeSnapshot"
                },
                "examples": {
                  "fixture": {
                    "externalValue": "../../tests/fixtures/webui/examples/runtime.snapshot.json"
                  }
                }
              }
            }
          },
          "404": {
            "description": "Unknown model id",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "401": {
            "description": "Authentication required",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          },
          "403": {
            "description": "Forbidden by browser security checks",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorEnvelope"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "securitySchemes": {
      "bearerAuth": {
        "type": "http",
        "scheme": "bearer"
      }
    },
    "schemas": {
      "SchemaVersion": {
        "type": "string",
        "const": "webui.ui-api.v1"
      },
      "ServerMode": {
        "type": "string",
        "enum": [
          "model_free",
          "single_model",
          "router_pool"
        ]
      },
      "FeatureFlag": {
        "type": "string",
        "enum": [
          "webui",
          "catalog",
          "download",
          "cache_delete",
          "load",
          "unload",
          "chat",
          "runtime",
          "settings",
          "local_history",
          "image_input"
        ]
      },
      "CapabilityPhase": {
        "type": "string",
        "enum": [
          "pre_load",
          "provider_ready"
        ]
      },
      "ActionState": {
        "type": "string",
        "enum": [
          "enabled",
          "disabled",
          "read_only"
        ]
      },
      "ModelLifecycleState": {
        "type": "string",
        "enum": [
          "unloaded",
          "loading",
          "ready",
          "draining",
          "unloading",
          "failed"
        ]
      },
      "DownloadState": {
        "type": "string",
        "enum": [
          "absent",
          "downloading",
          "complete",
          "incomplete",
          "failed"
        ]
      },
      "OperationState": {
        "type": "string",
        "enum": [
          "queued",
          "running",
          "cancelling",
          "succeeded",
          "failed",
          "cancelled"
        ]
      },
      "OperationKind": {
        "type": "string",
        "enum": [
          "catalog_refresh",
          "model_load",
          "model_unload",
          "download",
          "model_removal",
          "settings_patch"
        ]
      },
      "CatalogSourceKind": {
        "type": "string",
        "enum": [
          "cache",
          "models_dir",
          "preset",
          "single_model"
        ]
      },
      "TaskKind": {
        "type": "string",
        "enum": [
          "chat",
          "completion",
          "embedding",
          "rerank",
          "audio_transcription",
          "audio_speech",
          "vision_input",
          "image_generation"
        ]
      },
      "UiEventType": {
        "type": "string",
        "enum": [
          "snapshot",
          "model_revision",
          "operation",
          "download_progress",
          "runtime",
          "settings",
          "reset",
          "gap",
          "server_restart",
          "heartbeat"
        ]
      },
      "ServerInstanceId": {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "pattern": "^[A-Za-z0-9._~-]{1,128}(?![\\\\s\\\\S])",
        "description": "Opaque per-process server instance id; never carries a path or credential."
      },
      "ModelId": {
        "type": "string",
        "pattern": "^mdl_[A-Za-z0-9_-]{43}(?![\\\\s\\\\S])",
        "description": "Opaque stable UI model id."
      },
      "OperationId": {
        "type": "string",
        "minLength": 1,
        "maxLength": 128,
        "pattern": "^[A-Za-z0-9._~-]{1,128}(?![\\\\s\\\\S])",
        "description": "Opaque operation id; printable token only."
      },
      "EventId": {
        "type": "string",
        "minLength": 1,
        "maxLength": 160,
        "pattern": "^evt_[A-Za-z0-9._~-]{1,160}(?![\\\\s\\\\S])",
        "description": "Opaque SSE event id; also sent as the SSE id field."
      },
      "EventSequence": {
        "type": "integer",
        "minimum": 0,
        "maximum": 9007199254740991,
        "description": "Non-negative UI event sequence small enough to round-trip through JavaScript Number without precision loss."
      },
      "EventReplayQuery": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "server_instance_id",
          "after_sequence"
        ],
        "properties": {
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "after_sequence": {
            "$ref": "#/components/schemas/EventSequence"
          }
        }
      },
      "CursorToken": {
        "type": "string",
        "minLength": 1,
        "maxLength": 512,
        "pattern": "^[A-Za-z0-9._~-]{1,512}(?![\\\\s\\\\S])",
        "description": "Opaque pagination cursor. Control characters, slashes and backslashes are forbidden."
      },
      "IdempotencyKey": {
        "type": "string",
        "minLength": 8,
        "maxLength": 128,
        "pattern": "^[A-Za-z0-9._~-]{8,128}(?![\\\\s\\\\S])",
        "description": "Client-supplied idempotency token scoped to one server instance; printable token only."
      },
      "HuggingFaceRepoId": {
        "type": "string",
        "minLength": 1,
        "maxLength": 193,
        "pattern": "^(?!.*(?:^|/)\\\\.\\\\.?($|/))[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}(?![\\\\s\\\\S])",
        "description": "Public HuggingFace owner/name id. Dot-only path segments and URL/path syntax are rejected."
      },
      "RevisionRef": {
        "type": [
          "string",
          "null"
        ],
        "minLength": 1,
        "maxLength": 128,
        "pattern": "^(?!.*(?:^|/)\\\\.\\\\.?($|/))[A-Za-z0-9._~+/-]+(?![\\\\s\\\\S])",
        "description": "Repository revision ref resolved and pinned before writing weights. Dot-only path segments and control characters are rejected."
      },
      "ApiBase": {
        "type": "string",
        "minLength": 0,
        "maxLength": 128,
        "pattern": "^(?![\\\\s\\\\S])|^/(?!/)(?!.*//)(?!.*(?:^|/)\\\\.\\\\.?($|/))[A-Za-z0-9._~!$&'()*+,;=:@%/-]*(?![\\\\s\\\\S])",
        "description": "Validated same-origin relative API prefix. Protocol-relative URLs, query strings, fragments, dot segments and doubled slashes are rejected."
      },
      "KvCacheModeName": {
        "type": "string",
        "enum": [
          "fp16",
          "float16",
          "int8",
          "i8",
          "turbo4-asym",
          "fp16+turbo4",
          "turbo3-asym",
          "fp16+turbo3",
          "turbo3",
          "turbo4",
          "turbo4-sym",
          "turbo4-delegated",
          "fp16+turbo4-delegated"
        ],
        "description": "Accepted by KVCacheMode::from_str / --kv-cache-mode; GGML-only cache spellings such as q8_0 are intentionally absent."
      },
      "ErrorCode": {
        "type": "string",
        "enum": [
          "invalid_request",
          "unauthorized",
          "forbidden",
          "not_found",
          "stale_revision",
          "conflict",
          "unsupported",
          "rate_limited",
          "unavailable",
          "payload_too_large",
          "server_restarted",
          "event_gap",
          "partial_success"
        ]
      },
      "FieldError": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "field",
          "code",
          "message"
        ],
        "properties": {
          "field": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": "^[A-Za-z0-9_.:-]+(?![\\\\s\\\\S])"
          },
          "code": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
            "pattern": "^[A-Za-z0-9_.:-]+(?![\\\\s\\\\S])"
          },
          "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512
          }
        }
      },
      "ErrorBody": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "code",
          "message",
          "retryable"
        ],
        "properties": {
          "code": {
            "$ref": "#/components/schemas/ErrorCode"
          },
          "message": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512
          },
          "retryable": {
            "type": "boolean"
          },
          "field_errors": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/FieldError"
            },
            "maxItems": 16
          },
          "operation_id": {
            "$ref": "#/components/schemas/OperationId"
          }
        }
      },
      "ErrorEnvelope": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "error",
          "request_id"
        ],
        "properties": {
          "error": {
            "$ref": "#/components/schemas/ErrorBody"
          },
          "request_id": {
            "$ref": "#/components/schemas/OperationId"
          }
        }
      },
      "ActionAvailability": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "state",
          "reason"
        ],
        "properties": {
          "state": {
            "$ref": "#/components/schemas/ActionState"
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 256
          },
          "instructions": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "BuildInfo": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "version",
          "git_commit",
          "target",
          "features"
        ],
        "properties": {
          "version": {
            "type": "string"
          },
          "git_commit": {
            "type": [
              "string",
              "null"
            ]
          },
          "target": {
            "type": "string"
          },
          "features": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/FeatureFlag"
            }
          }
        }
      },
      "BackendIdentity": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "server_instance_id",
          "mode",
          "api_base",
          "auth_required",
          "build"
        ],
        "properties": {
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "mode": {
            "$ref": "#/components/schemas/ServerMode"
          },
          "api_base": {
            "$ref": "#/components/schemas/ApiBase"
          },
          "auth_required": {
            "type": "boolean"
          },
          "build": {
            "$ref": "#/components/schemas/BuildInfo"
          }
        }
      },
      "RootSummary": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "kind",
          "display_name",
          "redacted"
        ],
        "properties": {
          "kind": {
            "type": "string",
            "enum": [
              "cache",
              "models_dir",
              "preset_file",
              "single_model"
            ]
          },
          "display_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128
          },
          "redacted": {
            "type": "boolean"
          },
          "writable": {
            "type": "boolean"
          },
          "error": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "LimitSummary": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "catalog_default_page_size",
          "catalog_max_page_size",
          "json_body_bytes",
          "metadata_bytes_per_entry",
          "events_ring_size",
          "events_retention_seconds",
          "terminal_operations_retained",
          "terminal_operations_retention_seconds",
          "max_active_operations",
          "max_concurrent_loads",
          "max_concurrent_downloads",
          "next_load_ctx_size_max",
          "next_load_n_parallel_max",
          "cursor_bytes",
          "settings_fields_max",
          "measurements_max"
        ],
        "properties": {
          "catalog_default_page_size": {
            "type": "integer",
            "const": 50
          },
          "catalog_max_page_size": {
            "type": "integer",
            "const": 200
          },
          "json_body_bytes": {
            "type": "integer",
            "const": 2097152
          },
          "metadata_bytes_per_entry": {
            "type": "integer",
            "const": 16384
          },
          "events_ring_size": {
            "type": "integer",
            "const": 1024
          },
          "events_retention_seconds": {
            "type": "integer",
            "const": 600
          },
          "terminal_operations_retained": {
            "type": "integer",
            "const": 200
          },
          "terminal_operations_retention_seconds": {
            "type": "integer",
            "const": 3600
          },
          "max_active_operations": {
            "type": "integer",
            "const": 64
          },
          "max_concurrent_loads": {
            "type": "integer",
            "const": 1
          },
          "max_concurrent_downloads": {
            "type": "integer",
            "const": 1
          },
          "next_load_ctx_size_max": {
            "type": "integer",
            "const": 262144
          },
          "next_load_n_parallel_max": {
            "type": "integer",
            "const": 32
          },
          "cursor_bytes": {
            "type": "integer",
            "const": 512
          },
          "settings_fields_max": {
            "type": "integer",
            "const": 64
          },
          "measurements_max": {
            "type": "integer",
            "const": 64
          }
        }
      },
      "BootstrapResponse": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server",
          "features",
          "actions",
          "roots",
          "limits",
          "media_limits"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server": {
            "$ref": "#/components/schemas/BackendIdentity"
          },
          "features": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/FeatureFlag"
            },
            "maxItems": 16
          },
          "actions": {
            "type": "object",
            "additionalProperties": {
              "$ref": "#/components/schemas/ActionAvailability"
            },
            "maxProperties": 32
          },
          "roots": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/RootSummary"
            },
            "maxItems": 8
          },
          "limits": {
            "$ref": "#/components/schemas/LimitSummary"
          },
          "media_limits": {
            "$ref": "#/components/schemas/MediaLimits"
          }
        }
      },
      "ModelIdentity": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "id",
          "inference_id",
          "display_name",
          "source",
          "source_key_hash",
          "generation",
          "revision",
          "content_fingerprint"
        ],
        "properties": {
          "id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "inference_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256
          },
          "display_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256,
            "description": "The model's inference id with no character rewriting: the full id for cache and preset entries (owner/name for a cache repo), the directory name for models_dir entries, and the last /-separated segment of the served id for single_model. A label only: select and operate on a model by id."
          },
          "source": {
            "$ref": "#/components/schemas/CatalogSourceKind"
          },
          "source_key_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}(?![\\\\s\\\\S])",
            "description": "SHA-256 of the redacted canonical source key; used for collision tests without exposing paths."
          },
          "generation": {
            "type": "integer",
            "minimum": 1
          },
          "revision": {
            "type": "integer",
            "minimum": 1
          },
          "content_fingerprint": {
            "type": [
              "string",
              "null"
            ],
            "description": "Fingerprint of currently observed checkpoint content when known; affects generation/revision, not stable ID.",
            "maxLength": 128
          }
        }
      },
      "MeasuredValue": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "value",
          "unit",
          "scope",
          "measured_at",
          "reason"
        ],
        "properties": {
          "value": {
            "type": [
              "number",
              "null"
            ]
          },
          "unit": {
            "type": "string",
            "maxLength": 32
          },
          "scope": {
            "type": "string",
            "enum": [
              "model",
              "slot",
              "pool",
              "server",
              "unknown"
            ]
          },
          "measured_at": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ]
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 256
          }
        }
      },
      "Capability": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "task",
          "phase",
          "available",
          "reason"
        ],
        "properties": {
          "task": {
            "$ref": "#/components/schemas/TaskKind"
          },
          "phase": {
            "$ref": "#/components/schemas/CapabilityPhase"
          },
          "available": {
            "type": "boolean"
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 256
          }
        }
      },
      "LifecycleSnapshot": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "state",
          "download",
          "busy",
          "active_requests",
          "draining_requests",
          "worker_exit_observed",
          "last_error"
        ],
        "properties": {
          "state": {
            "$ref": "#/components/schemas/ModelLifecycleState"
          },
          "download": {
            "$ref": "#/components/schemas/DownloadState"
          },
          "busy": {
            "type": "boolean"
          },
          "active_requests": {
            "type": "integer",
            "minimum": 0
          },
          "draining_requests": {
            "type": "integer",
            "minimum": 0
          },
          "worker_exit_observed": {
            "type": "boolean"
          },
          "last_error": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "CatalogEntry": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "identity",
          "capabilities",
          "lifecycle",
          "complete",
          "supported",
          "removable",
          "metadata",
          "removal"
        ],
        "properties": {
          "identity": {
            "$ref": "#/components/schemas/ModelIdentity"
          },
          "capabilities": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Capability"
            },
            "maxItems": 16
          },
          "lifecycle": {
            "$ref": "#/components/schemas/LifecycleSnapshot"
          },
          "complete": {
            "type": "boolean"
          },
          "supported": {
            "type": "boolean"
          },
          "removable": {
            "type": "boolean"
          },
          "metadata": {
            "$ref": "#/components/schemas/CatalogMetadata"
          },
          "removal": {
            "$ref": "#/components/schemas/RemovalStatus"
          }
        }
      },
      "Pagination": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "limit",
          "next_cursor",
          "total_known"
        ],
        "properties": {
          "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 200
          },
          "next_cursor": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512,
            "pattern": "^[A-Za-z0-9._~-]{1,512}(?![\\\\s\\\\S])"
          },
          "total_known": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          }
        }
      },
      "CatalogListResponse": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "items",
          "pagination",
          "server_instance_id",
          "snapshot_sequence"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "items": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/CatalogEntry"
            },
            "maxItems": 200
          },
          "pagination": {
            "$ref": "#/components/schemas/Pagination"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "snapshot_sequence": {
            "$ref": "#/components/schemas/EventSequence",
            "description": "Single global event sequence after which this snapshot is authoritative."
          }
        }
      },
      "LoadProfile": {
        "type": "object",
        "additionalProperties": false,
        "required": [],
        "description": "Next-load profile override. Contract gate #1835 intentionally exposes only #1846-approved fields backed by existing server startup validators.",
        "properties": {
          "ctx_size": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 1,
            "maximum": 262144,
            "description": "Maps to existing --ctx-size; WebUI rejects larger values rather than attempting a new runtime mode."
          },
          "n_parallel": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 1,
            "maximum": 32,
            "description": "Maps to existing --parallel/--n-parallel slot count."
          },
          "kv_cache_mode": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/KvCacheModeName"
              },
              {
                "type": "null"
              }
            ]
          }
        }
      },
      "ModelActionRequest": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "model_id",
          "action",
          "expected_revision",
          "idempotency_key"
        ],
        "properties": {
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "action": {
            "type": "string",
            "enum": [
              "load",
              "unload"
            ]
          },
          "expected_revision": {
            "type": "integer",
            "minimum": 1
          },
          "idempotency_key": {
            "$ref": "#/components/schemas/IdempotencyKey"
          },
          "load_profile": {
            "$ref": "#/components/schemas/LoadProfile"
          },
          "eviction_target_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "eviction_target_expected_revision": {
            "type": "integer",
            "minimum": 1,
            "description": "Catalog revision of eviction_target_id observed when the user confirmed the eviction; required when eviction_target_id is present and forbidden otherwise."
          }
        }
      },
      "DownloadRequest": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "repo_id",
          "idempotency_key"
        ],
        "properties": {
          "repo_id": {
            "$ref": "#/components/schemas/HuggingFaceRepoId"
          },
          "revision": {
            "$ref": "#/components/schemas/RevisionRef"
          },
          "idempotency_key": {
            "$ref": "#/components/schemas/IdempotencyKey"
          }
        }
      },
      "RemovalRequest": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "model_id",
          "expected_revision",
          "idempotency_key"
        ],
        "properties": {
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "expected_revision": {
            "type": "integer",
            "minimum": 1
          },
          "idempotency_key": {
            "$ref": "#/components/schemas/IdempotencyKey"
          }
        }
      },
      "CatalogOperationTarget": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "target_kind",
          "scope"
        ],
        "properties": {
          "target_kind": {
            "const": "catalog"
          },
          "scope": {
            "type": "string",
            "enum": [
              "full",
              "roots",
              "entry"
            ]
          },
          "model_id": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelId"
              },
              {
                "type": "null"
              }
            ]
          }
        }
      },
      "ModelOperationTarget": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "target_kind",
          "model_id",
          "requested_revision"
        ],
        "properties": {
          "target_kind": {
            "const": "model"
          },
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "requested_revision": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 1
          },
          "eviction_target_id": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelId"
              },
              {
                "type": "null"
              }
            ]
          },
          "eviction_target_expected_revision": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 1
          }
        }
      },
      "DownloadOperationTarget": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "target_kind",
          "repo_id",
          "revision"
        ],
        "properties": {
          "target_kind": {
            "const": "download"
          },
          "repo_id": {
            "$ref": "#/components/schemas/HuggingFaceRepoId"
          },
          "revision": {
            "$ref": "#/components/schemas/RevisionRef"
          }
        }
      },
      "SettingsOperationTarget": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "target_kind",
          "model_id",
          "scope"
        ],
        "properties": {
          "target_kind": {
            "const": "settings"
          },
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "scope": {
            "type": "string",
            "enum": [
              "next_load_profile",
              "loaded_model_live",
              "request_only"
            ]
          }
        }
      },
      "OperationTarget": {
        "oneOf": [
          {
            "$ref": "#/components/schemas/CatalogOperationTarget"
          },
          {
            "$ref": "#/components/schemas/ModelOperationTarget"
          },
          {
            "$ref": "#/components/schemas/DownloadOperationTarget"
          },
          {
            "$ref": "#/components/schemas/SettingsOperationTarget"
          }
        ]
      },
      "CatalogRefreshResult": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "result_kind",
          "scanned_entries",
          "changed_entries",
          "snapshot_sequence"
        ],
        "properties": {
          "result_kind": {
            "const": "catalog_refresh"
          },
          "scanned_entries": {
            "type": "integer",
            "minimum": 0
          },
          "changed_entries": {
            "type": "integer",
            "minimum": 0
          },
          "snapshot_sequence": {
            "$ref": "#/components/schemas/EventSequence"
          }
        }
      },
      "ModelEvictionReport": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "requested_target_id",
          "displaced_model_id",
          "outcome",
          "rollbackable"
        ],
        "properties": {
          "requested_target_id": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelId"
              },
              {
                "type": "null"
              }
            ]
          },
          "displaced_model_id": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelId"
              },
              {
                "type": "null"
              }
            ]
          },
          "outcome": {
            "type": "string",
            "enum": [
              "not_needed",
              "displaced",
              "failed_after_displacement"
            ]
          },
          "rollbackable": {
            "type": "boolean"
          }
        }
      },
      "ModelActionResult": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "result_kind",
          "model_id",
          "revision",
          "lifecycle"
        ],
        "properties": {
          "result_kind": {
            "type": "string",
            "enum": [
              "model_load",
              "model_unload",
              "model_removal"
            ]
          },
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "revision": {
            "type": "integer",
            "minimum": 1
          },
          "lifecycle": {
            "$ref": "#/components/schemas/LifecycleSnapshot"
          },
          "eviction": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelEvictionReport"
              },
              {
                "type": "null"
              }
            ]
          }
        }
      },
      "DownloadResult": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "result_kind",
          "repo_id",
          "revision",
          "download"
        ],
        "properties": {
          "result_kind": {
            "const": "download"
          },
          "repo_id": {
            "$ref": "#/components/schemas/HuggingFaceRepoId"
          },
          "revision": {
            "$ref": "#/components/schemas/RevisionRef"
          },
          "model_id": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ModelId"
              },
              {
                "type": "null"
              }
            ]
          },
          "download": {
            "$ref": "#/components/schemas/DownloadState"
          }
        }
      },
      "SettingsPatchResult": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "result_kind",
          "model_id",
          "settings"
        ],
        "properties": {
          "result_kind": {
            "const": "settings_patch"
          },
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "settings": {
            "$ref": "#/components/schemas/RuntimeSettingsReport"
          }
        }
      },
      "OperationResult": {
        "oneOf": [
          {
            "$ref": "#/components/schemas/CatalogRefreshResult"
          },
          {
            "$ref": "#/components/schemas/ModelActionResult"
          },
          {
            "$ref": "#/components/schemas/DownloadResult"
          },
          {
            "$ref": "#/components/schemas/SettingsPatchResult"
          }
        ]
      },
      "OperationAccepted": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "operation_id",
          "state",
          "idempotent_replay"
        ],
        "properties": {
          "operation_id": {
            "$ref": "#/components/schemas/OperationId"
          },
          "state": {
            "$ref": "#/components/schemas/OperationState"
          },
          "idempotent_replay": {
            "type": "boolean"
          }
        }
      },
      "ProgressBytes": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "completed_bytes",
          "total_bytes",
          "indeterminate"
        ],
        "properties": {
          "completed_bytes": {
            "type": "integer",
            "minimum": 0
          },
          "total_bytes": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "indeterminate": {
            "type": "boolean"
          }
        }
      },
      "Operation": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "operation_id",
          "kind",
          "state",
          "created_at",
          "updated_at",
          "idempotency_scope",
          "target",
          "progress",
          "result",
          "error",
          "cancellable",
          "cancel_reason"
        ],
        "properties": {
          "operation_id": {
            "$ref": "#/components/schemas/OperationId"
          },
          "kind": {
            "$ref": "#/components/schemas/OperationKind"
          },
          "state": {
            "$ref": "#/components/schemas/OperationState"
          },
          "created_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC timestamp with offset, e.g. 2026-09-12T03:04:05Z."
          },
          "updated_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC timestamp with offset, e.g. 2026-09-12T03:04:05Z."
          },
          "idempotency_scope": {
            "type": "string",
            "const": "server_instance"
          },
          "target": {
            "$ref": "#/components/schemas/OperationTarget"
          },
          "progress": {
            "$ref": "#/components/schemas/ProgressBytes"
          },
          "result": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/OperationResult"
              },
              {
                "type": "null"
              }
            ]
          },
          "error": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ErrorBody"
              },
              {
                "type": "null"
              }
            ]
          },
          "cancellable": {
            "type": "boolean"
          },
          "cancel_reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "OperationsListResponse": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "items",
          "pagination",
          "server_instance_id",
          "snapshot_sequence"
        ],
        "properties": {
          "items": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Operation"
            },
            "maxItems": 200
          },
          "pagination": {
            "$ref": "#/components/schemas/Pagination"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "snapshot_sequence": {
            "$ref": "#/components/schemas/EventSequence",
            "description": "Single global event sequence after which this snapshot is authoritative."
          }
        }
      },
      "RuntimeSettingsReport": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "scope",
          "effective",
          "overridden_by_cli",
          "partial_errors"
        ],
        "properties": {
          "scope": {
            "type": "string",
            "enum": [
              "server_startup",
              "per_model_preset",
              "next_load_profile",
              "request_only",
              "loaded_model_live"
            ]
          },
          "effective": {
            "type": "object",
            "additionalProperties": {
              "type": [
                "string",
                "number",
                "boolean",
                "null"
              ]
            },
            "maxProperties": 64
          },
          "overridden_by_cli": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "maxItems": 64
          },
          "partial_errors": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/FieldError"
            },
            "maxItems": 16
          }
        }
      },
      "RuntimeSnapshot": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "model_id",
          "revision",
          "snapshot_sequence",
          "measurements",
          "settings",
          "slots"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "revision": {
            "type": "integer",
            "minimum": 1
          },
          "measurements": {
            "type": "object",
            "additionalProperties": {
              "$ref": "#/components/schemas/MeasuredValue"
            },
            "maxProperties": 64
          },
          "settings": {
            "$ref": "#/components/schemas/RuntimeSettingsReport"
          },
          "snapshot_sequence": {
            "$ref": "#/components/schemas/EventSequence",
            "description": "Single global event sequence after which this snapshot is authoritative."
          },
          "slots": {
            "$ref": "#/components/schemas/RuntimeSlots"
          }
        }
      },
      "UiEvent": {
        "oneOf": [
          {
            "$ref": "#/components/schemas/SnapshotEvent"
          },
          {
            "$ref": "#/components/schemas/ModelRevisionEvent"
          },
          {
            "$ref": "#/components/schemas/OperationEvent"
          },
          {
            "$ref": "#/components/schemas/DownloadProgressEvent"
          },
          {
            "$ref": "#/components/schemas/RuntimeEvent"
          },
          {
            "$ref": "#/components/schemas/SettingsEvent"
          },
          {
            "$ref": "#/components/schemas/ResetEvent"
          },
          {
            "$ref": "#/components/schemas/GapEvent"
          },
          {
            "$ref": "#/components/schemas/ServerRestartEvent"
          },
          {
            "$ref": "#/components/schemas/HeartbeatEvent"
          }
        ]
      },
      "StringContract": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "key",
          "en",
          "ko",
          "test_id"
        ],
        "properties": {
          "key": {
            "type": "string",
            "pattern": "^[a-z0-9_.-]+(?![\\\\s\\\\S])",
            "maxLength": 128
          },
          "en": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512
          },
          "ko": {
            "type": "string",
            "minLength": 1,
            "maxLength": 512
          },
          "test_id": {
            "type": "string",
            "pattern": "^[a-z0-9-]+(?![\\\\s\\\\S])",
            "maxLength": 128
          }
        }
      },
      "TransitionStep": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "from",
          "action",
          "to",
          "http_status",
          "event",
          "allowed"
        ],
        "properties": {
          "from": {
            "$ref": "#/components/schemas/ModelLifecycleState"
          },
          "action": {
            "type": "string",
            "minLength": 1
          },
          "to": {
            "$ref": "#/components/schemas/ModelLifecycleState"
          },
          "http_status": {
            "type": "integer",
            "minimum": 100,
            "maximum": 599
          },
          "event": {
            "$ref": "#/components/schemas/UiEventType"
          },
          "allowed": {
            "type": "boolean"
          },
          "error": {
            "anyOf": [
              {
                "$ref": "#/components/schemas/ErrorEnvelope"
              },
              {
                "type": "null"
              }
            ]
          }
        }
      },
      "WebUiContractFixture": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "scenario",
          "description",
          "given",
          "steps",
          "expected"
        ],
        "properties": {
          "scenario": {
            "type": "string",
            "enum": [
              "duplicate_load",
              "load_unload_race",
              "busy_eviction",
              "stale_revision",
              "failed_load",
              "download_cancel",
              "deletion_refusal",
              "sse_gap",
              "server_restart",
              "unknown_null_partial_error"
            ]
          },
          "description": {
            "type": "string",
            "minLength": 1
          },
          "given": {
            "type": "object",
            "additionalProperties": {
              "type": [
                "string",
                "number",
                "boolean",
                "null",
                "array",
                "object"
              ]
            }
          },
          "steps": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TransitionStep"
            },
            "minItems": 1
          },
          "expected": {
            "type": "object",
            "additionalProperties": {
              "type": [
                "string",
                "number",
                "boolean",
                "null",
                "array",
                "object"
              ]
            }
          }
        }
      },
      "SupportStatus": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "architecturally_supported",
          "runnable_on_backend",
          "complete",
          "reason",
          "architecturally_supported_reason",
          "runnable_on_backend_reason",
          "complete_reason",
          "tested_checkpoint",
          "tested_checkpoint_reason"
        ],
        "properties": {
          "architecturally_supported": {
            "type": "boolean"
          },
          "runnable_on_backend": {
            "type": "boolean"
          },
          "complete": {
            "type": "boolean"
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "architecturally_supported_reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "runnable_on_backend_reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "complete_reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "tested_checkpoint": {
            "type": "boolean",
            "description": "True only when the catalog has explicit compatibility evidence for this checkpoint; static registry support alone does not set this."
          },
          "tested_checkpoint_reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "CatalogMetadata": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "architecture",
          "declared_architectures",
          "input_tasks",
          "output_tasks",
          "quantization",
          "dtype",
          "format",
          "parameter_count",
          "disk_bytes",
          "memory_estimate_bytes",
          "support",
          "model_type",
          "unknown_reasons"
        ],
        "properties": {
          "architecture": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 128
          },
          "declared_architectures": {
            "type": [
              "array",
              "null"
            ],
            "items": {
              "type": "string",
              "maxLength": 128
            },
            "maxItems": 16,
            "description": "Raw config.json architectures array when readable and bounded."
          },
          "input_tasks": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TaskKind"
            },
            "maxItems": 16
          },
          "output_tasks": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/TaskKind"
            },
            "maxItems": 16
          },
          "quantization": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 128
          },
          "dtype": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 32,
            "pattern": "^[a-z0-9_]+$",
            "description": "Declared weight dtype from config.json torch_dtype (or dtype), normalized to a short lowercase name such as bf16 or fp16; null when absent, auto, or not a recognized tensor dtype. Quantized checkpoints report the dtype of their unquantized tensors."
          },
          "format": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 128
          },
          "parameter_count": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "disk_bytes": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "memory_estimate_bytes": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "support": {
            "$ref": "#/components/schemas/SupportStatus"
          },
          "model_type": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 128,
            "description": "Raw config.json model_type when readable; architecture is the resolved mlxcel registry id."
          },
          "unknown_reasons": {
            "$ref": "#/components/schemas/CatalogMetadataUnknownReasons"
          }
        }
      },
      "SnapshotPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "snapshot_sequence",
          "catalog_changed",
          "operations_changed",
          "runtime_model_ids"
        ],
        "properties": {
          "snapshot_sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "catalog_changed": {
            "type": "boolean"
          },
          "operations_changed": {
            "type": "boolean"
          },
          "runtime_model_ids": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/ModelId"
            },
            "maxItems": 200
          }
        }
      },
      "ModelRevisionPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "model_id",
          "revision",
          "lifecycle"
        ],
        "properties": {
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "revision": {
            "type": "integer",
            "minimum": 1
          },
          "lifecycle": {
            "$ref": "#/components/schemas/LifecycleSnapshot"
          }
        }
      },
      "OperationPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "operation"
        ],
        "properties": {
          "operation": {
            "$ref": "#/components/schemas/Operation"
          }
        }
      },
      "DownloadProgressPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "operation_id",
          "progress"
        ],
        "properties": {
          "operation_id": {
            "$ref": "#/components/schemas/OperationId"
          },
          "progress": {
            "$ref": "#/components/schemas/ProgressBytes"
          }
        }
      },
      "RuntimePayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "runtime"
        ],
        "properties": {
          "runtime": {
            "$ref": "#/components/schemas/RuntimeSnapshot"
          }
        }
      },
      "SettingsPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "model_id",
          "settings"
        ],
        "properties": {
          "model_id": {
            "$ref": "#/components/schemas/ModelId"
          },
          "settings": {
            "$ref": "#/components/schemas/RuntimeSettingsReport"
          }
        }
      },
      "ResetPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "reason",
          "resnapshot"
        ],
        "properties": {
          "reason": {
            "enum": [
              "gap",
              "server_restart",
              "retention_expired",
              "manual_reset"
            ]
          },
          "resnapshot": {
            "type": "boolean"
          }
        }
      },
      "HeartbeatPayload": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "server_time"
        ],
        "properties": {
          "server_time": {
            "type": "string",
            "format": "date-time"
          }
        }
      },
      "RequirementMapEntry": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "id",
          "epic_requirement",
          "child",
          "contract_artifact",
          "test_fixture"
        ],
        "properties": {
          "id": {
            "type": "string",
            "maxLength": 256
          },
          "epic_requirement": {
            "type": "string",
            "maxLength": 256
          },
          "child": {
            "type": "string",
            "pattern": "^#[0-9]+(?![\\\\s\\\\S])"
          },
          "contract_artifact": {
            "type": "string",
            "maxLength": 256
          },
          "test_fixture": {
            "type": "string",
            "maxLength": 256
          }
        }
      },
      "RequirementMap": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "requirements"
        ],
        "properties": {
          "requirements": {
            "type": "array",
            "minItems": 1,
            "items": {
              "$ref": "#/components/schemas/RequirementMapEntry"
            },
            "maxItems": 64
          }
        }
      },
      "StringCatalog": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "strings"
        ],
        "properties": {
          "strings": {
            "type": "array",
            "minItems": 1,
            "items": {
              "$ref": "#/components/schemas/StringContract"
            },
            "maxItems": 1024
          }
        }
      },
      "SnapshotEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "snapshot"
          },
          "payload": {
            "$ref": "#/components/schemas/SnapshotPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "ModelRevisionEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "model_revision"
          },
          "payload": {
            "$ref": "#/components/schemas/ModelRevisionPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "OperationEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "operation"
          },
          "payload": {
            "$ref": "#/components/schemas/OperationPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "DownloadProgressEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "download_progress"
          },
          "payload": {
            "$ref": "#/components/schemas/DownloadProgressPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "RuntimeEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "runtime"
          },
          "payload": {
            "$ref": "#/components/schemas/RuntimePayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "SettingsEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "settings"
          },
          "payload": {
            "$ref": "#/components/schemas/SettingsPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "ResetEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "reset"
          },
          "payload": {
            "$ref": "#/components/schemas/ResetPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "GapEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "gap"
          },
          "payload": {
            "$ref": "#/components/schemas/ResetPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "ServerRestartEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "server_restart"
          },
          "payload": {
            "$ref": "#/components/schemas/ResetPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "HeartbeatEvent": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "schema_version",
          "server_instance_id",
          "sequence",
          "type",
          "payload",
          "event_id",
          "emitted_at"
        ],
        "properties": {
          "schema_version": {
            "$ref": "#/components/schemas/SchemaVersion"
          },
          "server_instance_id": {
            "$ref": "#/components/schemas/ServerInstanceId"
          },
          "sequence": {
            "$ref": "#/components/schemas/EventSequence"
          },
          "type": {
            "const": "heartbeat"
          },
          "payload": {
            "$ref": "#/components/schemas/HeartbeatPayload"
          },
          "event_id": {
            "$ref": "#/components/schemas/EventId"
          },
          "emitted_at": {
            "type": "string",
            "format": "date-time",
            "description": "RFC3339 UTC time at which the event was appended to the server ring."
          }
        }
      },
      "CanonicalIdentityInput": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "version",
          "source",
          "source_rank",
          "namespace_hash",
          "entry_key"
        ],
        "properties": {
          "version": {
            "type": "integer",
            "const": 1
          },
          "source": {
            "$ref": "#/components/schemas/CatalogSourceKind"
          },
          "source_rank": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3
          },
          "namespace_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}(?![\\\\s\\\\S])"
          },
          "entry_key": {
            "type": "string",
            "minLength": 1,
            "maxLength": 256
          }
        }
      },
      "IdentityVector": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "source",
          "source_rank",
          "redacted_source_key",
          "entry_key",
          "source_key_hash",
          "canonical_identity",
          "expected_id"
        ],
        "properties": {
          "source": {
            "$ref": "#/components/schemas/CatalogSourceKind"
          },
          "source_rank": {
            "type": "integer",
            "minimum": 0,
            "maximum": 3
          },
          "redacted_source_key": {
            "type": "string",
            "minLength": 1
          },
          "entry_key": {
            "type": "string",
            "minLength": 1
          },
          "source_key_hash": {
            "type": "string",
            "pattern": "^[a-f0-9]{64}(?![\\\\s\\\\S])"
          },
          "canonical_identity": {
            "$ref": "#/components/schemas/CanonicalIdentityInput"
          },
          "expected_id": {
            "type": "string",
            "pattern": "^mdl_[A-Za-z0-9_-]{43}(?![\\\\s\\\\S])"
          }
        }
      },
      "IdentityVectors": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "identity_vectors"
        ],
        "properties": {
          "identity_vectors": {
            "type": "array",
            "minItems": 2,
            "items": {
              "$ref": "#/components/schemas/IdentityVector"
            }
          }
        }
      },
      "RemovalStatus": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "eligible",
          "reason",
          "instructions"
        ],
        "properties": {
          "eligible": {
            "type": "boolean"
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "instructions": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "CatalogMetadataUnknownReasons": {
        "type": "object",
        "additionalProperties": false,
        "required": [],
        "description": "Reasons for metadata fields that are null or deliberately unmeasured.",
        "properties": {
          "architecture": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "declared_architectures": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "model_type": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "quantization": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "format": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "parameter_count": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "disk_bytes": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          },
          "memory_estimate_bytes": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 512
          }
        }
      },
      "RuntimeSlot": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "id",
          "processing",
          "prompt_tokens",
          "cached_prompt_tokens",
          "decoded_tokens"
        ],
        "properties": {
          "id": {
            "type": "integer",
            "minimum": 0
          },
          "processing": {
            "type": "boolean"
          },
          "prompt_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0,
            "description": "Current context occupancy from GET /slots n_prompt_tokens: includes processed prompt and accepted decoded tokens. Never add decoded_tokens again."
          },
          "cached_prompt_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "decoded_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          }
        }
      },
      "RuntimeSlots": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "available",
          "reason",
          "measured_at",
          "configured_parallelism",
          "effective_parallelism",
          "request_context_tokens",
          "shared_pool_context_tokens",
          "items"
        ],
        "properties": {
          "available": {
            "type": "boolean"
          },
          "reason": {
            "type": [
              "string",
              "null"
            ],
            "maxLength": 256
          },
          "measured_at": {
            "anyOf": [
              {
                "type": "string",
                "format": "date-time"
              },
              {
                "type": "null"
              }
            ]
          },
          "configured_parallelism": {
            "type": "integer",
            "minimum": 0
          },
          "effective_parallelism": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "request_context_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "shared_pool_context_tokens": {
            "type": [
              "integer",
              "null"
            ],
            "minimum": 0
          },
          "items": {
            "type": "array",
            "maxItems": 256,
            "items": {
              "$ref": "#/components/schemas/RuntimeSlot"
            }
          }
        }
      },
      "MediaLimits": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "max_images",
          "max_image_bytes",
          "max_width",
          "max_height",
          "max_decoded_bytes",
          "max_body_bytes"
        ],
        "properties": {
          "max_images": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          },
          "max_image_bytes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          },
          "max_width": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          },
          "max_height": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          },
          "max_decoded_bytes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          },
          "max_body_bytes": {
            "type": "integer",
            "minimum": 0,
            "maximum": 9007199254740991
          }
        },
        "description": "Resolved server image admission and WebUI JSON body limits; clients may impose additional lower local safety ceilings. JSON budget includes base64 encoding, transcript and request envelope."
      }
    }
  }
}
`,A=class extends Error{path;constructor(e,t){super(`${e}: ${t}`),this.path=e,this.name=`ValidationError`}},Ir=zr(Fr);function Lr(e,t,n=`$`){let r=Zr(Ir,[`components`,`schemas`],`#/components/schemas`)[e];if(r===void 0)throw new A(n,`unknown schema ${e}`);Br(r,t,n)}function Rr(e){return JSON.parse(e)}function zr(e){return Xr(Rr(e),`$contract`)}function Br(e,t,n){let r=Xr(e,`${n}#schema`);if(typeof r.$ref==`string`){Br(Jr(r.$ref),t,n);return}if(r.anyOf!==void 0){if(!Array.isArray(r.anyOf)||!r.anyOf.some(e=>Yr(e,t,n)))throw new A(n,`did not match any allowed schema`);return}if(r.oneOf!==void 0){if(!Array.isArray(r.oneOf))throw new A(n,`schema oneOf must be an array`);let e=r.oneOf.filter(e=>Yr(e,t,n)).length;if(e!==1)throw new A(n,`matched ${e} oneOf schemas`);return}if(r.const!==void 0&&t!==r.const)throw new A(n,`expected constant ${String(r.const)}`);if(Array.isArray(r.enum)&&!r.enum.some(e=>e===t))throw new A(n,`unexpected enum value ${String(t)}`);let i=r.type;if(Array.isArray(i)){if(!i.some(e=>qr(e,t)))throw new A(n,`expected one of ${i.join(`, `)}`)}else if(typeof i==`string`&&!qr(i,t))throw new A(n,`expected ${i}`);typeof t==`string`&&Wr(r,t,n),typeof t==`number`&&Kr(r,t,n),Array.isArray(t)&&Hr(r,t,n),Qr(t)&&Vr(r,t,n)}function Vr(e,t,n){let r=$r(e.required,`${n}#schema.required`,!1),i=e.properties===void 0?{}:Xr(e.properties,`${n}#schema.properties`);for(let e of r)if(!(e in t))throw new A(`${n}.${e}`,`missing required property`);if(e.maxProperties!==void 0&&Object.keys(t).length>ei(e.maxProperties,`${n}#schema.maxProperties`))throw new A(n,`too many object properties`);for(let[r,a]of Object.entries(t)){let t=i[r];if(t!==void 0)Br(t,a,`${n}.${r}`);else if(e.additionalProperties===!1)throw new A(`${n}.${r}`,`unexpected property`);else Qr(e.additionalProperties)&&Br(e.additionalProperties,a,`${n}.${r}`)}}function Hr(e,t,n){if(e.minItems!==void 0&&t.length<ei(e.minItems,`${n}#schema.minItems`))throw new A(n,`array shorter than minimum`);if(e.maxItems!==void 0&&t.length>ei(e.maxItems,`${n}#schema.maxItems`))throw new A(n,`array longer than maximum`);e.items!==void 0&&t.forEach((t,r)=>Br(e.items,t,`${n}[${r}]`))}function Ur(e){let t=0;for(let n=0;n<e.length;t++)n+=(e.codePointAt(n)??0)>65535?2:1;return t}function Wr(e,t,n){let r=Ur(t);if(e.minLength!==void 0&&r<ei(e.minLength,`${n}#schema.minLength`))throw new A(n,`string shorter than minimum`);if(e.maxLength!==void 0&&r>ei(e.maxLength,`${n}#schema.maxLength`))throw new A(n,`string longer than maximum`);if(typeof e.pattern==`string`&&!new RegExp(e.pattern,`u`).test(t))throw new A(n,`string does not match pattern`);if(e.format===`date-time`&&!Gr(t))throw new A(n,`invalid RFC3339 date-time`)}function Gr(e){return/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(e)?!Number.isNaN(Date.parse(e)):!1}function Kr(e,t,n){if(!Number.isFinite(t))throw new A(n,`expected finite number`);if(e.type===`integer`&&!Number.isInteger(t))throw new A(n,`expected integer`);if(e.minimum!==void 0&&t<ei(e.minimum,`${n}#schema.minimum`))throw new A(n,`number below minimum`);if(e.maximum!==void 0&&t>ei(e.maximum,`${n}#schema.maximum`))throw new A(n,`number above maximum`)}function qr(e,t){return e===`null`?t===null:e===`array`?Array.isArray(t):e===`object`?Qr(t):e===`integer`?typeof t==`number`&&Number.isInteger(t):e===`number`?typeof t==`number`&&Number.isFinite(t):typeof t===e}function Jr(e){if(!e.startsWith(`#/`))throw new A(`$ref`,`unsupported external ref ${e}`);return e.slice(2).split(`/`).map(e=>e.replace(/~1/g,`/`).replace(/~0/g,`~`)).reduce((t,n)=>Xr(t,e)[n],Ir)}function Yr(e,t,n){try{return Br(e,t,n),!0}catch(e){if(e instanceof A)return!1;throw e}}function Xr(e,t){if(!Qr(e))throw new A(t,`expected object`);return e}function Zr(e,t,n){return t.reduce((e,t)=>Xr(e,n)[t],e)}function Qr(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function $r(e,t,n){if(e===void 0&&!n)return[];if(!Array.isArray(e)||!e.every(e=>typeof e==`string`))throw new A(t,`expected string array`);return e}function ei(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw new A(t,`expected numeric schema bound`);return e}function ti(e,t,n){return Lr(e,t,n),t}function ni(e,t=`$`){return ti(`ErrorEnvelope`,e,t)}function ri(e,t=`$`){return ti(`BootstrapResponse`,e,t)}function ii(e,t=`$`){return ti(`CatalogEntry`,e,t)}function ai(e,t=`$`){return ti(`CatalogListResponse`,e,t)}function oi(e,t=`$`){return ti(`OperationsListResponse`,e,t)}function si(e,t=`$`){return ti(`Operation`,e,t)}function ci(e,t=`$`){return ti(`OperationAccepted`,e,t)}function li(e,t=`$`){return ti(`RuntimeSnapshot`,e,t)}function ui(e,t=`$`){return ti(`UiEvent`,e,t)}var di=class extends Error{status;envelope;constructor(e,t,n){super(n??t?.error.message??`WebUI request failed with HTTP ${e}`),this.status=e,this.envelope=t,this.name=`WebUiHttpError`}},fi=2097152,pi=class{apiBase;fetchImpl;bearerToken=null;onUnauthorized;controllers=new Set;constructor(e={}){this.apiBase=Ar(e.apiBase),this.fetchImpl=e.fetchImpl??fetch.bind(globalThis),this.onUnauthorized=e.onUnauthorized}setBearerToken(e){this.bearerToken=e}abortAll(){for(let e of this.controllers)e.abort();this.controllers.clear()}async bootstrap(e){return this.request(`/ui-api/v1/bootstrap`,ri,{method:`GET`,signal:e})}async catalog(e={},t){return this.request(`/ui-api/v1/catalog`,ai,{method:`GET`,query:{...e},signal:t})}async catalogEntry(e,t){return this.request(`/ui-api/v1/catalog/${Pr(e)}`,ii,{method:`GET`,signal:t})}async refreshCatalog(e,t){return this.request(`/ui-api/v1/catalog/refresh`,ci,{method:`POST`,body:{idempotency_key:e},signal:t})}async modelAction(e,t){return this.request(`/ui-api/v1/model-actions`,ci,{method:`POST`,body:e,signal:t})}async download(e,t){return this.request(`/ui-api/v1/downloads`,ci,{method:`POST`,body:e,signal:t})}async removeModel(e,t){return this.request(`/ui-api/v1/model-removals`,ci,{method:`POST`,body:e,signal:t})}async operationsPage(e={},t){return this.request(`/ui-api/v1/operations`,oi,{method:`GET`,query:e,signal:t})}async operations(e){let t=[],n,r=null;for(;;){let i=await this.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null)r=i.server_instance_id;else if(i.server_instance_id!==r)throw Error(`Operations pagination crossed a server restart; refresh the authoritative snapshot.`);if(t.push(...i.items),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operation(e,t){return this.request(`/ui-api/v1/operations/${Pr(e)}`,si,{method:`GET`,signal:t})}async cancelOperation(e,t){return this.request(`/ui-api/v1/operations/${Pr(e)}/cancel`,ci,{method:`POST`,body:{},signal:t})}async runtime(e,t){let n=await this.request(`/ui-api/v1/runtime`,li,{method:`GET`,query:{model_id:e,autoload:!1},signal:t});if(n.model_id!==e)throw Error(`Runtime response model_id did not match the requested model.`);return n}async tokenCount(e,t,n){return this.request(`/tokenize`,Tr,{method:`POST`,query:{model:e,autoload:!1},body:{content:t,add_special:!1,parse_special:!0,with_pieces:!1},signal:n})}async settings(e,t){return this.request(`/settings`,_r,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async patchSettings(e,t,n){return this.request(`/settings`,vr,{method:`PATCH`,query:{model:e,autoload:!1},body:{op:`merge`,values:t},signal:n})}async modelProps(e,t){return this.request(`/props`,yr,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async events(e,t,n){let r=n?.serverInstanceId!==void 0&&n.serverInstanceId!==null&&n.afterSequence!==null&&n.afterSequence!==void 0,i=r?{server_instance_id:n.serverInstanceId,after_sequence:n.afterSequence}:void 0,a=new Headers({Accept:`text/event-stream`});!r&&n?.lastEventId!==void 0&&n.lastEventId!==null&&a.set(`Last-Event-ID`,n.lastEventId),await this.sse(Nr(this.apiBase,`/ui-api/v1/events`,i),{method:`GET`,signal:t,headers:a},{onDone:e.onDone,onFrame:t=>{t.retry!==null&&e.onRetryAfter?.(t.retry),t.data.length>0&&e.onEvent(ui(Rr(t.data)))}})}async chatCompletions(e,t,n,r){await this.sse(Nr(this.apiBase,`/v1/chat/completions`,{autoload:!1}),{method:`POST`,body:gi(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async responses(e,t,n,r){await this.sse(Nr(this.apiBase,`/v1/responses`,{autoload:!1}),{method:`POST`,body:gi(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async request(e,t,n){let r=await this.fetchWithAuth(Nr(this.apiBase,e,n.query),n);try{return r.response.ok||await this.throwHttp(r.response,r.signal),t(Rr(await this.readBody(r.response,r.signal)))}finally{this.controllers.delete(r.controller),r.cleanup()}}async sse(e,t,n){let r=await this.fetchWithAuth(e,t);try{let e=r.response;if(e.ok||await this.throwHttp(e,r.signal),e.body===null)throw Error(`WebUI event stream response has no body.`);let t=new Dr({onDone:n.onDone,onMessage:n.onFrame}),i=e.body.getReader();try{for(;;){let e=await _i(i,r.signal);if(e.done)break;t.push(e.value)}if(t.close(),!t.done&&r.signal.aborted!==!0)throw Error(`WebUI event stream ended before a DONE frame.`)}catch(e){throw await i.cancel().catch(()=>void 0),e}finally{i.releaseLock()}}finally{this.controllers.delete(r.controller),r.cleanup()}}async fetchWithAuth(e,t){let n=new AbortController,{signal:r,cleanup:i}=vi(n,t.signal);this.controllers.add(n);let a=new Headers(t.headers);a.set(`Accept`,a.get(`Accept`)??`application/json`),this.bearerToken!==null&&a.set(`Authorization`,`Bearer ${this.bearerToken}`),t.body!==void 0&&a.set(`Content-Type`,`application/json`);try{return{response:await this.fetchImpl(e,{method:t.method,headers:a,body:t.body===void 0?void 0:JSON.stringify(t.body),signal:r,credentials:`same-origin`,cache:`no-store`,redirect:`error`}),controller:n,signal:r,cleanup:i}}catch(e){throw this.controllers.delete(n),i(),e}}async throwHttp(e,t){let n=this.bearerToken;e.status===401&&(this.bearerToken=null,this.abortAll(),this.onUnauthorized?.());let r=null;try{let n=await this.readBody(e,t);n.length>0&&(r=ni(Rr(n)))}catch{r=null}throw new di(e.status,mi(r,n),hi(r?.error.message??`WebUI request failed with HTTP ${e.status}`,n))}async readBody(e,t){if(e.body===null)return``;let n=e.body.getReader(),r=[],i=0;try{for(;;){let e=await _i(n,t);if(e.done)break;if(i+=e.value.byteLength,i>fi)throw Error(`WebUI JSON response exceeded the configured byte limit.`);r.push(e.value)}}catch(e){throw await n.cancel().catch(()=>void 0),e}finally{n.releaseLock()}let a=new Uint8Array(i),o=0;for(let e of r)a.set(e,o),o+=e.byteLength;return new TextDecoder().decode(a)}};function mi(e,t){return e===null?null:{...e,error:{...e.error,message:hi(e.error.message,t),field_errors:e.error.field_errors?.map(e=>({...e,message:hi(e.message,t)}))}}}function hi(e,t){let n=e.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`).replace(/token[=:]\s*\S+/gi,`token=[redacted]`);return t!==null&&t.length>0&&(n=n.split(t).join(`[redacted]`)),n}function gi(e,t){if(t.length===0)throw Error(`Inference model id is required for streaming inference.`);if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Streaming inference bodies must be JSON objects.`);return{...e,model:t,stream:!0}}async function _i(e,t){if(t.aborted)throw await e.cancel().catch(()=>void 0),yi();let n=null;try{let r=await Promise.race([e.read(),new Promise((r,i)=>{n=()=>{e.cancel().catch(()=>void 0),i(yi())},t.addEventListener(`abort`,n,{once:!0})})]);if(t.aborted)throw yi();return r}finally{n!==null&&t.removeEventListener(`abort`,n)}}function vi(e,t){if(t===void 0)return{signal:e.signal,cleanup:()=>void 0};let n=()=>e.abort(t.reason);return t.aborted?n():t.addEventListener(`abort`,n,{once:!0}),{signal:e.signal,cleanup:()=>t.removeEventListener(`abort`,n)}}function yi(){return typeof DOMException==`function`?new DOMException(`The WebUI request was aborted.`,`AbortError`):Error(`The WebUI request was aborted.`)}function bi(e){return e instanceof di?e.status===401?`wrong-key`:e.status===403?`forbidden`:`generic`:e instanceof A||e instanceof Error&&e.name===`ValidationError`?`schema`:e instanceof TypeError||e instanceof Error&&/fetch|network|offline|connection/i.test(e.message)?`offline`:`generic`}var xi=new Set([`ready`,`loading`,`draining`,`unloading`]);function Si(e){return xi.has(e)}function Ci(e,t){let n=e=>e.lifecycle.state===`ready`?0:Si(e.lifecycle.state)?1:2;return n(e)-n(t)||e.identity.display_name.localeCompare(t.identity.display_name)||e.identity.id.localeCompare(t.identity.id)}function wi(e){return e.catalog.filter(e=>Si(e.lifecycle.state)).sort(Ci)}function Ti(e,t){return e.catalog.some(e=>e.identity.id===t)?t:null}var Ei={model_free:`connection.mode.model_free`,single_model:`connection.mode.single_model`,router_pool:`connection.mode.router_pool`};function Di(e,t){return x(e,Object.hasOwn(Ei,t)?Ei[t]:`connection.mode.unknown`)}function Oi(e,t){return t.bootstrap===null?x(e,`connection.ready`):x(e,`connection.footer.connected`,{mode:Di(e,t.bootstrap.server.mode),version:t.bootstrap.server.build.version,status:Ai(e,t.connection)})}function ki(e,t){if(t.bootstrap===null)return null;let n=x(e,`format.unknown`),r=t.serverInstanceId??t.bootstrap.server.server_instance_id,i=t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot;return{instance:r||n,sequence:i===null?n:String(i)}}function Ai(e,t){return x(e,{idle:`connection.status.idle`,bootstrapping:`connection.status.bootstrapping`,ready:`connection.status.ready`,streaming:`connection.status.streaming`,polling:`connection.status.polling`,offline:`connection.status.offline`,stale:`connection.status.stale`,unauthorized:`connection.status.unauthorized`,forbidden:`connection.status.forbidden`,"schema-mismatch":`connection.status.schema_mismatch`,error:`connection.status.error`}[t])}function ji(e){return e.snapshot.auth.status===`authenticated`?e.snapshot.connection===`schema-mismatch`?(0,C.jsx)(Pi,{...e}):(0,C.jsx)(Ni,{...e}):(0,C.jsx)(Mi,{...e})}function Mi(e){return e.authFailure===`schema`?(0,C.jsx)(Pi,{...e}):(0,C.jsxs)(`div`,{className:`screen-stack`,children:[(0,C.jsx)(ie,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:x(e.locale,`connection.prompt.body`),descriptionTestId:b(`connection.prompt.body`)}),(0,C.jsx)(oe,{title:x(e.locale,`state.unauthorized.title`),body:x(e.locale,`state.unauthorized.body`),tokenLabel:x(e.locale,`login.token.label`),tokenHelp:x(e.locale,`login.token.help`),submitLabel:x(e.locale,`login.submit`),logoutLabel:x(e.locale,`login.logout`),error:e.authFailure===null?void 0:Li(e.locale,e.authFailure),busy:e.snapshot.auth.status===`authenticating`,onSubmit:e.onLogin,onLogout:e.onLogout,testId:b(`auth.login`)})]})}function Ni(e){let t=e.snapshot.connection===`offline`||e.snapshot.connection===`stale`||e.snapshot.connection===`error`||e.snapshot.connection===`unauthorized`||e.snapshot.connection===`forbidden`;return(0,C.jsxs)(`div`,{className:`screen-stack`,children:[(0,C.jsx)(ie,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:x(e.locale,`connection.authenticated.body`),descriptionTestId:b(`connection.authenticated.body`)}),(0,C.jsx)(v,{tone:`info`,title:x(e.locale,`connection.authenticated.title`),body:Fi(e.locale,e.snapshot)}),t?(0,C.jsx)(v,{title:x(e.locale,`connection.error.title`),body:Ii(e.locale,e.snapshot.connection),action:(0,C.jsx)(m,{onClick:e.onRetry,children:x(e.locale,`common.retry`)}),testId:b(`connection.error.title`)}):null]})}function Pi(e){return(0,C.jsxs)(`div`,{className:`screen-stack`,children:[(0,C.jsx)(ie,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId}),(0,C.jsx)(se,{title:x(e.locale,`state.schema_mismatch.title`),body:x(e.locale,`state.schema_mismatch.body`),actionLabel:x(e.locale,`common.reload`),onRecover:e.onRecoverSchema})]})}function Fi(e,t){let n=t.bootstrap;return n===null?x(e,`connection.prompt.detail`):x(e,`connection.authenticated.detail`,{mode:Di(e,n.server.mode),version:n.server.build.version,status:Ai(e,t.connection),count:t.catalogSequence===null?x(e,`connection.snapshot.pending`):String(t.catalog.length),operations:t.resourceFences.operationsSnapshot===null?x(e,`connection.snapshot.pending`):String(t.operations.size),sequence:zi(e,t)})}function Ii(e,t){return t===`offline`?x(e,`state.offline.body`):t===`stale`?x(e,`connection.error.stale`):t===`forbidden`?x(e,`connection.error.forbidden`):t===`unauthorized`?x(e,`connection.error.unauthorized`):x(e,`connection.error.generic`)}function Li(e,t){return t===`wrong-key`?x(e,`login.error.wrong_key`):t===`offline`?x(e,`login.error.offline`):t===`forbidden`?x(e,`login.error.forbidden`):t===`schema`?x(e,`login.error.schema`):x(e,`login.error.generic`)}function Ri(e,t){return x(e,{unloaded:`models.status.unloaded`,loading:`models.status.loading`,ready:`models.status.ready`,draining:`models.status.draining`,unloading:`models.status.unloading`,failed:`models.status.failed`}[t])}function zi(e,t){return String(t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot??x(e,`connection.snapshot.pending`))}var Bi=[{id:`models`,key:`nav.models`,testId:`command-route-models`},{id:`chat`,key:`nav.chat`,testId:`command-route-chat`},{id:`activity`,key:`nav.activity`,testId:`command-route-activity`},{id:`settings`,key:`nav.settings`,testId:`command-route-settings`},{id:`new-chat`,key:`chat.new_conversation`,testId:`command-new-chat`}];function Vi(e,t,n=20){let r=t.trim().toLowerCase();return r===``?[]:e.filter(e=>e.identity.display_name.toLowerCase().includes(r)||e.identity.id.toLowerCase().includes(r)).sort(Ci).slice(0,Math.max(0,n))}function Hi(e){let[t,n]=(0,S.useState)(``),[r,i]=(0,S.useState)(e.open),a=(0,S.useId)(),o=(0,S.useId)();e.open!==r&&(i(e.open),n(``));let s=t.trim().toLowerCase(),c=Bi.filter(t=>s===``||t.id.includes(s)||x(e.locale,t.key).toLowerCase().includes(s)),l=s===``?e.loaded:Vi(e.catalog,s),u=t=>{t.id===`new-chat`?e.onNewChat():e.onNavigate(t.id)};return(0,C.jsxs)(ne,{open:e.open,title:x(e.locale,`command.title`),onClose:e.onClose,testId:`command-dialog`,closeLabel:x(e.locale,`common.close`),children:[(0,C.jsx)(te,{label:x(e.locale,`command.search`),value:t,onChange:n,testId:b(`command.search`)}),c.length===0&&l.length===0?(0,C.jsx)(`p`,{"data-testid":b(`command.no_results`),children:x(e.locale,`command.no_results`)}):null,c.length>0?(0,C.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":a,children:[(0,C.jsx)(`h3`,{id:a,children:x(e.locale,`command.section.commands`)}),(0,C.jsx)(`div`,{className:`command-list`,children:c.map(t=>(0,C.jsx)(m,{"data-testid":t.testId,onClick:()=>u(t),children:x(e.locale,t.key)},t.id))})]}):null,l.length>0?(0,C.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":o,children:[(0,C.jsx)(`h3`,{id:o,children:x(e.locale,`command.section.models`)}),(0,C.jsx)(`div`,{className:`command-list`,children:l.map(t=>(0,C.jsx)(m,{className:`command-model`,title:t.identity.display_name,"data-testid":`command-model`,onClick:()=>e.onOpenModel(t.identity.id),children:`${t.identity.display_name} · ${Ri(e.locale,t.lifecycle.state)}`},t.identity.id))})]}):null]})}var Ui=[{id:`models`,key:`nav.models`,icon:`models`},{id:`chat`,key:`nav.chat`,icon:`chat`},{id:`activity`,key:`nav.activity`,icon:`activity`},{id:`settings`,key:`nav.settings`,icon:`settings`}],Wi=[{id:`command`,keys:[`⌘/Ctrl`,`K`],separator:`+`,key:`help.shortcut.command`},{id:`new-chat`,keys:[`⌘/Ctrl`,`N`],separator:`+`,key:`help.shortcut.new_chat`},{id:`send`,keys:[`⌘/Ctrl`,`Enter`],separator:`+`,key:`help.shortcut.send`},{id:`escape`,keys:[`Esc`],separator:`+`,key:`help.shortcut.escape`},{id:`navigate`,keys:[`[`,`]`],separator:`/`,key:`help.shortcut.navigate`},{id:`help`,keys:[`?`],separator:`+`,key:`help.shortcut.help`}];function Gi(e){let t=(0,S.useRef)(null),n=(0,S.useRef)(e.onCommand),r=(0,S.useRef)(e.onHelp),i=(0,S.useRef)(e.onNewChat),[a,o]=(0,S.useState)(!1);(0,S.useEffect)(()=>{n.current=e.onCommand,r.current=e.onHelp,i.current=e.onNewChat},[e.onCommand,e.onHelp,e.onNewChat]),(0,S.useEffect)(()=>{let e=e=>{let t=e.target,a=t instanceof HTMLElement&&(t.isContentEditable||!!t.closest(`[role="combobox"], [role="listbox"]`)||[`INPUT`,`TEXTAREA`,`SELECT`].includes(t.tagName)),s=t instanceof HTMLElement&&!!t.closest(`dialog[open], [role="dialog"][aria-modal="true"]`);if(e.isComposing||e.altKey||a||s)return;let c=e.metaKey||e.ctrlKey;if(c&&e.key.toLowerCase()===`k`&&(e.preventDefault(),o(!1),n.current()),c&&!e.shiftKey&&e.key.toLowerCase()===`n`){if(e.preventDefault(),e.repeat)return;o(!1),i.current()}(e.key===`?`||e.key===`/`&&e.shiftKey)&&(e.preventDefault(),o(!1),r.current())};return window.addEventListener(`keydown`,e),()=>window.removeEventListener(`keydown`,e)},[]),(0,S.useEffect)(()=>{if(!a||typeof window.matchMedia!=`function`)return;let e=window.matchMedia(`(max-width: 960px)`),n=()=>{if(e.matches)return;o(!1);let n=0,r=()=>{if(document.querySelector(`[data-testid="mobile-nav-sheet"][open]`)&&n++<120){requestAnimationFrame(r);return}let e=document.activeElement;if(e&&e!==document.body&&!e.closest(`[aria-modal="true"]`))return;let i=t.current;(i?.querySelector(`a[aria-current="page"]`)??i?.querySelector(`a`))?.focus()};requestAnimationFrame(r)};return e.addEventListener(`change`,n),()=>e.removeEventListener(`change`,n)},[a]);let s=n=>{let r=Ui[(Ui.findIndex(t=>t.id===e.route)+n+Ui.length)%Ui.length];e.onRouteChange(r.id),requestAnimationFrame(()=>t.current?.querySelector(`a[href="#${r.id}"]`)?.focus())},c=e=>{(e.key===`[`||e.key===`]`)&&(e.altKey||e.metaKey||e.ctrlKey||e.nativeEvent.isComposing||(e.preventDefault(),s(e.key===`]`?1:-1)))},l=t=>{e.onRouteChange(t),o(!1)};return(0,C.jsxs)(`div`,{className:`app-shell`,children:[(0,C.jsx)(Ji,{locale:e.locale,route:e.route,onRouteChange:l,onKeyDown:c,ref:t,className:`app-sidebar desktop-sidebar material-glass`,connection:e.connection,primary:!0}),(0,C.jsx)(ae,{open:a,title:x(e.locale,`nav.primary`),onClose:()=>o(!1),closeLabel:x(e.locale,`common.close`),testId:`mobile-nav-sheet`,children:(0,C.jsx)(Ji,{locale:e.locale,route:e.route,onRouteChange:l,onKeyDown:c,className:`app-sidebar sheet-sidebar`,connection:e.connection})}),(0,C.jsxs)(`main`,{className:`app-main`,"aria-labelledby":`app-title`,inert:a,children:[(0,C.jsxs)(`header`,{className:`app-toolbar material-glass`,children:[(0,C.jsx)(h,{className:`mobile-menu-button`,label:x(e.locale,`toolbar.menu`),icon:`menu`,onClick:()=>o(!0),"data-testid":b(`toolbar.menu`)}),(0,C.jsxs)(`div`,{className:`toolbar-title`,children:[(0,C.jsx)(`p`,{id:`app-title`,title:x(e.locale,`app.title`),"data-testid":b(`app.title`),children:x(e.locale,`app.title`)}),(0,C.jsx)(`span`,{title:x(e.locale,`app.subtitle`),"data-testid":b(`app.subtitle`),children:x(e.locale,`app.subtitle`)})]}),(0,C.jsx)(Ki,{locale:e.locale,models:e.loadedModels,onOpenModel:e.onOpenModel,onMore:e.onCommand}),(0,C.jsxs)(`div`,{className:`toolbar-actions`,children:[e.sessionAction,(0,C.jsx)(h,{label:x(e.locale,`toolbar.command`),icon:`command`,onClick:e.onCommand,"data-testid":b(`toolbar.command`)}),(0,C.jsx)(h,{label:x(e.locale,`toolbar.help`),icon:`help`,onClick:e.onHelp,"data-testid":b(`toolbar.help`)})]})]}),(0,C.jsxs)(`div`,{className:`app-content-grid ${e.inspector?`has-inspector`:``}`.trim(),children:[(0,C.jsx)(re,{className:`app-content`,children:e.children}),e.inspector]})]})]})}function Ki(e){let t=e.models??[],n=t.slice(0,3),r=t.length-n.length,i=e.models===null?`toolbar.loaded.unknown`:t.length===0?`toolbar.loaded.none`:null;return(0,C.jsx)(`div`,{className:`toolbar-loaded`,role:`group`,"aria-label":x(e.locale,`toolbar.loaded.label`),"data-testid":b(`toolbar.loaded.label`),onFocus:qi,children:i===null?(0,C.jsxs)(C.Fragment,{children:[(0,C.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":b(`toolbar.loaded.count`),children:x(e.locale,`toolbar.loaded.count`,{count:String(t.length)})}),n.map(t=>(0,C.jsx)(m,{tone:`ghost`,className:`toolbar-chip`,title:t.name,"data-testid":`toolbar-loaded-chip`,onClick:()=>e.onOpenModel(t.id),children:(0,C.jsxs)(`span`,{className:`toolbar-chip-content`,children:[(0,C.jsx)(`span`,{className:`truncate`,children:t.name}),(0,C.jsx)(ce,{state:t.state,appearance:`inline`,children:t.stateLabel})]})},t.id)),r>0?(0,C.jsx)(m,{tone:`ghost`,className:`toolbar-chip toolbar-chip-more`,"aria-label":x(e.locale,`toolbar.loaded.more_label`,{count:String(r)}),"data-testid":b(`toolbar.loaded.more`),onClick:e.onMore,children:x(e.locale,`toolbar.loaded.more`,{count:String(r)})}):null]}):(0,C.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":b(i),children:x(e.locale,i)})})}function qi(e){let t=e.target;t instanceof HTMLElement&&typeof t.scrollIntoView==`function`&&t.matches(`:focus-visible`)&&t.scrollIntoView({block:`nearest`,inline:`nearest`})}var Ji=S.forwardRef((e,t)=>(0,C.jsxs)(`aside`,{className:e.className,"aria-label":x(e.locale,`nav.primary`),onKeyDown:e.onKeyDown,ref:t,tabIndex:-1,children:[(0,C.jsxs)(`a`,{className:`brand-mark`,href:`#models`,"aria-label":x(e.locale,`nav.home`),onClick:t=>{t.preventDefault(),e.onRouteChange(`models`)},children:[(0,C.jsx)(`span`,{className:`brand-symbol`,"aria-hidden":`true`,children:`mx`}),(0,C.jsx)(`span`,{className:`brand-name`,children:`mlxcel`})]}),(0,C.jsx)(`nav`,{className:`app-nav`,children:Ui.map(t=>(0,C.jsxs)(`a`,{href:`#${t.id}`,"aria-label":x(e.locale,t.key),"aria-current":e.route===t.id?`page`:void 0,"data-testid":b(t.key),onClick:n=>{n.preventDefault(),e.onRouteChange(t.id)},children:[(0,C.jsx)(p,{name:t.icon}),(0,C.jsx)(`span`,{className:`app-nav-label`,children:x(e.locale,t.key)})]},t.id))}),(0,C.jsxs)(`footer`,{children:[(0,C.jsxs)(`div`,{className:`connection-status`,children:[(0,C.jsx)(`span`,{className:`connection-dot`,"data-state":e.connection.state,"aria-hidden":`true`}),(0,C.jsx)(`span`,{"data-testid":b(`connection.ready`),children:e.connection.label})]}),e.connection.details?(0,C.jsxs)(`details`,{className:`connection-details`,"data-testid":b(`connection.footer.details`),children:[(0,C.jsx)(`summary`,{children:x(e.locale,`connection.footer.details`)}),e.connection.summary?(0,C.jsx)(`p`,{"data-testid":e.primary?b(`connection.authenticated.detail`):void 0,children:e.connection.summary}):null,(0,C.jsx)(`p`,{"data-testid":b(`connection.footer.instance`),children:x(e.locale,`connection.footer.instance`,{id:e.connection.details.instance})}),(0,C.jsx)(`p`,{"data-testid":b(`connection.footer.sequence`),children:x(e.locale,`connection.footer.sequence`,{sequence:e.connection.details.sequence})})]}):null]})]}));Ji.displayName=`Sidebar`;var Yi=[`mlxcel`,`glass`],Xi=[`system`,`light`,`dark`],Zi={mlxcel:`mlxcel`,glass:`mlxcel-glass`},Qi=`mlxcel`,$i=`system`,ea=`(prefers-color-scheme: dark)`;function ta(e){return typeof e==`string`&&Yi.includes(e)}function na(e){return typeof e==`string`&&Xi.includes(e)}function ra(){try{return typeof window>`u`||typeof window.matchMedia!=`function`?`light`:window.matchMedia(`(prefers-color-scheme: dark)`).matches?`dark`:`light`}catch{return`light`}}function ia(e,t=ra()){return e===`system`?t:e}function aa(e,t,n=ra()){return{family:e,astryxTheme:Zi[e],scheme:ia(t,n),preference:t}}function oa(e){if(typeof window>`u`||typeof window.matchMedia!=`function`)return()=>void 0;let t;try{t=window.matchMedia(ea)}catch{return()=>void 0}let n=t=>e(t.matches?`dark`:`light`);return t.addEventListener(`change`,n),()=>t.removeEventListener(`change`,n)}var sa={themeFamily:Qi,colorScheme:$i,material:`glass`,glassIntensity:35,reduceMotion:!1,reduceTransparency:!1,highContrast:`system`,locale:`en`},ca=`mlxcel.webui.appearance`;function la(e){let t=typeof e==`number`&&Number.isFinite(e)?e:sa.glassIntensity;return Math.max(0,Math.min(100,Math.round(t)))}function ua(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function da(e){return e===`glass`||e===`tinted`||e===`opaque`}function fa(e){return e===`system`||e===`on`||e===`off`}function j(e){return fa(e)?e:e===!0?`on`:e===!1?`off`:sa.highContrast}function M(e){return na(e.colorScheme)?e.colorScheme:e.colorScheme===void 0&&na(e.theme)?e.theme:sa.colorScheme}function pa(e){return e===`en`||e===`ko`}function ma(){if(typeof window>`u`)return sa;let e;try{e=window.localStorage.getItem(ca)}catch{return sa}if(!e)return sa;try{let t=JSON.parse(e);return ua(t)?{themeFamily:ta(t.themeFamily)?t.themeFamily:sa.themeFamily,colorScheme:M(t),material:da(t.material)?t.material:sa.material,glassIntensity:la(t.glassIntensity),reduceMotion:t.reduceMotion===!0,reduceTransparency:t.reduceTransparency===!0,highContrast:j(t.highContrast),locale:pa(t.locale)?t.locale:sa.locale}:sa}catch{return sa}}function ha(e){try{window.localStorage.setItem(ca,JSON.stringify(e))}catch{}}function ga(){return typeof CSS>`u`||typeof CSS.supports!=`function`?!1:CSS.supports(`backdrop-filter: blur(1px)`)||CSS.supports(`-webkit-backdrop-filter: blur(1px)`)}function _a(e,t,n=ra()){let r=aa(t.themeFamily,t.colorScheme,n);e.dataset.theme=r.scheme,e.dataset.astryxTheme=r.astryxTheme,e.dataset.themeFamily=r.family,e.dataset.colorScheme=r.preference,e.dataset.material=t.reduceTransparency?`opaque`:t.material,e.dataset.glassIntensity=String(la(t.glassIntensity)),e.dataset.reduceMotion=String(t.reduceMotion),e.dataset.reduceTransparency=String(t.reduceTransparency),e.dataset.highContrast=t.highContrast,e.dataset.backdropFilter=ga()?`supported`:`unsupported`,e.lang=t.locale}var va=e=>e?.replace(/([a-z0-9])([A-Z])/g,`$1-$2`).toLowerCase();function ya(e,t,n=[]){if(t==null)throw Error(`[lucide]: iconNode is required when icon name is used`);return{name:va(e),size:24,node:t,...n.length>0?{aliases:n}:{}}}var ba=e=>{let t=``,n=!1;for(let r of e){if(r===`-`||r===`_`||r<=` `){n=t.length>0;continue}t.length===0?t+=r.toLowerCase():t+=n?r.toUpperCase():r,n=!1}return t},xa=e=>{let t=ba(e);return t.charAt(0).toUpperCase()+t.slice(1)},Sa=(...e)=>e.filter((e,t,n)=>!!e&&e.trim()!==``&&n.indexOf(e)===t).join(` `).trim(),Ca={xmlns:`http://www.w3.org/2000/svg`,width:24,height:24,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,"stroke-width":2,"stroke-linecap":`round`,"stroke-linejoin":`round`};function wa(e){return e!=null}function Ta(e,t={}){let n=t.attributeNames??{},r=e=>n[e]??e,i=e.size??e.width??Ca.width,a=e.size??e.height??Ca.height,o=e.aliases?.filter(e=>typeof e==`string`&&e.trim()!==``).map(e=>`lucide-${e}`)??[],s=[...e.name?[`lucide-${e.name}`]:[],...o],c=t.className?.split(` `).filter(Boolean)??[],l=t.includeDefaultClasses===!1?Sa(...c):Sa(`lucide`,...s,...c),u=t.absoluteStrokeWidth?Number(t.strokeWidth??Ca[`stroke-width`])*Number(e.size??e.width??Ca.width)/Number(t.size??t.width??Ca.width):t.strokeWidth??Ca[`stroke-width`];return[`svg`,{...Object.entries(Ca).reduce((e,[t,n])=>(e[r(t)]=n,e),{}),...`color`in t&&t.color&&{[r(`stroke`)]:t.color},...`size`in t&&wa(t.size)&&{[r(`width`)]:t.size,[r(`height`)]:t.size},...`width`in t&&wa(t.width)&&{[r(`width`)]:t.width},...`height`in t&&wa(t.height)&&{[r(`height`)]:t.height},[r(`stroke-width`)]:u,...l&&{[r(`class`)]:l},[r(`viewBox`)]:`0 0 ${i} ${a}`,...t.hasA11yProp===!1?{[r(`aria-hidden`)]:`true`}:{},...`attributes`in t&&t.attributes},e.node.map(e=>{let[n,i,a]=e,o=t.nonScalingStroke?{[r(`vector-effect`)]:`non-scaling-stroke`,...i}:i;return a?[n,o,a]:[n,o]})]}function Ea(e,t={}){return Ta(e,{...t,attributeNames:{...t.attributeNames,class:`className`,"stroke-width":`strokeWidth`,"stroke-linecap":`strokeLinecap`,"stroke-linejoin":`strokeLinejoin`,"vector-effect":`vectorEffect`}})}var Da=e=>{for(let t in e)if(t.startsWith(`aria-`)||t===`role`||t===`title`)return!0;return!1},Oa=(0,S.createContext)({}),ka=()=>(0,S.useContext)(Oa),Aa=(0,S.forwardRef)(({color:e,size:t,width:n,height:r,strokeWidth:i,absoluteStrokeWidth:a,nonScalingStroke:o,className:s=``,children:c,iconNode:l=[],icon:u={node:l,aliases:[],size:24},...d},f)=>{let{size:p=24,strokeWidth:m=2,absoluteStrokeWidth:h=!1,nonScalingStroke:g=!1,color:_=`currentColor`,className:v=``}=ka()??{},y=!!c||Da(d),[b,ee,te=[]]=Ea(u,{color:e??_,width:n??t??p,height:r??t??p,strokeWidth:i??m,absoluteStrokeWidth:a??h,nonScalingStroke:o??g,className:Sa(v,s),hasA11yProp:y,attributes:d});return(0,S.createElement)(b,{ref:f,...ee},[...te.map(([e,t])=>(0,S.createElement)(e,t)),...Array.isArray(c)?c:[c]])});function N(e,t=[],n=[]){let r=typeof e==`string`?ya(e,t,n):e,i=(0,S.forwardRef)(({className:e,...t},n)=>(0,S.createElement)(Aa,{ref:n,icon:r,className:e,...t}));return r.name&&(i.displayName=xa(r.name)),i}var ja={name:`arrow-down`,size:24,node:[[`path`,{d:`M12 5v14`,key:`s699le`}],[`path`,{d:`m19 12-7 7-7-7`,key:`1idqje`}]]};ja.node;var Ma=N(ja),Na={name:`arrow-up-down`,size:24,node:[[`path`,{d:`m21 16-4 4-4-4`,key:`f6ql7i`}],[`path`,{d:`M17 20V4`,key:`1ejh1v`}],[`path`,{d:`m3 8 4-4 4 4`,key:`11wl7u`}],[`path`,{d:`M7 4v16`,key:`1glfcx`}]]};Na.node;var Pa=N(Na),Fa={name:`arrow-up`,size:24,node:[[`path`,{d:`m5 12 7-7 7 7`,key:`hav0vg`}],[`path`,{d:`M12 19V5`,key:`x0mq9r`}]]};Fa.node;var Ia=N(Fa),La={name:`calendar`,size:24,node:[[`path`,{d:`M8 2v3`,key:`1ioesn`}],[`path`,{d:`M16 2v3`,key:`otl347`}],[`rect`,{x:`3`,y:`3`,width:`18`,height:`18`,rx:`2`,key:`h1oib`}],[`path`,{d:`M3 9h18`,key:`1pudct`}]]};La.node;var Ra=N(La),za={name:`check-check`,size:24,node:[[`path`,{d:`M18 6 7 17l-5-5`,key:`116fxf`}],[`path`,{d:`m22 10-7.5 7.5L13 16`,key:`ke71qq`}]]};za.node;var Ba=N(za),Va={name:`check`,size:24,node:[[`path`,{d:`M20 6 9 17l-5-5`,key:`1gmf2c`}]]};Va.node;var Ha=N(Va),Ua={name:`chevron-down`,size:24,node:[[`path`,{d:`m6 9 6 6 6-6`,key:`qrunsl`}]]};Ua.node;var Wa=N(Ua),Ga={name:`chevron-left`,size:24,node:[[`path`,{d:`m15 18-6-6 6-6`,key:`1wnfg3`}]]};Ga.node;var Ka=N(Ga),qa={name:`chevron-right`,size:24,node:[[`path`,{d:`m9 18 6-6-6-6`,key:`mthhwq`}]]};qa.node;var Ja=N(qa),Ya={name:`chevrons-left`,size:24,node:[[`path`,{d:`m11 17-5-5 5-5`,key:`13zhaf`}],[`path`,{d:`m18 17-5-5 5-5`,key:`h8a8et`}]]};Ya.node;var Xa=N(Ya),Za={name:`chevrons-right`,size:24,node:[[`path`,{d:`m6 17 5-5-5-5`,key:`xnjwq`}],[`path`,{d:`m13 17 5-5-5-5`,key:`17xmmf`}]]};Za.node;var Qa=N(Za),$a={name:`circle-check-big`,size:24,node:[[`path`,{d:`M21.801 10A10 10 0 1 1 17 3.335`,key:`yps3ct`}],[`path`,{d:`m9 11 3 3L22 4`,key:`1pflzl`}]],aliases:[`check-circle`]};$a.node;var eo=N($a),to={name:`circle-x`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`10`,key:`1mglay`}],[`path`,{d:`m15 9-6 6`,key:`1uzhvr`}],[`path`,{d:`m9 9 6 6`,key:`z0biqf`}]],aliases:[`x-circle`]};to.node;var no=N(to),ro={name:`clock`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`10`,key:`1mglay`}],[`path`,{d:`M12 6v6l4 2`,key:`mmk7yg`}]]};ro.node;var io=N(ro),ao={name:`columns-2`,size:24,node:[[`rect`,{width:`18`,height:`18`,x:`3`,y:`3`,rx:`2`,key:`afitv7`}],[`path`,{d:`M12 3v18`,key:`108xh3`}]],aliases:[`columns`]};ao.node;var oo=N(ao),so={name:`copy`,size:24,node:[[`rect`,{width:`14`,height:`14`,x:`8`,y:`8`,rx:`2`,ry:`2`,key:`17jyea`}],[`path`,{d:`M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2`,key:`zix9uf`}]]};so.node;var co=N(so),lo={name:`ellipsis`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`1`,key:`41hilf`}],[`circle`,{cx:`19`,cy:`12`,r:`1`,key:`1wjl8i`}],[`circle`,{cx:`5`,cy:`12`,r:`1`,key:`1pcz8c`}]],aliases:[`more-horizontal`]};lo.node;var uo=N(lo),fo={name:`external-link`,size:24,node:[[`path`,{d:`M15 3h6v6`,key:`1q9fwt`}],[`path`,{d:`M10 14 21 3`,key:`gplh6r`}],[`path`,{d:`M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6`,key:`a6xqqp`}]]};fo.node;var po=N(fo),mo={name:`eye-off`,size:24,node:[[`path`,{d:`M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49`,key:`ct8e1f`}],[`path`,{d:`M14.084 14.158a3 3 0 0 1-4.242-4.242`,key:`151rxh`}],[`path`,{d:`M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143`,key:`13bj9a`}],[`path`,{d:`m2 2 20 20`,key:`1ooewy`}]]};mo.node;var ho=N(mo),go={name:`funnel`,size:24,node:[[`path`,{d:`M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z`,key:`sc7q7i`}]],aliases:[`filter`]};go.node;var _o=N(go),vo={name:`info`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`10`,key:`1mglay`}],[`path`,{d:`M12 16v-4`,key:`1dtifu`}],[`path`,{d:`M12 8h.01`,key:`e9boi3`}]]};vo.node;var yo=N(vo),bo={name:`menu`,size:24,node:[[`path`,{d:`M4 5h16`,key:`1tepv9`}],[`path`,{d:`M4 12h16`,key:`1lakjw`}],[`path`,{d:`M4 19h16`,key:`1djgab`}]]};bo.node;var xo=N(bo),So={name:`mic`,size:24,node:[[`path`,{d:`M12 19v3`,key:`npa21l`}],[`path`,{d:`M19 10v2a7 7 0 0 1-14 0v-2`,key:`1vc78b`}],[`rect`,{x:`9`,y:`2`,width:`6`,height:`13`,rx:`3`,key:`s6n7sd`}]]};So.node;var Co=N(So),wo={name:`search`,size:24,node:[[`path`,{d:`m21 21-4.34-4.34`,key:`14j7rj`}],[`circle`,{cx:`11`,cy:`11`,r:`8`,key:`4ej97u`}]]};wo.node;var To=N(wo),Eo={name:`square`,size:24,node:[[`rect`,{width:`18`,height:`18`,x:`3`,y:`3`,rx:`2`,key:`afitv7`}]]};Eo.node;var Do=N(Eo),Oo={name:`triangle-alert`,size:24,node:[[`path`,{d:`m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3`,key:`wmoenq`}],[`path`,{d:`M12 9v4`,key:`juzpu7`}],[`path`,{d:`M12 17h.01`,key:`p32p05`}]],aliases:[`alert-triangle`]};Oo.node;var ko=N(Oo),Ao={name:`upload`,size:24,node:[[`path`,{d:`M12 3v12`,key:`1x0j5s`}],[`path`,{d:`m17 8-5-5-5 5`,key:`7q97r8`}],[`path`,{d:`M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4`,key:`ih7n3h`}]]};Ao.node;var jo=N(Ao),Mo={name:`wrench`,size:24,node:[[`path`,{d:`M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.106-3.105c.32-.322.863-.22.983.218a6 6 0 0 1-8.259 7.057l-7.91 7.91a1 1 0 0 1-2.999-3l7.91-7.91a6 6 0 0 1 7.057-8.259c.438.12.54.662.219.984z`,key:`1ngwbx`}]]};Mo.node;var No=N(Mo),Po={name:`x`,size:24,node:[[`path`,{d:`M18 6 6 18`,key:`1bl5f8`}],[`path`,{d:`m6 6 12 12`,key:`d8bk6v`}]]};Po.node;var Fo=N(Po),P={size:`1em`,"aria-hidden":!0},Io={name:`neutral`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--font-family-body":`Figtree, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,"--font-family-heading":`Figtree, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-background-surface":`light-dark(#ffffff, #262626)`,"--color-background-body":`light-dark(#f1f1f1, #1b1b1b)`,"--color-background-card":`light-dark(#ffffff, #1b1b1b)`,"--color-background-popover":`light-dark(#ffffff, #1b1b1b)`,"--color-background-muted":`light-dark(#f1f1f1, #1b1b1b)`,"--color-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-accent-muted":`light-dark(#f1f1f1, #262626)`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(#00000080, #000000CC)`,"--color-overlay-hover":`light-dark(#0000000D, #ffffff0D)`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#000000, #ffffff)`,"--color-text-secondary":`light-dark(#474747, #9e9e9e)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #111111)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-icon-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-icon-primary":`light-dark(#000000, #ffffff)`,"--color-icon-secondary":`light-dark(#6a6a6a, #9e9e9e)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#00490b, #a4d6a3)`,"--color-error":`light-dark(#76000c, #ffc4be)`,"--color-warning":`light-dark(#4b3900, #f8d36a)`,"--color-success-muted":`light-dark(#bce0bb, #90ca903D)`,"--color-error-muted":`light-dark(#ffc4be, #ff98903D)`,"--color-warning-muted":`light-dark(#fae19e, #e2b6233D)`,"--color-border":`light-dark(#00000014, #ffffff1A)`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(black, white)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#0074e2, #6d9cfe)`,"--astryx-theme-neutral-color-status-fill-success":`light-dark(#198100, #64af4c)`,"--astryx-theme-neutral-color-status-fill-warning":`#ffce2f`,"--astryx-theme-neutral-color-status-fill-error":`light-dark(#c9303a, #ff705d)`,"--astryx-theme-neutral-color-status-muted-accent":`light-dark(#b9d7ff, #88bcff3D)`,"--astryx-theme-neutral-color-on-tint-neutral":`light-dark(#fafafa4D, #0a0a0a4D)`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`light-dark(#fafafa1A, #0a0a0a1A)`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`light-dark(#fafafa33, #0a0a0a33)`,"--astryx-theme-neutral-color-destructive-overlay-hover":`light-dark(#ff7f770D, #ee736c0D)`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`light-dark(#ff7f771A, #ee736c1A)`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`,"--astryx-theme-neutral-color-status-fill-success":`neutral`,"--astryx-theme-neutral-color-status-fill-warning":`neutral`,"--astryx-theme-neutral-color-status-fill-error":`neutral`,"--astryx-theme-neutral-color-status-muted-accent":`neutral`,"--astryx-theme-neutral-color-on-tint-neutral":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-hover":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`neutral`},__localTokenLineage:[`neutral`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-error)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-destructive-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-destructive-overlay-pressed)`}},badge:{"variant:info":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`,color:`var(--color-on-accent)`},"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`,color:`var(--color-on-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`,color:`var(--color-on-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`,color:`var(--color-on-error)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`},"variant:accent":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`}},"avatar-status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},banner:{base:{"--color-neutral":`var(--astryx-theme-neutral-color-on-tint-neutral)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-on-tint-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-on-tint-overlay-pressed)`},"status:info":{"--color-accent-muted":`var(--astryx-theme-neutral-color-status-muted-accent)`,"--color-text-primary":`var(--color-text-blue)`,"--color-text-secondary":`var(--color-text-blue)`,"--color-accent":`var(--color-text-blue)`},"status:success":{"--color-text-primary":`var(--color-text-green)`,"--color-text-secondary":`var(--color-text-green)`,"--color-success":`var(--color-text-green)`},"status:warning":{"--color-text-primary":`var(--color-text-yellow)`,"--color-text-secondary":`var(--color-text-yellow)`,"--color-warning":`var(--color-text-yellow)`},"status:error":{"--color-text-primary":`var(--color-text-red)`,"--color-text-secondary":`var(--color-text-red)`,"--color-error":`var(--color-text-red)`}},"step-indicator":{"status:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"status:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"status:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"status:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`},"variant:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"variant:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`Figtree`,fallbacks:`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`},heading:{family:`Figtree`,fallbacks:`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:{close:(0,C.jsx)(Fo,{...P}),chevronDown:(0,C.jsx)(Wa,{...P}),chevronLeft:(0,C.jsx)(Ka,{...P}),chevronRight:(0,C.jsx)(Ja,{...P}),chevronsLeft:(0,C.jsx)(Xa,{...P}),chevronsRight:(0,C.jsx)(Qa,{...P}),check:(0,C.jsx)(Ha,{...P}),success:(0,C.jsx)(eo,{...P}),error:(0,C.jsx)(no,{...P}),warning:(0,C.jsx)(ko,{...P}),info:(0,C.jsx)(yo,{...P}),calendar:(0,C.jsx)(Ra,{...P}),clock:(0,C.jsx)(io,{...P}),externalLink:(0,C.jsx)(po,{...P}),menu:(0,C.jsx)(xo,{...P}),moreHorizontal:(0,C.jsx)(uo,{...P}),search:(0,C.jsx)(To,{...P}),upload:(0,C.jsx)(jo,{...P}),arrowUp:(0,C.jsx)(Ia,{...P}),arrowDown:(0,C.jsx)(Ma,{...P}),arrowsUpDown:(0,C.jsx)(Pa,{...P}),funnel:(0,C.jsx)(_o,{...P}),eyeSlash:(0,C.jsx)(ho,{...P}),viewColumns:(0,C.jsx)(oo,{...P}),copy:(0,C.jsx)(co,{...P}),checkDouble:(0,C.jsx)(Ba,{...P}),wrench:(0,C.jsx)(No,{...P}),stop:(0,C.jsx)(Do,{...P}),microphone:(0,C.jsx)(Co,{...P})}},Lo={name:`mlxcel`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--font-family-body":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-heading":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-background-surface":`light-dark(#ffffff, #181d27)`,"--color-background-body":`light-dark(#f3f4f6, #0e1117)`,"--color-background-card":`light-dark(#ffffff, #181d27)`,"--color-background-popover":`light-dark(#ffffff, #181d27)`,"--color-background-muted":`light-dark(#f8f8fa, #111722)`,"--color-accent":`light-dark(#245bd8, #245bd8)`,"--color-accent-muted":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(rgba(8, 10, 14, 0.42), rgba(8, 10, 14, 0.42))`,"--color-overlay-hover":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#151a24, #f5f7fb)`,"--color-text-secondary":`light-dark(#59647a, #aeb8c9)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#245bd8, #9dbdff)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #ffffff)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-icon-accent":`light-dark(#245bd8, #9dbdff)`,"--color-icon-primary":`light-dark(#151a24, #f5f7fb)`,"--color-icon-secondary":`light-dark(#59647a, #aeb8c9)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#137333, #9be6b0)`,"--color-error":`light-dark(#b42318, #ffb4a8)`,"--color-warning":`light-dark(#8a5200, #ffd28a)`,"--color-success-muted":`light-dark(#eaf7ee, rgba(25, 108, 55, 0.22))`,"--color-error-muted":`light-dark(#fff0ed, rgba(152, 43, 28, 0.24))`,"--color-warning-muted":`light-dark(#fff4d8, rgba(139, 92, 0, 0.22))`,"--color-border":`light-dark(rgba(33, 42, 61, 0.12), rgba(255, 255, 255, 0.12))`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(black, white)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`,"--focus-outline-color":`light-dark(#245bd8, #9dbdff)`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#245bd8, #245bd8)`,"--astryx-theme-neutral-color-status-fill-success":`light-dark(#137333, #9be6b0)`,"--astryx-theme-neutral-color-status-fill-warning":`#ffce2f`,"--astryx-theme-neutral-color-status-fill-error":`light-dark(#b42318, #ffb4a8)`,"--astryx-theme-neutral-color-status-muted-accent":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`,"--astryx-theme-neutral-color-on-tint-neutral":`light-dark(#fafafa4D, #0a0a0a4D)`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`light-dark(#fafafa1A, #0a0a0a1A)`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`light-dark(#fafafa33, #0a0a0a33)`,"--astryx-theme-neutral-color-destructive-overlay-hover":`light-dark(#ff7f770D, #ee736c0D)`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`light-dark(#ff7f771A, #ee736c1A)`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`,"--astryx-theme-neutral-color-status-fill-success":`neutral`,"--astryx-theme-neutral-color-status-fill-warning":`neutral`,"--astryx-theme-neutral-color-status-fill-error":`neutral`,"--astryx-theme-neutral-color-status-muted-accent":`neutral`,"--astryx-theme-neutral-color-on-tint-neutral":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-hover":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`neutral`},__localTokenLineage:[`neutral`,`mlxcel`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-error)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-destructive-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-destructive-overlay-pressed)`}},badge:{"variant:info":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`,color:`var(--color-on-accent)`},"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`,color:`var(--color-on-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`,color:`var(--color-on-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`,color:`var(--color-on-error)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`},"variant:accent":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`}},"avatar-status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},banner:{base:{"--color-neutral":`var(--astryx-theme-neutral-color-on-tint-neutral)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-on-tint-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-on-tint-overlay-pressed)`},"status:info":{"--color-accent-muted":`var(--astryx-theme-neutral-color-status-muted-accent)`,"--color-text-primary":`var(--color-text-blue)`,"--color-text-secondary":`var(--color-text-blue)`,"--color-accent":`var(--color-text-blue)`},"status:success":{"--color-text-primary":`var(--color-text-green)`,"--color-text-secondary":`var(--color-text-green)`,"--color-success":`var(--color-text-green)`},"status:warning":{"--color-text-primary":`var(--color-text-yellow)`,"--color-text-secondary":`var(--color-text-yellow)`,"--color-warning":`var(--color-text-yellow)`},"status:error":{"--color-text-primary":`var(--color-text-red)`,"--color-text-secondary":`var(--color-text-red)`,"--color-error":`var(--color-text-red)`}},"step-indicator":{"status:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"status:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"status:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"status:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`},"variant:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"variant:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`},heading:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:Io.icons},Ro={name:`mlxcel-glass`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--font-family-body":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-heading":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#6a6a6a, #9e9e9e)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-background-surface":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-body":`light-dark(#e6ebf3, #0b0d12)`,"--color-background-card":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-popover":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-muted":`light-dark(#f1f2f6, #232327)`,"--color-accent":`light-dark(#0058b9, #0a5bc0)`,"--color-accent-muted":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(rgba(8, 10, 14, 0.32), rgba(8, 10, 14, 0.32))`,"--color-overlay-hover":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#1d1d1f, #f5f5f7)`,"--color-text-secondary":`light-dark(#545458, #b8b8bc)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#0058b9, #6cb2ff)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #ffffff)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-icon-accent":`light-dark(#0058b9, #6cb2ff)`,"--color-icon-primary":`light-dark(#1d1d1f, #f5f5f7)`,"--color-icon-secondary":`light-dark(#545458, #b8b8bc)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#1b6f3c, #4cd964)`,"--color-error":`light-dark(#b3261e, #ff7b72)`,"--color-warning":`light-dark(#8a4f00, #ffb340)`,"--color-success-muted":`light-dark(rgba(52, 199, 89, 0.12), rgba(50, 215, 75, 0.14))`,"--color-error-muted":`light-dark(rgba(255, 59, 48, 0.10), rgba(255, 69, 58, 0.16))`,"--color-warning-muted":`light-dark(rgba(255, 149, 0, 0.12), rgba(255, 159, 10, 0.14))`,"--color-border":`light-dark(rgba(60, 60, 67, 0.14), rgba(255, 255, 255, 0.12))`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(black, white)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`,"--focus-outline-color":`light-dark(#0058b9, #6cb2ff)`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#0058b9, #0a5bc0)`,"--astryx-theme-neutral-color-status-fill-success":`light-dark(#1b6f3c, #4cd964)`,"--astryx-theme-neutral-color-status-fill-warning":`#ffce2f`,"--astryx-theme-neutral-color-status-fill-error":`light-dark(#b3261e, #ff7b72)`,"--astryx-theme-neutral-color-status-muted-accent":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`,"--astryx-theme-neutral-color-on-tint-neutral":`light-dark(#fafafa4D, #0a0a0a4D)`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`light-dark(#fafafa1A, #0a0a0a1A)`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`light-dark(#fafafa33, #0a0a0a33)`,"--astryx-theme-neutral-color-destructive-overlay-hover":`light-dark(#ff7f770D, #ee736c0D)`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`light-dark(#ff7f771A, #ee736c1A)`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`,"--astryx-theme-neutral-color-status-fill-success":`neutral`,"--astryx-theme-neutral-color-status-fill-warning":`neutral`,"--astryx-theme-neutral-color-status-fill-error":`neutral`,"--astryx-theme-neutral-color-status-muted-accent":`neutral`,"--astryx-theme-neutral-color-on-tint-neutral":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-hover":`neutral`,"--astryx-theme-neutral-color-on-tint-overlay-pressed":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-hover":`neutral`,"--astryx-theme-neutral-color-destructive-overlay-pressed":`neutral`},__localTokenLineage:[`neutral`,`mlxcel-glass`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-error)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-destructive-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-destructive-overlay-pressed)`}},badge:{"variant:info":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`,color:`var(--color-on-accent)`},"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`,color:`var(--color-on-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`,color:`var(--color-on-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`,color:`var(--color-on-error)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`},"variant:accent":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-accent)`}},"avatar-status-dot":{"variant:success":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:error":{backgroundColor:`var(--astryx-theme-neutral-color-status-fill-error)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},banner:{base:{"--color-neutral":`var(--astryx-theme-neutral-color-on-tint-neutral)`,"--color-overlay-hover":`var(--astryx-theme-neutral-color-on-tint-overlay-hover)`,"--color-overlay-pressed":`var(--astryx-theme-neutral-color-on-tint-overlay-pressed)`},"status:info":{"--color-accent-muted":`var(--astryx-theme-neutral-color-status-muted-accent)`,"--color-text-primary":`var(--color-text-blue)`,"--color-text-secondary":`var(--color-text-blue)`,"--color-accent":`var(--color-text-blue)`},"status:success":{"--color-text-primary":`var(--color-text-green)`,"--color-text-secondary":`var(--color-text-green)`,"--color-success":`var(--color-text-green)`},"status:warning":{"--color-text-primary":`var(--color-text-yellow)`,"--color-text-secondary":`var(--color-text-yellow)`,"--color-warning":`var(--color-text-yellow)`},"status:error":{"--color-text-primary":`var(--color-text-red)`,"--color-text-secondary":`var(--color-text-red)`,"--color-error":`var(--color-text-red)`}},"step-indicator":{"status:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"status:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"status:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"status:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`},"variant:accent":{"--color-accent":`var(--astryx-theme-neutral-color-status-fill-accent)`},"variant:success":{"--color-success":`var(--astryx-theme-neutral-color-status-fill-success)`},"variant:warning":{"--color-warning":`var(--astryx-theme-neutral-color-status-fill-warning)`},"variant:error":{"--color-error":`var(--astryx-theme-neutral-color-status-fill-error)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`},heading:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:Io.icons},zo={en:`en`,ko:`ko-KR`},Bo=null;function Vo(){return Bo??=Promise.all([ee(()=>import(`./ko-KR.generated-CGvBKrxJ.js`),[],import.meta.url),ee(()=>import(`./ko-KR-C8OSSaum.js`),[],import.meta.url)]).then(([e,t])=>({"ko-KR":{...e.default,...t.default}})),Bo}var Ho={ko:Vo},Uo={mlxcel:Lo,glass:Ro};function Wo(e){let[t,n]=(0,S.useState)(null);(0,S.useEffect)(()=>{let r=Ho[e.locale];if(!r||t?.[zo[e.locale]])return;let i=!0;return r().then(e=>{i&&n(e)},()=>void 0),()=>{i=!1}},[e.locale,t]);let r=(0,S.useMemo)(()=>t??{},[t]);return(0,C.jsx)(mr,{theme:Uo[e.family],mode:e.scheme,children:(0,C.jsx)(Te,{locale:zo[e.locale],messages:r,children:e.children})})}var Go=0,Ko=0,qo=new Set;function Jo(e){return qo.add(e),()=>qo.delete(e)}function F(){Ko+=1,Go=Ko;for(let e of qo)e()}function I(){return(0,S.useSyncExternalStore)(Jo,()=>Go)}function L(){if(Go===0)return!1;Go=0;for(let e of qo)e();return!0}var Yo=`(min-width: 1100px)`,Xo=null;function Zo(){return typeof window.matchMedia==`function`?(Xo?.matchMedia!==window.matchMedia&&(Xo={matchMedia:window.matchMedia,list:window.matchMedia(Yo)}),Xo.list):null}function Qo(e){let t=Zo();return t?(t.addEventListener(`change`,e),()=>t.removeEventListener(`change`,e)):()=>void 0}function $o(){return(0,S.useSyncExternalStore)(Qo,()=>Zo()?.matches??!0)}function es(e,t,n){let r=e.filter(e=>e.runtime.server_instance_id===t.server_instance_id&&e.runtime.model_id===t.model_id&&e.receivedAt>n-3e5&&e.receivedAt<=n),i=r.at(-1);return i!==void 0&&Math.floor(i.receivedAt/2e3)===Math.floor(n/2e3)&&r.pop(),[...r,{receivedAt:n,runtime:t}].slice(-150)}function ts(e,t){let n=[...e.values()].sort((e,t)=>t.updated_at.localeCompare(e.updated_at)),r=n.filter(e=>!R(e)),i=n.filter(e=>R(e)&&Date.parse(e.updated_at)>t-36e5).slice(0,200);return new Map([...r,...i].map(e=>[e.operation_id,e]))}function R(e){return[`succeeded`,`failed`,`cancelled`].includes(e.state)}var ns=`webui.ui-api.v1`;function rs(){return{schemaVersion:ns,auth:{status:`signed-out`,tokenPresent:!1},connection:`idle`,bootstrap:null,catalog:[],catalogSequence:null,operations:new Map,runtimes:new Map,runtimeHistory:[],selectedModelId:null,serverInstanceId:null,lastEventId:null,lastSequence:null,lastUpdatedAt:null,lastSuccessfulAt:null,error:null,pendingReconciliations:new Map,resourceFences:{catalog:null,operationsSnapshot:null,operations:new Map,models:new Map,runtimes:new Map}}}function is(e,t){return t.type===`login-start`?{...e,auth:{status:`authenticating`,tokenPresent:!0},connection:`bootstrapping`,error:null}:t.type===`login-success`?as(e,t.bootstrap,t.now):t.type===`logout`?{...rs(),lastUpdatedAt:t.now}:t.type===`select-model`?{...e,selectedModelId:t.modelId,runtimeHistory:[]}:t.type===`catalog`?os(e,t.response,t.now):t.type===`operations-snapshot`?ss(e,t.response,t.now):t.type===`operation`?cs(e,t.operation,t.sequence,t.now):t.type===`runtime`?ls(e,t.runtime,t.sequence,t.now):t.type===`event`?us(e,t.event,t.now):t.type===`pending`?{...e,pendingReconciliations:_s(e.pendingReconciliations,t.item.idempotencyKey,t.item)}:t.type===`reconciled`?{...e,pendingReconciliations:vs(e.pendingReconciliations,t.idempotencyKey)}:{...e,connection:t.connection,error:t.error??null,lastUpdatedAt:t.now}}function as(e,t,n){let r=t.server.server_instance_id;return e.serverInstanceId!==null&&e.serverInstanceId!==r?{...rs(),auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n}:{...e,auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n,error:null}}function os(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return ps(e,t.server_instance_id,n);if(e.catalogSequence!==null&&t.snapshot_sequence<e.catalogSequence)return e;let r=e.catalogSequence===t.snapshot_sequence;return ms({...e,connection:e.connection===`bootstrapping`?`ready`:e.connection,catalog:hs(r?e.catalog:[],t),catalogSequence:t.snapshot_sequence,serverInstanceId:t.server_instance_id,lastSequence:gs({...e.resourceFences,catalog:t.snapshot_sequence}),error:null,resourceFences:{...e.resourceFences,catalog:t.snapshot_sequence}},n)}function ss(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return ps(e,t.server_instance_id,n);if(e.resourceFences.operationsSnapshot!==null&&t.snapshot_sequence<e.resourceFences.operationsSnapshot)return e;let r=e.resourceFences.operationsSnapshot===t.snapshot_sequence,i=r?new Map(e.operations):new Map,a=r?new Map(e.resourceFences.operations):new Map;for(let e of t.items)i.set(e.operation_id,e),a.set(e.operation_id,t.snapshot_sequence);let o=ts(i,n),s={...e.resourceFences,operationsSnapshot:t.snapshot_sequence,operations:new Map([...a].filter(([e])=>o.has(e)))};return ms({...e,operations:o,serverInstanceId:t.server_instance_id,lastSequence:gs(s),error:null,resourceFences:s},n)}function cs(e,t,n,r){if(n!==null&&e.resourceFences.operationsSnapshot!==null&&n<=e.resourceFences.operationsSnapshot)return e;let i=e.resourceFences.operations.get(t.operation_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=ts(_s(e.operations,t.operation_id,t),r),o=n===null?e.resourceFences.operations:_s(e.resourceFences.operations,t.operation_id,n),s={...e.resourceFences,operations:new Map([...o].filter(([e])=>a.has(e)))};return ms({...e,operations:a,lastSequence:gs(s),error:null,resourceFences:s},r)}function ls(e,t,n,r){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return ps(e,t.server_instance_id,r);let i=e.resourceFences.runtimes.get(t.model_id);if(n!==null&&i!==void 0&&n<i)return e;let a=n===null?e.resourceFences.runtimes:_s(e.resourceFences.runtimes,t.model_id,n),o={...e.resourceFences,runtimes:a};return ms({...e,runtimes:_s(e.runtimes,t.model_id,t),runtimeHistory:t.model_id===e.selectedModelId?es(e.runtimeHistory,t,r):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastSequence:gs(o),error:null,resourceFences:o},r)}function us(e,t,n){if(t.schema_version!==`webui.ui-api.v1`)return{...e,connection:`schema-mismatch`,error:{code:`schema_mismatch`,message:`Unsupported WebUI schema ${t.schema_version}`,retryable:!1},lastUpdatedAt:n};if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return ps(e,t.server_instance_id,n);if(t.type===`heartbeat`)return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:e.connection===`polling`?`polling`:`streaming`,lastUpdatedAt:n};if(t.type===`server_restart`||t.type===`gap`||t.type===`reset`)return fs(e,t,n);if(t.type===`operation`)return{...cs(e,t.payload.operation,t.sequence,n),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`};if(t.type===`model_revision`)return z(e,t,n);if(t.type===`runtime`)return ds(e,t,n);if(t.type===`snapshot`){let r={...e.resourceFences,catalog:t.payload.catalog_changed?t.sequence:e.resourceFences.catalog};return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:gs(r),connection:`streaming`,lastUpdatedAt:n,resourceFences:r}}return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`,lastUpdatedAt:n}}function z(e,t,n){let r=t.payload.model_id,i=e.resourceFences.models.get(r);if(i!==void 0&&t.sequence<=i)return e;let a=e.catalog.map(e=>e.identity.id===r&&t.payload.revision>=e.identity.revision?{...e,identity:{...e.identity,revision:t.payload.revision},lifecycle:t.payload.lifecycle}:e),o={...e.resourceFences,models:_s(e.resourceFences.models,r,t.sequence)};return ms({...e,catalog:a,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:gs(o),connection:`streaming`,resourceFences:o},n)}function ds(e,t,n){let r=t.payload.runtime,i=e.resourceFences.runtimes.get(r.model_id);if(i!==void 0&&t.sequence<=i)return e;let a={...e.resourceFences,runtimes:_s(e.resourceFences.runtimes,r.model_id,t.sequence)};return ms({...e,runtimes:_s(e.runtimes,r.model_id,r),runtimeHistory:r.model_id===e.selectedModelId?es(e.runtimeHistory,r,n):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:gs(a),connection:`streaming`,resourceFences:a},n)}function fs(e,t,n){let r=t.type===`server_restart`?null:e.lastSuccessfulAt;return{...rs(),auth:e.auth,connection:`stale`,serverInstanceId:t.server_instance_id,selectedModelId:e.selectedModelId,lastEventId:t.event_id,error:{code:t.type,message:t.payload.reason,retryable:t.payload.resnapshot},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n,lastSuccessfulAt:r}}function ps(e,t,n){return{...rs(),auth:e.auth,connection:`stale`,serverInstanceId:t,selectedModelId:e.selectedModelId,error:{code:`server_restarted`,message:`The mlxcel server restarted; refresh the authoritative snapshot before continuing.`,retryable:!0},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n}}function ms(e,t){return{...e,lastUpdatedAt:t,lastSuccessfulAt:t}}function hs(e,t){if(e.length>0){let n=new Map(e.map(e=>[e.identity.id,e]));for(let e of t.items)n.set(e.identity.id,e);return Array.from(n.values()).sort((e,t)=>e.identity.display_name.localeCompare(t.identity.display_name))}return t.items}function gs(e){let t=[e.catalog,e.operationsSnapshot,...e.operations.values(),...e.models.values(),...e.runtimes.values()].filter(e=>e!==null);return t.length===0?null:Math.min(...t)}function _s(e,t,n){let r=new Map(e);return r.set(t,n),r}function vs(e,t){let n=new Map(e);return n.delete(t),n}var ys=2e3,bs=1e4,xs=3e4,Ss=6e4,Cs=6e4,ws=new Set([`snapshot`,`model_revision`]),Ts=class{client;dispatch;getSnapshot;clock;visibility;random;timer=null;observationTimer=null;inflight=null;eventAbort=null;stopped=!0;failures=0;generation=0;needsFullSnapshot=!0;lastFullSnapshotAt=null;unsubscribe;constructor(e){this.client=e.client,this.dispatch=e.dispatch,this.getSnapshot=e.getSnapshot,this.clock=e.clock??js,this.visibility=e.visibility??Ms,this.random=e.random??Math.random,this.unsubscribe=this.visibility.subscribe(()=>this.visibilityChanged())}start(){this.stopped&&(this.stopped=!1,this.generation+=1,this.needsFullSnapshot=!0,this.reschedule(0))}stop(){this.generation+=1,this.stopped=!0,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null}selectionChanged(){this.cancelObservation(),this.reschedule(0)}fullSnapshotDue(e){let t=this.getSnapshot();return t.bootstrap===null||t.catalogSequence===null||this.needsFullSnapshot||this.lastFullSnapshotAt===null||this.eventAbort===null?!0:e-this.lastFullSnapshotAt>=Cs}cancelObservation(){this.generation+=1,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null}visibilityChanged(){this.cancelObservation(),this.needsFullSnapshot=!0,!this.stopped&&(this.dispatch({type:`connection`,connection:`stale`,now:this.clock.now()}),this.reschedule(0))}dispose(){this.stop(),this.unsubscribe()}async resnapshot(){this.needsFullSnapshot=!0,await this.refresh()}async refresh(){if(this.inflight!==null)return;let e=new AbortController;this.inflight=e;let t=this.generation,n=this.clock.now(),r=this.clock.setTimeout(()=>{e.abort(),this.generation===t&&this.dispatch({type:`connection`,connection:`stale`,error:{code:`observation_timeout`,message:`Runtime observation timed out; retrying.`,retryable:!0},now:this.clock.now()})},bs);this.observationTimer=r;try{if(this.getSnapshot().auth.status===`signed-out`)return;let r=this.fullSnapshotDue(n),i=null,a=[];if(r){if(i=await this.client.bootstrap(e.signal),!this.isCurrent(e,t)||this.getSnapshot().auth.status===`signed-out`||(this.dispatch({type:`login-success`,bootstrap:i,now:n}),a=await this.catalogSnapshot(e.signal),!this.isCurrent(e,t)))return;for(let e of a)this.dispatch({type:`catalog`,response:e,now:n})}let o=await this.operationSnapshot(e.signal);if(!this.isCurrent(e,t))return;let s=a.at(-1),c=o[0];if(s!==void 0&&c!==void 0&&c.server_instance_id!==s.server_instance_id){this.dispatch({type:`connection`,connection:`stale`,error:{code:`snapshot_mismatch`,message:`Operation and catalog snapshots came from different server instances.`,retryable:!0},now:n});return}let l=o.flatMap(e=>(this.dispatch({type:`operations-snapshot`,response:e,now:n}),e.items)),u=this.getSnapshot().selectedModelId,d=r&&i!==null&&u!==null&&a.every(e=>e.server_instance_id===i.server.server_instance_id)&&!a.some(e=>e.items.some(e=>e.identity.id===u));if(d&&this.dispatch({type:`select-model`,modelId:null}),d||await this.refreshSelectedRuntime(e.signal,t),r&&(this.needsFullSnapshot=!1,this.lastFullSnapshotAt=n),!this.isCurrent(e,t))return;let f=this.reconcilePending(l,n);this.failures=0,f||this.dispatch({type:`connection`,connection:`ready`,now:n}),this.eventAbort===null&&!this.visibility.hidden()&&this.startEvents()}catch(n){!e.signal.aborted&&this.generation===t&&(this.failures+=1,this.dispatch({type:`connection`,connection:Ds(n),error:Os(n),now:this.clock.now()}))}finally{this.clock.clearTimeout(r),this.observationTimer===r&&(this.observationTimer=null),this.inflight===e&&(this.inflight=null),!this.stopped&&this.generation===t&&this.reschedule(this.pollDelay())}}noteUnknownPost(e){this.dispatch({type:`pending`,item:e}),this.reschedule(0)}startEvents(){this.eventAbort?.abort();let e=new AbortController;this.eventAbort=e;let t=this.generation,n=Es(this.getSnapshot());this.client.events({onEvent:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.handleEvent(n)},onRetryAfter:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.reschedule(n)}},e.signal,n).then(()=>{this.generation===t&&(this.eventAbort===e&&(this.eventAbort=null),!e.signal.aborted&&!this.stopped&&this.reschedule(this.backoffDelay()))}).catch(n=>{e.signal.aborted||this.stopped||this.generation!==t||(this.failures+=1,this.eventAbort=null,this.dispatch({type:`connection`,connection:Ds(n),error:Os(n),now:this.clock.now()}),this.reschedule(this.backoffDelay()))})}handleEvent(e){this.dispatch({type:`event`,event:e,now:this.clock.now()}),ws.has(e.type)&&(this.needsFullSnapshot=!0),(e.type===`server_restart`||e.type===`gap`||e.type===`reset`)&&e.payload.resnapshot&&(this.needsFullSnapshot=!0,this.eventAbort?.abort(),this.eventAbort=null,this.refresh())}async catalogSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.catalog(n===void 0?{}:{cursor:n},e);if(r===null?r=i:As(`catalog`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operationSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null?r=i:As(`operations`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async refreshSelectedRuntime(e,t){let n=this.getSnapshot().selectedModelId;if(n===null)return;let r;try{r=await this.client.runtime(n,e)}catch(e){throw typeof e==`object`&&e&&`status`in e&&e.status===404?new ks(`Selected model changed during observation; refreshing the catalog.`):e}e.aborted||this.generation!==t||this.getSnapshot().selectedModelId===n&&this.dispatch({type:`runtime`,runtime:r,sequence:r.snapshot_sequence,now:this.clock.now()})}reconcilePending(e,t){let n=!1,r=new Set(e.map(e=>e.operation_id));for(let e of this.getSnapshot().pendingReconciliations.values())e.operationId!==null&&r.has(e.operationId)?this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}):t-e.createdAt>=Ss&&(this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}),this.dispatch({type:`connection`,connection:`stale`,error:{code:`unknown_post_unresolved`,message:`A previous control request could not be matched to an operation after reconciliation.`,retryable:!0},now:t}),n=!0);return n}reschedule(e){this.stopped||this.visibility.hidden()||(this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=this.clock.setTimeout(()=>{this.timer=null,this.refresh()},e))}pollDelay(){return ys}backoffDelay(){let e=Math.min(xs,500*2**Math.min(6,this.failures));return Math.round(e/2+this.random()*(e/2))}isCurrent(e,t){return this.inflight===e&&this.generation===t&&!e.signal.aborted}};function Es(e){let t=gs(e.resourceFences);if(e.serverInstanceId!==null&&t!==null)return{lastEventId:null,serverInstanceId:e.serverInstanceId,afterSequence:t};if(e.lastEventId!==null)return{lastEventId:e.lastEventId,serverInstanceId:null,afterSequence:null}}function Ds(e){if(e instanceof ks)return`stale`;let t=typeof e==`object`&&e&&`status`in e&&typeof e.status==`number`?e.status:null;return t===401||e instanceof Error&&/401|unauthorized/i.test(e.message)?`unauthorized`:t===403||e instanceof Error&&/403|forbidden/i.test(e.message)?`forbidden`:e instanceof DOMException&&e.name===`AbortError`?`stale`:`offline`}function Os(e){return e instanceof ks?{code:e.code,message:e.message,retryable:!0}:{code:`sync_error`,message:e instanceof Error?e.message.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`):`Unknown WebUI client error`,retryable:!0}}var ks=class extends Error{code=`snapshot_mismatch`;constructor(e){super(e),this.name=`SnapshotConsistencyError`}};function As(e,t,n){if(n.server_instance_id!==t.server_instance_id)throw new ks(`${e} pagination crossed a server restart; refresh the authoritative snapshot.`);if(n.snapshot_sequence!==t.snapshot_sequence)throw new ks(`${e} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`)}var js={setTimeout:(e,t)=>globalThis.setTimeout(e,t),clearTimeout:e=>globalThis.clearTimeout(e),now:()=>Date.now()},Ms={hidden:()=>typeof document<`u`&&document.hidden,subscribe:e=>typeof document>`u`?()=>void 0:(document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e))},Ns=S.createContext(null),Ps=S.createContext(null);function Fs({children:e,apiBase:t,fetchImpl:n}){let[r,i]=S.useReducer(is,void 0,rs),a=S.useRef(r);a.current=r;let o=S.useRef(0),s=S.useRef(null),c=S.useMemo(()=>new pi({apiBase:t,fetchImpl:n,onUnauthorized:()=>{o.current+=1,s.current?.stop(),i({type:`logout`,now:Date.now()})}}),[t,n]);S.useEffect(()=>{let e=new Ts({client:c,dispatch:i,getSnapshot:()=>a.current});return s.current=e,()=>{e.dispose(),s.current=null,c.abortAll()}},[c]);let l=S.useMemo(()=>({getTokenCount:(e,t,n)=>c.tokenCount(d(e),t,n),getSettings:(e,t)=>c.settings(d(e),t),patchSettings:(e,t,n)=>c.patchSettings(d(e),t,n),getModelProps:(e,t)=>c.modelProps(d(e),t),login:async e=>{o.current+=1;let t=o.current;c.abortAll(),c.setBearerToken(e),i({type:`login-start`});try{let e=await c.bootstrap();if(o.current!==t)return;i({type:`login-success`,bootstrap:e,now:Date.now()}),s.current?.start()}catch(e){throw o.current===t&&(c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})),e}},logout:()=>{o.current+=1,s.current?.stop(),c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})},refresh:async()=>{await s.current?.resnapshot()},selectModel:e=>{i({type:`select-model`,modelId:e}),s.current?.selectionChanged()},loadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},unloadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},downloadModel:async e=>{await u(`download`,e.idempotency_key,void 0,()=>c.download(e))},removeModel:async e=>{await u(`removal`,e.idempotency_key,e.model_id,()=>c.removeModel(e))},refreshRuntime:async e=>{let t=o.current,n=await c.runtime(e);return o.current===t&&i({type:`runtime`,runtime:n,sequence:n.snapshot_sequence,now:Date.now()}),n},refreshCatalog:async e=>{await u(`catalog-refresh`,e,void 0,()=>c.refreshCatalog(e))},cancelOperation:async e=>{let t=o.current;await c.cancelOperation(e),o.current===t&&await s.current?.resnapshot()},streamChatCompletions:async(e,t,n,r)=>{await c.chatCompletions(d(e),t,n,r)},streamResponses:async(e,t,n,r)=>{await c.responses(d(e),t,n,r)}}),[c]);async function u(e,t,n,r){let i=o.current;try{let a=await r();if(o.current!==i)return;s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:a.operation_id,modelId:n,createdAt:Date.now()})}catch(r){throw o.current===i&&!(r instanceof di)&&s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:null,modelId:n,createdAt:Date.now()}),r}}function d(e){let t=a.current.catalog.find(t=>t.identity.id===e);if(t===void 0)throw Error(`Selected model is not present in the catalog snapshot.`);if(t.identity.inference_id.length===0)throw Error(`Selected model does not expose an inference model id.`);return t.identity.inference_id}return(0,C.jsx)(Ps.Provider,{value:l,children:(0,C.jsx)(Ns.Provider,{value:r,children:e})})}function Is(){let e=S.useContext(Ns);if(e===null)throw Error(`useWebUi must be used within WebUiProvider.`);return e}function Ls(){let e=S.useContext(Ps);if(e===null)throw Error(`useWebUiActions must be used within WebUiProvider.`);return e}var Rs=Object.freeze({conversations:50,turnsPerConversation:200,totalTurns:1e3,textCharacters:262144,jsonBytes:16777216,imageBytes:8388608,imagesPerTurn:8,toolsPerTurn:128}),zs=`mlxcel.webui.chat.v1`,Bs=`history`,Vs=`conversations`,Hs=1,Us=2,Ws=[`id`,`modelId`,`modelRevision`,`inferenceId`,`modelName`,`prompt`,`content`,`reasoning`,`tools`,`status`,`finishReason`,`usage`,`ttftMs`,`elapsedMs`,`error`,`parameters`,`images`],Gs=[...Ws,`startedAt`],Ks=new Set([`temperature`,`top_p`,`top_k`,`min_p`,`max_tokens`,`seed`,`repetition_penalty`,`presence_penalty`,`frequency_penalty`]),qs=class extends Error{constructor(e=`Conversation history is invalid or exceeds the local history limits.`){super(e),this.name=`HistoryValidationError`}},Js=class extends Error{constructor(e,t){super(e,t),this.name=`HistoryStorageError`}};function B(){throw new qs}function Ys(e,t){if(!e||typeof e!=`object`||Array.isArray(e))return B();let n=Object.getPrototypeOf(e);if(n!==Object.prototype&&n!==null)return B();let r=Object.keys(e);return r.length!==t.length||r.some(e=>!t.includes(e))?B():e}function Xs(e,t=Rs.textCharacters){return typeof e!=`string`||e.length>t?B():e}function Zs(e){let t=Xs(e,512);return!t||Array.from(t).some(e=>e.charCodeAt(0)<32||e.charCodeAt(0)===127)?B():t}function Qs(e,t=!1){return typeof e!=`number`||!Number.isFinite(e)||e<0||t&&!Number.isSafeInteger(e)?B():e}var $s=864e13;function ec(e){let t=Qs(e,!0);return t>$s?B():t}function tc(e,t){return e===null?null:t(e)}function nc(e,t){return!Array.isArray(e)||e.length>t?B():e}function rc(e){return new TextEncoder().encode(e).byteLength}function ic(e,t,n){let r=0,i=0,a=0,o=new Set;return nc(e,Rs.conversations).map(e=>{let s=Ys(e,[`id`,`title`,`systemPrompt`,`turns`,`updatedAt`]),c=Zs(s.id);if(o.has(c))return B();o.add(c);let l=new Set,u=nc(s.turns,Rs.turnsPerConversation).map(e=>{if(++r>Rs.totalTurns)return B();let o=Ys(e,n===1?Ws:Gs),s=Zs(o.id);if(l.has(s))return B();l.add(s);let c=Xs(o.status,32);if(![`streaming`,`complete`,`cancelled`,`interrupted`,`error`].includes(c))return B();let u={};if(!o.parameters||typeof o.parameters!=`object`||Array.isArray(o.parameters)||![Object.prototype,null].includes(Object.getPrototypeOf(o.parameters)))return B();for(let[e,t]of Object.entries(o.parameters)){if(!Ks.has(e)||typeof t!=`number`||!Number.isFinite(t))return B();u[e]=t}let d=new Set,f=nc(o.tools,Rs.toolsPerTurn).map(e=>{let t=Ys(e,[`index`,`id`,`name`,`arguments`]),n=Qs(t.index,!0);return d.has(n)?B():(d.add(n),{index:n,id:Xs(t.id,512),name:Xs(t.name,512),arguments:Xs(t.arguments)})}),p=nc(o.images,Rs.imagesPerTurn).map(e=>{let t=Ys(e,[`name`,`type`,`dataUrl`]),n=Xs(t.type,64);if(![`image/png`,`image/jpeg`,`image/webp`].includes(n))return B();let r=Xs(t.dataUrl,Rs.imageBytes*2),a=`data:${n};base64,`;if(!r.startsWith(a))return B();let o=r.slice(a.length);return!o||o.length%4!=0||!/^[A-Za-z0-9+/]*={0,2}$/u.test(o)||(i+=o.length*3/4-(o.endsWith(`==`)?2:+!!o.endsWith(`=`)),i>Rs.imageBytes)?B():{name:Xs(t.name,512),type:n,dataUrl:r}}),m={id:s,modelId:Zs(o.modelId),modelRevision:Qs(o.modelRevision,!0),inferenceId:Zs(o.inferenceId),modelName:Xs(o.modelName,512),prompt:Xs(o.prompt),content:Xs(o.content),reasoning:Xs(o.reasoning),tools:f,status:c===`streaming`?`interrupted`:c,finishReason:tc(o.finishReason,Xs),usage:tc(o.usage,e=>{let t=Ys(e,[`prompt_tokens`,`completion_tokens`,`total_tokens`]);return{prompt_tokens:Qs(t.prompt_tokens,!0),completion_tokens:Qs(t.completion_tokens,!0),total_tokens:Qs(t.total_tokens,!0)}}),ttftMs:tc(o.ttftMs,Qs),elapsedMs:tc(o.elapsedMs,Qs),error:tc(o.error,Xs),parameters:u,images:t.includeImages?p:[],startedAt:n===1?0:ec(o.startedAt)};return m.status===`complete`&&(!m.finishReason||!m.content&&!m.reasoning&&!m.tools.length)||(a+=rc(JSON.stringify(m)),a>Rs.jsonBytes)?B():m});return{id:c,title:Xs(s.title,512),systemPrompt:Xs(s.systemPrompt),turns:u,updatedAt:Qs(s.updatedAt,!0)}})}function ac(e,t={}){return oc(e,t,Us)}function oc(e,t,n){let r=ic(e,t,n);return rc(JSON.stringify(r))>Rs.jsonBytes?B():r}function sc(e,t={}){let n=JSON.stringify({version:Us,conversations:ac(e,t)});return rc(n)>Rs.jsonBytes?B():n}function cc(e,t={}){if(rc(e)>Rs.jsonBytes)return B();let n;try{n=JSON.parse(e)}catch{return B()}let r=Ys(n,[`version`,`conversations`]);return r.version!==1&&r.version!==2?B():oc(r.conversations,t,r.version)}function lc(e){return new Js(e instanceof DOMException&&e.name===`QuotaExceededError`?`Local history storage is full. Export or clear history, or disable persistence; the current conversation remains in memory.`:`Local history storage is unavailable. The current conversation remains in memory.`,{cause:e})}function uc(e={}){let t=!1,n=!1,r=0,i,a,o=new Set,s=()=>{let t=e.indexedDB??globalThis.indexedDB;if(!t)throw new Js(`This browser does not support local history storage.`);return t},c=()=>{r++;for(let e of o)try{e.abort()}catch{}o.clear(),i?.close(),i=void 0,a=void 0},l=()=>{if(i)return Promise.resolve(i);if(a)return a;let e=r;return a=new Promise((n,a)=>{let o=s().open(zs,Hs);o.onupgradeneeded=()=>{o.result.objectStoreNames.contains(Bs)||o.result.createObjectStore(Bs)},o.onerror=()=>a(lc(o.error));let l=!1;o.onblocked=()=>{l=!0,a(new Js(`Close other WebUI tabs to access local history.`))},o.onsuccess=()=>{if(l||!t||e!==r){o.result.close(),a(new Js(`Local history persistence was disabled.`));return}i=o.result,i.onversionchange=c,n(i)}}),a.catch(t=>{throw e===r&&(a=void 0),t})};async function u(e,i){if(!t||n)return;let a=r;try{let s=await l();return!t||n||r!==a?void 0:await new Promise((t,n)=>{let r=s.transaction(Bs,e);o.add(r);let a=i(r.objectStore(Bs));r.oncomplete=()=>{o.delete(r),t(a.result)},r.onabort=r.onerror=()=>{o.delete(r),n(lc(r.error??a.error))}})}catch(e){throw e instanceof Js?e:lc(e)}}return{setEnabled(e){t!==e&&(t=e,e||c())},async load(e={}){let t=await u(`readonly`,e=>e.get(Vs));return t===void 0?[]:cc(Xs(t,Rs.jsonBytes),e)},async save(e,n={}){if(!t)return;let r=sc(e,n);await u(`readwrite`,e=>e.put(r,Vs))},async clear(){if(n)throw new Js(`Local history is already being cleared.`);n=!0;let e=!1;c();try{await new Promise((t,r)=>{let i=s().deleteDatabase(zs);i.onsuccess=()=>{n=!1,t()},i.onerror=()=>{n=!1,r(lc(i.error))},i.onblocked=()=>{e=!0,r(new Js(`Close other WebUI tabs to clear local history.`))}})}catch(t){throw e||(n=!1),t instanceof Js?t:lc(t)}},close:c}}var dc=[],fc=0,pc=0,mc=0,hc=new Set;function gc(e){return hc.add(e),()=>{hc.delete(e)}}function _c(){return fc}function vc(e){return{id:crypto.randomUUID(),title:e,systemPrompt:``,turns:[],updatedAt:Date.now()}}var yc=new WeakMap;function bc(e){let t=yc.get(e);if(t!==void 0)return t;let n=new TextEncoder().encode(JSON.stringify(e)).byteLength;return yc.set(e,n),n}function xc(e){return e.length>Rs.conversations||e.reduce((e,t)=>e+t.turns.length,0)>Rs.totalTurns?!1:2+Math.max(0,e.length-1)+e.reduce((e,t)=>e+new TextEncoder().encode(JSON.stringify({...t,turns:[]})).byteLength+t.turns.reduce((e,t)=>e+bc(t),0)+Math.max(0,t.turns.length-1),0)<=Rs.jsonBytes}function Sc(){for(let e of hc)e()}function Cc(e){dc=e,Sc()}function wc(e){if(!xc(e))throw Error(`Conversation memory limit exceeded.`);fc++,pc=0,Cc(e)}function Tc(){mc++,pc=mc,Sc()}function Ec(){return(0,S.useSyncExternalStore)(gc,()=>pc)}function Dc(){let e=pc!==0;return pc=0,e}function Oc(e){let t=dc.some(t=>t.id===e.id)?dc.map(t=>t.id===e.id?e:t):[...dc,e];return xc(t)?(Cc(t),!0):!1}function kc(){return(0,S.useSyncExternalStore)(gc,()=>dc)}var Ac=e=>[`succeeded`,`failed`,`cancelled`].includes(e.state),jc=e=>e.auth.status===`authenticated`&&e.bootstrap!==null&&e.catalogSequence!==null&&[`ready`,`streaming`,`polling`].includes(e.connection);function Mc(e,t){return jc(e)&&e.bootstrap?.actions[t]?.state===`enabled`}function Nc(e,t){return[...e.operations.values()].some(e=>!Ac(e)&&e.target.target_kind===`model`&&(e.target.model_id===t||e.target.eviction_target_id===t))||[...e.pendingReconciliations.values()].some(e=>e.modelId===t)}function Pc(e,t){return Mc(e,`load`)&&t.supported&&t.complete&&t.metadata.support.runnable_on_backend&&!t.lifecycle.busy&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Nc(e,t.identity.id)}function Fc(e,t){return Mc(e,`unload`)&&t.lifecycle.state===`ready`&&!Nc(e,t.identity.id)}function Ic(e,t){return Mc(e,`cache_delete`)&&t.identity.source===`cache`&&t.removable&&t.removal.eligible&&!t.lifecycle.busy&&t.lifecycle.active_requests===0&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Nc(e,t.identity.id)}function Lc(e,t){return jc(e)&&t.lifecycle.state===`ready`&&t.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available)}function Rc(e,t){return e.catalog.filter(n=>n.identity.id!==t&&Fc(e,n)&&!n.lifecycle.busy&&n.lifecycle.active_requests===0&&n.lifecycle.draining_requests===0)}function zc(e,t){if(e===null)return x(t,`models.library.unknown`);if(e===0)return`0 B`;let n=Math.min(4,Math.floor(Math.log(e)/Math.log(1024)));return`${(e/1024**n).toLocaleString(t,{maximumFractionDigits:1})} ${[`B`,`KiB`,`MiB`,`GiB`,`TiB`][n]}`}function Bc(e,t){return e instanceof di?e.envelope?.error.code===`stale_revision`?x(t,`models.library.stale`):e.message:x(t,`models.library.pending`)}var Vc={column:`name`,direction:`asc`},Hc={ready:0,loading:1,draining:2,unloading:3,failed:4,unloaded:5},Uc=new Intl.Collator(void 0,{numeric:!0});function Wc(e){let t=e.lifecycle.state;return t===`ready`?0:t===`loading`||t===`draining`||t===`unloading`?1:2}function Gc(e){return e.metadata.quantization??e.metadata.dtype??null}function Kc(e,t){return t===`size`?e.metadata.disk_bytes:t===`quantization`?Gc(e):t===`state`?Hc[e.lifecycle.state]:e.identity.display_name}function qc(e){return(t,n)=>{let r=Wc(t)-Wc(n);if(r!==0)return r;let i=Kc(t,e.column),a=Kc(n,e.column);if(i===null||a===null){if(i!==a)return i===null?1:-1}else{let t=typeof i==`number`&&typeof a==`number`?i-a:Uc.compare(String(i),String(a));if(t!==0)return e.direction===`asc`?t:-t}return Uc.compare(t.identity.display_name,n.identity.display_name)||t.identity.id.localeCompare(n.identity.id)}}function Jc(e,t,n=Vc){let r=t.query.trim().toLocaleLowerCase();return e.filter(e=>(!r||[e.identity.display_name,e.identity.inference_id,e.metadata.architecture??``].some(e=>e.toLocaleLowerCase().includes(r)))&&(!t.source||e.identity.source===t.source)&&(!t.task||e.capabilities.some(e=>e.task===t.task))&&(!t.status||e.lifecycle.state===t.status)).sort(qc(n))}function Yc(e){let t=[...new Set(e.capabilities.map(e=>e.task))],n=t=>{let n=e.metadata.output_tasks.indexOf(t);return n===-1?e.metadata.output_tasks.length:n};return t.map((e,t)=>({task:e,index:t})).sort((e,t)=>n(e.task)-n(t.task)||e.index-t.index).map(e=>e.task)}function Xc(e,t){let n=[...e.operations.values()].filter(e=>e.kind===`model_load`&&e.target.target_kind===`model`&&e.target.model_id===t).sort((e,t)=>t.created_at.localeCompare(e.created_at))[0];return n?.state===`failed`&&n.error?.code===`conflict`}function Zc(e,t=5){return[...e.operations.values()].filter(e=>e.kind===`download`&&e.target.target_kind===`download`&&e.state!==`succeeded`).sort((e,t)=>Number(Ac(e))-Number(Ac(t))||t.created_at.localeCompare(e.created_at)).slice(0,t)}function Qc(e){let t=e.split(`/`);return t.length===2&&t.every(e=>e.length<=96&&/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(e))}function $c(e){return e.length===0||e.length<=128&&e.split(`/`).every(e=>e!==`.`&&e!==`..`&&/^[A-Za-z0-9._~+-]+$/.test(e))}var el={filter:{query:``,source:``,task:``,status:``},sort:Vc,instance:null},tl=el,nl=new Set;function rl(e){return nl.add(e),()=>nl.delete(e)}function il(e){tl=e;for(let e of nl)e()}function al(e){let t=(0,S.useSyncExternalStore)(rl,()=>tl);return t.instance===null||t.instance===e?t:el}function ol(e){return tl.instance===null||tl.instance===e?tl:el}function sl(e,t){il({...ol(e),filter:t,instance:e})}function cl(e,t){il({...ol(e),sort:t,instance:e})}function ll(){il(el)}var ul=[`appearance`,`requests`,`model`,`server`],dl=`appearance`;function fl(e){return ul.includes(e??``)?e:dl}function pl(e){return`#settings/${e}`}var ml=[`models`,`chat`,`activity`,`settings`,`gallery`];function hl(e){let t=null,n=null,r=()=>n??=e().then(e=>(t=e,e),e=>{throw n=null,e});function i(e){let n=t??(0,S.use)(r());return(0,C.jsx)(n,{...e})}return{Route:i,preload:r}}var gl=hl(()=>ee(()=>import(`./screen-D1LH42Nc.js`).then(e=>e.ModelsLibrary),__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12]),import.meta.url)),_l=hl(()=>ee(()=>import(`./chat-BAZbjt2p.js`).then(e=>e.Chat),__vite__mapDeps([13,1,3,4,5,2,6,7,8,9,10,14,15,11,12,16]),import.meta.url)),vl=hl(()=>ee(()=>import(`./activity-C5mFReiM.js`).then(e=>e.ActivityPage),__vite__mapDeps([17,1,2,3,4,5,6,7,8,9,10,14,15]),import.meta.url)),yl=hl(()=>ee(()=>import(`./settings-screen-DcqdmwEk.js`).then(e=>e.SettingsScreen),__vite__mapDeps([18,1,2,3,4,5,6,7,8,9,10,12,16]),import.meta.url)),bl=hl(()=>ee(()=>import(`./gallery-5S5835UE.js`).then(e=>e.DesignGallery),__vite__mapDeps([19,1,2,3,4,5,6,7,8,9,10,14]),import.meta.url)),V=gl.Route,xl=_l.Route,Sl=vl.Route,Cl=yl.Route,wl=bl.Route;function Tl(){return Promise.all([gl.preload(),_l.preload(),vl.preload(),yl.preload(),bl.preload()])}function El(e){return e.auth.status===`authenticated`&&e.connection!==`schema-mismatch`}function Dl(e=window.location.hash){let t=e.replace(/^#/,``).replace(/^\//,``),[n,...r]=t.split(`/`);return n===`settings`?{route:`settings`,section:fl(r.join(`/`)||void 0)}:{route:ml.includes(t)?t:`models`,section:dl}}function Ol(){let e=Is(),t=Ls(),[n,r]=(0,S.useState)(()=>Dl()),i=n.route,[a,o]=(0,S.useState)(ma),[s,c]=(0,S.useState)(null),[l,u]=(0,S.useState)(null),d=(0,S.useRef)(0);(0,S.useEffect)(()=>()=>{d.current+=1},[]),(0,S.useEffect)(()=>{let e=()=>{Tl().catch(()=>void 0)};if(typeof window.requestIdleCallback==`function`){let t=window.requestIdleCallback(e,{timeout:2e3});return()=>window.cancelIdleCallback(t)}let t=window.setTimeout(e,500);return()=>window.clearTimeout(t)},[]),(0,S.useEffect)(()=>{let e=()=>r(Dl());return window.addEventListener(`hashchange`,e),()=>window.removeEventListener(`hashchange`,e)},[]);let[f,p]=(0,S.useState)(ra),m=ia(a.colorScheme,f);(0,S.useLayoutEffect)(()=>{_a(document.documentElement,a,f),ha(a)},[a,f]),(0,S.useEffect)(()=>{if(a.colorScheme===`system`)return p(ra()),oa(p)},[a.colorScheme]),(0,S.useEffect)(()=>{let e=()=>{document.documentElement.dataset.documentHidden=String(document.hidden)};return e(),document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e)},[]),(0,S.useEffect)(()=>{e.auth.status===`authenticated`&&u(null)},[e.auth.status]),(0,S.useEffect)(()=>{e.auth.status===`signed-out`&&wc([])},[e.auth.status]);let g=e=>{r({route:e,section:dl}),window.history.replaceState(null,``,`#${e}`)},_=e=>{r({route:`settings`,section:e}),window.history.replaceState(null,``,pl(e))},v=e=>{let n=d.current+1;d.current=n,u(null),t.login(e).catch(e=>{d.current===n&&(e instanceof DOMException&&e.name===`AbortError`||u(bi(e)))})},y=()=>{d.current+=1,u(null),ll(),t.logout()},ee=()=>{u(null),t.refresh()},te=()=>{y(),window.location.reload()},re=n=>{let r=Ti(e,n);r!==e.selectedModelId&&t.selectModel(r),r!==null&&F(),g(`models`),c(null)},ie=()=>{El(e)&&Tc(),g(`chat`),c(null)},ae=wi(e),oe=e.auth.status===`authenticated`&&e.catalogSequence!==null?ae.map(e=>({id:e.identity.id,name:e.identity.display_name,state:e.lifecycle.state,stateLabel:Ri(a.locale,e.lifecycle.state)})):null,se=kl(n,a,o,_,{snapshot:e,authFailure:l,login:v,logout:y,retry:ee,recoverSchema:te});return(0,C.jsxs)(Wo,{family:a.themeFamily,scheme:m,locale:a.locale,children:[(0,C.jsx)(Gi,{locale:a.locale,route:i,onRouteChange:g,onCommand:()=>c(`command`),onHelp:()=>c(`help`),onNewChat:ie,onOpenModel:re,loadedModels:oe,connection:{label:Oi(a.locale,e),state:e.connection,details:ki(a.locale,e),summary:e.bootstrap===null?null:Fi(a.locale,e)},sessionAction:e.auth.tokenPresent?(0,C.jsx)(h,{label:x(a.locale,`toolbar.logout`),icon:`key`,onClick:y,"data-testid":b(`toolbar.logout`)}):null,inspector:null,children:(0,C.jsx)(S.Suspense,{fallback:null,children:se})}),(0,C.jsx)(Hi,{open:s===`command`,locale:a.locale,catalog:e.catalog,loaded:ae,onClose:()=>c(null),onNavigate:e=>{g(e),c(null)},onNewChat:ie,onOpenModel:re}),(0,C.jsxs)(ne,{open:s===`help`,title:x(a.locale,`help.title`),onClose:()=>c(null),testId:`help-dialog`,closeLabel:x(a.locale,`common.close`),children:[(0,C.jsx)(`p`,{"data-testid":b(`help.body`),children:x(a.locale,`help.body`)}),(0,C.jsx)(`ul`,{className:`shortcut-list`,"data-testid":`help-shortcuts`,children:Wi.map(e=>(0,C.jsxs)(`li`,{"data-testid":`help-shortcut-${e.id}`,children:[(0,C.jsx)(`span`,{className:`shortcut-keys`,children:e.keys.map((t,n)=>(0,C.jsxs)(S.Fragment,{children:[n>0?` ${e.separator} `:null,(0,C.jsx)(`kbd`,{children:t})]},t))}),(0,C.jsx)(`span`,{children:x(a.locale,e.key)})]},e.id))})]})]})}function kl(e,t,n,r,i){let{route:a}=e;return a===`models`?(0,C.jsx)(Al,{locale:t.locale,context:i}):a===`chat`?(0,C.jsx)(jl,{locale:t.locale,context:i}):a===`activity`?(0,C.jsx)(Ml,{locale:t.locale,context:i}):a===`settings`?(0,C.jsx)(Cl,{appearance:t,setAppearance:n,section:e.section,onSectionChange:r}):(0,C.jsx)(wl,{locale:t.locale})}function Al(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,C.jsx)(V,{locale:e.locale}):(0,C.jsx)(ji,{locale:e.locale,title:x(e.locale,`models.title`),titleTestId:b(`models.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function jl(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,C.jsx)(xl,{locale:e.locale}):(0,C.jsx)(ji,{locale:e.locale,title:x(e.locale,`chat.title`),titleTestId:b(`chat.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function Ml(e){return e.context.snapshot.auth.status===`authenticated`&&![`schema-mismatch`,`forbidden`,`unauthorized`].includes(e.context.snapshot.connection)?(0,C.jsx)(Sl,{locale:e.locale}):(0,C.jsx)(ji,{locale:e.locale,title:x(e.locale,`activity.title`),titleTestId:b(`activity.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}var Nl=document.getElementById(`root`);if(Nl===null)throw Error(`Missing #root element for mlxcel WebUI.`);(0,pe.createRoot)(Nl).render((0,C.jsx)(S.StrictMode,{children:(0,C.jsx)(Fs,{children:(0,C.jsx)(Ol,{})})}));export{kc as A,$o as B,$c as C,wc as D,vc as E,Is as F,br as G,di as H,Ls as I,xr as K,R as L,uc as M,sc as N,_c as O,cc as P,L as R,Qc as S,Dc as T,Cr as U,Ri as V,wr as W,Rc as _,Vc as a,Gc as b,Lc as c,Fc as d,Xc as f,Bc as g,Yc as h,al as i,Ec as j,Oc as k,Ic as l,jc as m,sl as n,Mc as o,qc as p,cl as r,zc as s,ul as t,Pc as u,Jc as v,Zc as w,Ac as x,Nc as y,I as z};