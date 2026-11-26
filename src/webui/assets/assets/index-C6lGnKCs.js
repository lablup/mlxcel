const __vite__mapDeps=(i,m=__vite__mapDeps,d=(m.f||(m.f=["./screen-DY8g_dbJ.js","./jsx-runtime-Wy9s6hX0.js","./primitives-DfyACQNx.js","./useMenuHover-BzwCa-Bv.js","./useLayerDismissal-rbmsZuUT.js","./focusOutline.stylex-BC7_DHXh.js","./Tooltip-D_hNtMsy.js","./isRtlElement-BLtGBAfI.js","./scrollKeyboardDelegation-DUknF5xO.js","./useMenuHover-fLF2jYG2.css","./primitives-Djjnr2Ol.css","./load-profiles-DZB82Wpx.js","./load-action-CNkCMZD-.js","./chat-CVipOb7r.js","./format-CAesP9VQ.js","./format-C_q3_elU.js","./server-defaults-jgoBX8Jq.js","./activity-D7FGoQJ6.js","./settings-screen-CJRvqCAJ.js","./gallery-B2Rl3xjE.js"])))=>i.map(i=>d[i]);
import{a as e,n as t,r as n,t as r}from"./jsx-runtime-Wy9s6hX0.js";import{Et as i,F as a,K as o,O as s,P as c,S as l,St as u,Tt as d,_ as f,bt as p,d as m,g as h,h as g,i as _,k as v,l as y,lt as b,m as x,p as S,r as C,s as w,t as T,u as E,w as ee,wt as te,xt as D,y as ne,yt as O}from"./useLayerDismissal-rbmsZuUT.js";import{At as re,B as ie,E as k,F as ae,Ft as oe,G as se,H as ce,J as A,K as j,L as M,M as le,Mt as ue,N as de,Nt as fe,O as pe,Ot as me,Pt as he,Q as ge,R as N,St as P,Tt as _e,U as ve,V as ye,W as be,X as F,Y as I,Z as xe,_ as Se,a as Ce,b as we,bt as Te,dt as Ee,et as De,gt as Oe,ht as ke,jt as Ae,k as je,kt as Me,l as L,o as Ne,t as Pe,tt as Fe,u as R,v as Ie,w as Le,wt as Re,x as ze,yt as Be,z as Ve}from"./useMenuHover-BzwCa-Bv.js";import{D as He,E as Ue,a as We,c as Ge,f as Ke,i as qe,l as Je,o as Ye,p as Xe,r as Ze,s as Qe,u as $e,y as et}from"./focusOutline.stylex-BC7_DHXh.js";import{n as tt,r as nt}from"./isRtlElement-BLtGBAfI.js";import{D as rt,E as it,G as at,K as ot,N as st,O as ct,P as lt,T as ut,a as dt,g as ft,h as pt,i as mt,l as ht,m as gt,p as _t,s as vt,t as yt,u as bt}from"./scrollKeyboardDelegation-DUknF5xO.js";import{S as xt,r as St,x as Ct}from"./Tooltip-D_hNtMsy.js";var wt=n((e=>{function t(e,t){var n=e.length;e.push(t);a:for(;0<n;){var r=n-1>>>1,a=e[r];if(0<i(a,t))e[r]=t,e[n]=a,n=r;else break a}}function n(e){return e.length===0?null:e[0]}function r(e){if(e.length===0)return null;var t=e[0],n=e.pop();if(n!==t){e[0]=n;a:for(var r=0,a=e.length,o=a>>>1;r<o;){var s=2*(r+1)-1,c=e[s],l=s+1,u=e[l];if(0>i(c,n))l<a&&0>i(u,c)?(e[r]=u,e[l]=n,r=l):(e[r]=c,e[s]=n,r=s);else if(l<a&&0>i(u,n))e[r]=u,e[l]=n,r=l;else break a}}return t}function i(e,t){var n=e.sortIndex-t.sortIndex;return n===0?e.id-t.id:n}if(e.unstable_now=void 0,typeof performance==`object`&&typeof performance.now==`function`){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var c=[],l=[],u=1,d=null,f=3,p=!1,m=!1,h=!1,g=!1,_=typeof setTimeout==`function`?setTimeout:null,v=typeof clearTimeout==`function`?clearTimeout:null,y=typeof setImmediate<`u`?setImmediate:null;function b(e){for(var i=n(l);i!==null;){if(i.callback===null)r(l);else if(i.startTime<=e)r(l),i.sortIndex=i.expirationTime,t(c,i);else break;i=n(l)}}function x(e){if(h=!1,b(e),!m){if(n(c)!==null)m=!0,S||(S=!0,te());else{var t=n(l);t!==null&&O(x,t.startTime-e)}}}var S=!1,C=-1,w=5,T=-1;function E(){return g?!0:!(e.unstable_now()-T<w)}function ee(){if(g=!1,S){var t=e.unstable_now();T=t;var i=!0;try{a:{m=!1,h&&(h=!1,v(C),C=-1),p=!0;var a=f;try{b:{for(b(t),d=n(c);d!==null&&!(d.expirationTime>t&&E());){var o=d.callback;if(typeof o==`function`){d.callback=null,f=d.priorityLevel;var s=o(d.expirationTime<=t);if(t=e.unstable_now(),typeof s==`function`){d.callback=s,b(t),i=!0;break b}d===n(c)&&r(c),b(t)}else r(c);d=n(c)}if(d!==null)i=!0;else{var u=n(l);u!==null&&O(x,u.startTime-t),i=!1}}break a}finally{d=null,f=a,p=!1}i=void 0}}finally{i?te():S=!1}}}var te;if(typeof y==`function`)te=function(){y(ee)};else if(typeof MessageChannel<`u`){var D=new MessageChannel,ne=D.port2;D.port1.onmessage=ee,te=function(){ne.postMessage(null)}}else te=function(){_(ee,0)};function O(t,n){C=_(function(){t(e.unstable_now())},n)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(e){e.callback=null},e.unstable_forceFrameRate=function(e){0>e||125<e?console.error(`forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported`):w=0<e?Math.floor(1e3/e):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_next=function(e){switch(f){case 1:case 2:case 3:var t=3;break;default:t=f}var n=f;f=t;try{return e()}finally{f=n}},e.unstable_requestPaint=function(){g=!0},e.unstable_runWithPriority=function(e,t){switch(e){case 1:case 2:case 3:case 4:case 5:break;default:e=3}var n=f;f=e;try{return t()}finally{f=n}},e.unstable_scheduleCallback=function(r,i,a){var o=e.unstable_now();switch(typeof a==`object`&&a?(a=a.delay,a=typeof a==`number`&&0<a?o+a:o):a=o,r){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=a+s,r={id:u++,callback:i,priorityLevel:r,startTime:a,expirationTime:s,sortIndex:-1},a>o?(r.sortIndex=a,t(l,r),n(c)===null&&r===n(l)&&(h?(v(C),C=-1):h=!0,O(x,a-o))):(r.sortIndex=s,t(c,r),m||p||(m=!0,S||(S=!0,te()))),r},e.unstable_shouldYield=E,e.unstable_wrapCallback=function(e){var t=f;return function(){var n=f;f=t;try{return e.apply(this,arguments)}finally{f=n}}}})),Tt=n(((e,t)=>{t.exports=wt()})),Et=n((e=>{var n=Tt(),r=t(),a=i();function o(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function s(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function c(e){for(var t=e,n=t;n&&!n.alternate;)t=n,t.flags&4098&&(e=t.return),n=t.return;for(;t.return;)t=t.return;return t.tag===3?e:null}function l(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function u(e){if(e.tag===31){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function d(e){if(c(e)!==e)throw Error(o(188))}function f(e){var t=e.alternate;if(!t){if(t=c(e),t===null)throw Error(o(188));return t===e?e:null}for(var n=e,r=t;;){var i=n.return;if(i===null)break;var a=i.alternate;if(a===null){if(r=i.return,r!==null){n=r;continue}break}if(i.child===a.child){for(a=i.child;a;){if(a===n)return d(i),e;if(a===r)return d(i),t;a=a.sibling}throw Error(o(188))}if(n.return!==r.return)n=i,r=a;else{for(var s=!1,l=i.child;l;){if(l===n){s=!0,n=i,r=a;break}if(l===r){s=!0,r=i,n=a;break}l=l.sibling}if(!s){for(l=a.child;l;){if(l===n){s=!0,n=a,r=i;break}if(l===r){s=!0,r=a,n=i;break}l=l.sibling}if(!s)throw Error(o(189))}}if(n.alternate!==r)throw Error(o(190))}if(n.tag!==3)throw Error(o(188));return n.stateNode.current===n?e:t}function p(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e;for(e=e.child;e!==null;){if(t=p(e),t!==null)return t;e=e.sibling}return null}function m(e,t,n,r,i,a){for(;e!==null;){if((e.tag===5||e.tag===27||e.tag===6)&&n(e,r,i,a)||(e.tag!==22||e.memoizedState===null)&&(t||e.tag!==5&&e.tag!==27)&&m(e.child,t,n,r,i,a))return!0;e=e.sibling}return!1}function h(e){for(e=e.return;e!==null;){if(e.tag===3||e.tag===5||e.tag===27)return e;e=e.return}return null}function g(e){var t=!1;for(e=e.return;e!==null&&(e.tag===4&&(t=!0),e.tag!==3&&e.tag!==5&&e.tag!==27);)e=e.return;return t}function _(e){var t=[null,null],n=h(e);return n===null||v(t,e,n.child,{foundSelf:!1}),t}function v(e,t,n,r){for(;n!==null;){if(n===t)r.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(r.foundSelf)return e[1]=n,!0;e[0]=n}else if((n.tag!==22||n.memoizedState===null)&&v(e,t,n.child,r))return!0;n=n.sibling}return!1}function y(e){switch(e.tag){case 5:case 27:case 6:return e.stateNode;case 3:return e.stateNode.containerInfo;default:throw Error(o(559))}}var b=null,x=null;function S(e,t,n){return e===n||e===t&&(b=e,!0)}function C(e,t,n){return e===n?(x=e,!1):e===t&&(x!==null&&(b=e),!0)}function w(e){if(e===null)return null;do e=e===null?null:e.return;while(e&&e.tag!==5&&e.tag!==27&&e.tag!==3);return e||null}function T(e,t,n){for(var r=0,i=e;i;i=n(i))r++;i=0;for(var a=t;a;a=n(a))i++;for(;0<r-i;)e=n(e),r--;for(;0<i-r;)t=n(t),i--;for(;r--;){if(e===t||t!==null&&e===t.alternate)return e;e=n(e),t=n(t)}return null}var E=Object.assign,ee=Symbol.for(`react.element`),te=Symbol.for(`react.transitional.element`),D=Symbol.for(`react.portal`),ne=Symbol.for(`react.fragment`),O=Symbol.for(`react.strict_mode`),re=Symbol.for(`react.profiler`),ie=Symbol.for(`react.consumer`),k=Symbol.for(`react.context`),ae=Symbol.for(`react.forward_ref`),oe=Symbol.for(`react.suspense`),se=Symbol.for(`react.suspense_list`),ce=Symbol.for(`react.memo`),A=Symbol.for(`react.lazy`),j=Symbol.for(`react.activity`),M=Symbol.for(`react.legacy_hidden`),le=Symbol.for(`react.memo_cache_sentinel`),ue=Symbol.for(`react.view_transition`),de=Symbol.for(`react.recoverable`),fe=Symbol.iterator;function pe(e){return typeof e!=`object`||!e?null:(e=fe&&e[fe]||e[`@@iterator`],typeof e==`function`?e:null)}var me=Symbol.for(`react.client.reference`);function he(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===me?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case ne:return`Fragment`;case re:return`Profiler`;case O:return`StrictMode`;case oe:return`Suspense`;case se:return`SuspenseList`;case j:return`Activity`;case ue:return`ViewTransition`}if(typeof e==`object`)switch(e.$$typeof){case D:return`Portal`;case k:return e.displayName||`Context`;case ie:return(e._context.displayName||`Context`)+`.Consumer`;case ae:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case ce:return t=e.displayName||null,t===null?he(e.type)||`Memo`:t;case A:t=e._payload,e=e._init;try{return he(e(t))}catch{}}return null}var ge=Array.isArray,N=r.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,P=a.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,_e={pending:!1,data:null,method:null,action:null},ve=[],ye=-1;function be(e){return{current:e}}function F(e){0>ye||(e.current=ve[ye],ve[ye]=null,ye--)}function I(e,t){ye++,ve[ye]=e.current,e.current=t}var xe=be(null),Se=be(null),Ce=be(null),we=be(null);function Te(e,t){switch(I(Ce,t),I(Se,e),I(xe,null),t.nodeType){case 9:case 11:e=(e=t.documentElement)&&(e=e.namespaceURI)?up(e):0;break;default:if(e=t.tagName,t=t.namespaceURI)t=up(t),e=dp(t,e);else switch(e){case`svg`:e=1;break;case`math`:e=2;break;default:e=0}}F(xe),I(xe,e)}function Ee(){F(xe),F(Se),F(Ce)}function De(e){var t=e.memoizedState;t!==null&&(sh._currentValue=t.memoizedState,I(we,e)),t=xe.current;var n=dp(t,e.type);t!==n&&(I(Se,e),I(xe,n))}function Oe(e){Se.current===e&&(F(xe),F(Se)),we.current===e&&(F(we),sh._currentValue=_e)}var ke,Ae;function je(e){if(ke===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);ke=t&&t[1]||``,Ae=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+ke+e+Ae}var Me=!1;function L(e,t){if(!e||Me)return``;Me=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}n=!1;try{var i=Object.getOwnPropertyDescriptor(e.prototype,`props`);Object.defineProperty(e.prototype,"props",{configurable:!0,set:function(){throw Error()}}),n=!0,new e}finally{n&&(i===void 0?delete e.prototype.props:Object.defineProperty(e.prototype,"props",i))}}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,"name",{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Me=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?je(n):``}function Ne(e,t){switch(e.tag){case 26:case 27:case 5:return je(e.type);case 16:return je(`Lazy`);case 13:return e.child!==t&&t!==null?je(`Suspense Fallback`):je(`Suspense`);case 19:return je(`SuspenseList`);case 0:case 15:return L(e.type,!1);case 11:return L(e.type.render,!1);case 1:return L(e.type,!0);case 31:return je(`Activity`);case 30:return je(`ViewTransition`);default:return``}}function Pe(e){try{var t=``,n=null;do t+=Ne(e,n),n=e,e=e.return;while(e);return t}catch(e){return`
Error generating stack: `+e.message+`
`+e.stack}}var Fe=Object.prototype.hasOwnProperty,R=n.unstable_scheduleCallback,Ie=n.unstable_cancelCallback,Le=n.unstable_shouldYield,Re=n.unstable_requestPaint,ze=n.unstable_now,Be=n.unstable_getCurrentPriorityLevel,Ve=n.unstable_ImmediatePriority,He=n.unstable_UserBlockingPriority,Ue=n.unstable_NormalPriority,We=n.unstable_LowPriority,Ge=n.unstable_IdlePriority,Ke=n.log,qe=n.unstable_setDisableYieldValue,Je=null,Ye=null;function Xe(e){if(typeof Ke==`function`&&qe(e),Ye&&typeof Ye.setStrictMode==`function`)try{Ye.setStrictMode(Je,e)}catch{}}var Ze=Math.clz32?Math.clz32:et,Qe=Math.log,$e=Math.LN2;function et(e){return e>>>=0,e===0?32:31-(Qe(e)/$e|0)|0}var tt=256,nt=262144,rt=4194304;function it(e){var t=e&42;if(t!==0)return t;switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return e&-e;case 262144:case 524288:case 1048576:case 2097152:return e&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return e&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return e}}function at(e,t,n){var r=e.pendingLanes;if(r===0)return 0;var i=0,a=e.suspendedLanes,o=e.pingedLanes;e=e.warmLanes;var s=r&134217727;return s===0?(s=r&~a,s===0?o===0?n||(n=r&~e,n!==0&&(i=it(n))):i=it(o):i=it(s)):(r=s&~a,r===0?(o&=s,o===0?n||(n=s&~e,n!==0&&(i=it(n))):i=it(o)):i=it(r)),i===0?0:t!==0&&t!==i&&(t&a)===0&&(a=i&-i,n=t&-t,a>=n||a===32&&n&4194048)?t:i}function ot(e,t){return(e.pendingLanes&~(e.suspendedLanes&~e.pingedLanes)&t)===0}function st(e,t){t&8&&(t|=t&32);var n=e.entangledLanes;if(n!==0)for(e=e.entanglements,n&=t;0<n;){var r=31-Ze(n),i=1<<r;t|=e[r],n&=~i}return t}function ct(e,t){switch(e){case 1:case 2:case 4:case 8:case 64:return t+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function lt(){var e=rt;return rt<<=1,!(rt&62914560)&&(rt=4194304),e}function ut(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function dt(e,t){e.pendingLanes|=t,t!==268435456&&(e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0)}function ft(e,t,n,r,i,a){var o=e.pendingLanes;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0,e.expiredLanes&=n,e.entangledLanes&=n,e.errorRecoveryDisabledLanes&=n,e.shellSuspendCounter=0;var s=e.entanglements,c=e.expirationTimes,l=e.hiddenUpdates;for(n=o&~n;0<n;){var u=31-Ze(n),d=1<<u;s[u]=0,c[u]=-1;var f=l[u];if(f!==null)for(l[u]=null,u=0;u<f.length;u++){var p=f[u];p!==null&&(p.lane&=-536870913)}n&=~d}r!==0&&pt(e,r,0),a!==0&&i===0&&e.tag!==0&&(e.suspendedLanes|=a&~(o&~t))}function pt(e,t,n){e.pendingLanes|=t,e.suspendedLanes&=~t;var r=31-Ze(t);e.entangledLanes|=t,e.entanglements[r]=e.entanglements[r]|1073741824|n&261930}function mt(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Ze(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}function ht(e,t){var n=t&-t;return n=n&42?1:gt(n),(n&(e.suspendedLanes|t))===0?n:0}function gt(e){switch(e){case 2:e=1;break;case 8:e=4;break;case 32:e=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:e=128;break;case 268435456:e=134217728;break;default:e=0}return e}function _t(e){return e&=-e,2<e?8<e?e&134217727?32:268435456:8:2}function vt(){var e=P.p;return e===0?(e=window.event,e===void 0?32:Ch(e.type)):e}function yt(e,t){var n=P.p;try{return P.p=e,t()}finally{P.p=n}}var bt=Math.random().toString(36).slice(2),xt=`__reactFiber$`+bt,St=`__reactProps$`+bt,Ct=`__reactContainer$`+bt,wt=`__reactEvents$`+bt,Et=`__reactListeners$`+bt,Dt=`__reactHandles$`+bt,z=`__reactResources$`+bt,Ot=`__reactMarker$`+bt,kt=`__reactLoad$`+bt;function At(e){delete e[xt],delete e[St],delete e[Et],delete e[Dt]}function jt(e){var t;if(t=e[xt])return t;for(var n=e.parentNode;n;){if(t=n[Ct]||n[xt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=fm(e);e!==null;){if(n=e[xt])return n;e=fm(e)}return t}e=n,n=e.parentNode}return null}function Mt(e){if(e=e[xt]||e[Ct]){var t=e.tag;if(t===5||t===6||t===13||t===31||t===26||t===27||t===3)return e}return null}function Nt(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e.stateNode;throw Error(o(33))}function Pt(e){var t=e[z];return t||=e[z]={hoistableStyles:new Map,hoistableScripts:new Map},t}function Ft(e){e[Ot]=!0}function It(e){e[kt]=void 0}var Lt=new Set,Rt={};function zt(e,t){Bt(e,t),Bt(e+`Capture`,t)}function Bt(e,t){for(Rt[e]=t,e=0;e<t.length;e++)Lt.add(t[e])}var Vt=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),Ht={},Ut={};function Wt(e){return Fe.call(Ut,e)?!0:Fe.call(Ht,e)?!1:Vt.test(e)?Ut[e]=!0:(Ht[e]=!0,!1)}var B=!1;function Gt(){var e=B;return B=!1,e}function Kt(e,t,n){if(Wt(t)){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:e.removeAttribute(t);return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`){e.removeAttribute(t);return}}e.setAttribute(t,n)}}}function qt(e,t,n){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(t);return}e.setAttribute(t,n)}}function Jt(e,t,n,r){if(r===null)e.removeAttribute(n);else{switch(typeof r){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(n);return}e.setAttributeNS(t,n,r)}}function Yt(e){switch(typeof e){case`bigint`:case`boolean`:case`number`:case`string`:case`undefined`:return e;case`object`:return e;default:return``}}function Xt(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()===`input`&&(t===`checkbox`||t===`radio`)}function Zt(e,t,n){var r=Object.getOwnPropertyDescriptor(e.constructor.prototype,t);if(!e.hasOwnProperty(t)&&r!==void 0&&typeof r.get==`function`&&typeof r.set==`function`){var i=r.get,a=r.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(e){n=``+e,a.call(this,e)}}),Object.defineProperty(e,t,{enumerable:r.enumerable}),{getValue:function(){return n},setValue:function(e){n=``+e},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function Qt(e){if(!e._valueTracker){var t=Xt(e)?`checked`:`value`;e._valueTracker=Zt(e,t,``+e[t])}}function $t(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r=``;return e&&(r=Xt(e)?e.checked?`true`:`false`:e.value),e=r,e!==n&&(t.setValue(e),!0)}var en=/[\n"\\]/g;function tn(e){return e.replace(en,function(e){return`\\`+e.charCodeAt(0).toString(16)+` `})}function nn(e,t,n,r,i,a,o,s){e.name=``,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`?e.type=o:e.removeAttribute(`type`),t==null?o!==`submit`&&o!==`reset`||e.removeAttribute(`value`):o===`number`?(t===0&&e.value===``||e.value!=t)&&(e.value=``+Yt(t)):e.value!==``+Yt(t)&&(e.value=``+Yt(t)),t==null?n==null?r!=null&&e.removeAttribute(`value`):an(e,Yt(n)):o===`number`&&e.value==t?an(e,Yt(e.value)):an(e,Yt(t)),i==null&&a!=null&&(e.defaultChecked=!!a),i!=null&&(e.checked=i&&typeof i!=`function`&&typeof i!=`symbol`),s!=null&&typeof s!=`function`&&typeof s!=`symbol`&&typeof s!=`boolean`?e.name=``+Yt(s):e.removeAttribute(`name`)}function rn(e,t,n,r,i,a,o,s){if(a!=null&&typeof a!=`function`&&typeof a!=`symbol`&&typeof a!=`boolean`&&(e.type=a),t!=null||n!=null){if(!(a!==`submit`&&a!==`reset`||t!=null)){Qt(e);return}n=n==null?``:``+Yt(n),t=t==null?n:``+Yt(t),s||t===e.value||(e.value=t),e.defaultValue=t}r??=i,r=typeof r!=`function`&&typeof r!=`symbol`&&!!r,e.checked=s?e.checked:!!r,e.defaultChecked=!!r,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`&&(e.name=o),Qt(e)}function an(e,t){e.defaultValue!==``+t&&(e.defaultValue=``+t)}function on(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t[`$`+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty(`$`+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=``+Yt(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function sn(e,t,n){t!=null&&(t=``+Yt(t),t!==e.value&&(e.value=t),n==null)?e.defaultValue!==t&&(e.defaultValue=t):e.defaultValue=n==null?``:``+Yt(n)}function cn(e,t,n,r){if(t==null){if(r!=null){if(n!=null)throw Error(o(92));if(ge(r)){if(1<r.length)throw Error(o(93));r=r[0]}n=r}n??=``,t=n}n=Yt(t),e.defaultValue=n,r=e.textContent,r===n&&r!==``&&r!==null&&(e.value=r),Qt(e)}function ln(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var un=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `));function dn(e,t,n){var r=t.indexOf(`--`)===0;n==null||typeof n==`boolean`||n===``?r?e.setProperty(t,``):t===`float`?e.cssFloat=``:e[t]=``:r?e.setProperty(t,n):typeof n!=`number`||n===0||un.has(t)?t===`float`?e.cssFloat=n:e[t]=(``+n).trim():e[t]=n+`px`}function fn(e,t,n){if(t!=null&&typeof t!=`object`)throw Error(o(62));if(e=e.style,n!=null){for(var r in n)!n.hasOwnProperty(r)||t!=null&&t.hasOwnProperty(r)||(r.indexOf(`--`)===0?e.setProperty(r,``):r===`float`?e.cssFloat=``:e[r]=``,B=!0);for(var i in t)r=t[i],t.hasOwnProperty(i)&&n[i]!==r&&(dn(e,i,r),B=!0)}else for(var a in t)t.hasOwnProperty(a)&&dn(e,a,t[a])}function pn(e){if(e.indexOf(`-`)===-1)return!1;switch(e){case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:return!1;default:return!0}}var mn=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`maskType`,`mask-type`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),hn=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function gn(e){return hn.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}function _n(){}var vn=null;function yn(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var bn=null,xn=null;function Sn(e){var t=Mt(e);if(t&&(e=t.stateNode)){var n=e[St]||null;a:switch(e=t.stateNode,t.type){case`input`:if(nn(e,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),t=n.name,n.type===`radio`&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll(`input[name="`+tn(``+t)+`"][type="radio"]`),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var i=r[St]||null;if(!i)throw Error(o(90));nn(r,i.value,i.defaultValue,i.defaultValue,i.checked,i.defaultChecked,i.type,i.name)}}for(t=0;t<n.length;t++)r=n[t],r.form===e.form&&$t(r)}break a;case`textarea`:sn(e,n.value,n.defaultValue);break a;case`select`:t=n.value,t!=null&&on(e,!!n.multiple,t,!1)}}}var Cn=!1;function wn(e,t,n){if(Cn)return e(t,n);Cn=!0;try{return e(t)}finally{if(Cn=!1,(bn!==null||xn!==null)&&(Ld(),bn&&(t=bn,e=xn,xn=bn=null,Sn(t),e)))for(t=0;t<e.length;t++)Sn(e[t])}}function Tn(e,t){var n=e.stateNode;if(n===null)return null;var r=n[St]||null;if(r===null)return null;n=r[t];a:switch(t){case`onClick`:case`onClickCapture`:case`onDoubleClick`:case`onDoubleClickCapture`:case`onMouseDown`:case`onMouseDownCapture`:case`onMouseMove`:case`onMouseMoveCapture`:case`onMouseUp`:case`onMouseUpCapture`:case`onMouseEnter`:(r=!r.disabled)||(e=e.type,r=e!==`button`&&e!==`input`&&e!==`select`&&e!==`textarea`),e=!r;break a;default:e=!1}if(e)return null;if(n&&typeof n!=`function`)throw Error(o(231,t,typeof n));return n}var En=typeof window<`u`&&window.document!==void 0&&window.document.createElement!==void 0,Dn=!1;if(En)try{var On={};Object.defineProperty(On,"passive",{get:function(){Dn=!0}}),window.addEventListener(`test`,On,On),window.removeEventListener(`test`,On,On)}catch{Dn=!1}var kn=null,An=null,jn=null;function Mn(){if(jn)return jn;var e,t=An,n=t.length,r,i=`value`in kn?kn.value:kn.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return jn=i.slice(e,1<r?1-r:void 0)}function Nn(e){var t=e.keyCode;return`charCode`in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Pn(){return!0}function Fn(){return!1}function In(e){function t(t,n,r,i,a){for(var o in this._reactName=t,this._targetInst=r,this.type=n,this.nativeEvent=i,this.target=a,this.currentTarget=null,e)e.hasOwnProperty(o)&&(t=e[o],this[o]=t?t(i):i[o]);return this.isDefaultPrevented=(i.defaultPrevented==null?!1===i.returnValue:i.defaultPrevented)?Pn:Fn,this.isPropagationStopped=Fn,this}return E(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var e=this.nativeEvent;e&&(e.preventDefault?e.preventDefault():typeof e.returnValue!=`unknown`&&(e.returnValue=!1),this.isDefaultPrevented=Pn)},stopPropagation:function(){var e=this.nativeEvent;e&&(e.stopPropagation?e.stopPropagation():typeof e.cancelBubble!=`unknown`&&(e.cancelBubble=!0),this.isPropagationStopped=Pn)},persist:function(){},isPersistent:Pn}),t}var Ln={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Rn=In(Ln),zn=E({},Ln,{view:0,detail:0}),Bn=In(zn),Vn,Hn,Un,Wn=E({},zn,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:tr,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return`movementX`in e?e.movementX:(e!==Un&&(Un&&e.type===`mousemove`?(Vn=e.screenX-Un.screenX,Hn=e.screenY-Un.screenY):Hn=Vn=0,Un=e),Vn)},movementY:function(e){return`movementY`in e?e.movementY:Hn}}),Gn=In(Wn),Kn=In(E({},Wn,{dataTransfer:0})),qn=In(E({},zn,{relatedTarget:0})),Jn=In(E({},Ln,{animationName:0,elapsedTime:0,pseudoElement:0})),Yn=In(E({},Ln,{clipboardData:function(e){return`clipboardData`in e?e.clipboardData:window.clipboardData}})),Xn=In(E({},Ln,{data:0})),Zn={Esc:`Escape`,Spacebar:` `,Left:`ArrowLeft`,Up:`ArrowUp`,Right:`ArrowRight`,Down:`ArrowDown`,Del:`Delete`,Win:`OS`,Menu:`ContextMenu`,Apps:`ContextMenu`,Scroll:`ScrollLock`,MozPrintableKey:`Unidentified`},Qn={8:`Backspace`,9:`Tab`,12:`Clear`,13:`Enter`,16:`Shift`,17:`Control`,18:`Alt`,19:`Pause`,20:`CapsLock`,27:`Escape`,32:` `,33:`PageUp`,34:`PageDown`,35:`End`,36:`Home`,37:`ArrowLeft`,38:`ArrowUp`,39:`ArrowRight`,40:`ArrowDown`,45:`Insert`,46:`Delete`,112:`F1`,113:`F2`,114:`F3`,115:`F4`,116:`F5`,117:`F6`,118:`F7`,119:`F8`,120:`F9`,121:`F10`,122:`F11`,123:`F12`,144:`NumLock`,145:`ScrollLock`,224:`Meta`},$n={Alt:`altKey`,Control:`ctrlKey`,Meta:`metaKey`,Shift:`shiftKey`};function er(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=$n[e])?!!t[e]:!1}function tr(){return er}var nr=In(E({},zn,{key:function(e){if(e.key){var t=Zn[e.key]||e.key;if(t!==`Unidentified`)return t}return e.type===`keypress`?(e=Nn(e),e===13?`Enter`:String.fromCharCode(e)):e.type===`keydown`||e.type===`keyup`?Qn[e.keyCode]||`Unidentified`:``},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:tr,charCode:function(e){return e.type===`keypress`?Nn(e):0},keyCode:function(e){return e.type===`keydown`||e.type===`keyup`?e.keyCode:0},which:function(e){return e.type===`keypress`?Nn(e):e.type===`keydown`||e.type===`keyup`?e.keyCode:0}})),rr=In(E({},Wn,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0})),ir=In(E({},Ln,{submitter:0})),ar=In(E({},zn,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:tr})),or=In(E({},Ln,{propertyName:0,elapsedTime:0,pseudoElement:0})),sr=In(E({},Wn,{deltaX:function(e){return`deltaX`in e?e.deltaX:`wheelDeltaX`in e?-e.wheelDeltaX:0},deltaY:function(e){return`deltaY`in e?e.deltaY:`wheelDeltaY`in e?-e.wheelDeltaY:`wheelDelta`in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0})),cr=In(E({},Ln,{newState:0,oldState:0,source:0})),lr=[9,13,27,32],ur=En&&`CompositionEvent`in window,dr=null;En&&`documentMode`in document&&(dr=document.documentMode);var fr=En&&`TextEvent`in window&&!dr,pr=En&&(!ur||dr&&8<dr&&11>=dr),mr=` `,hr=!1;function gr(e,t){switch(e){case`keyup`:return lr.indexOf(t.keyCode)!==-1;case`keydown`:return t.keyCode!==229;case`keypress`:case`mousedown`:case`focusout`:return!0;default:return!1}}function _r(e){return e=e.detail,typeof e==`object`&&`data`in e?e.data:null}var vr=!1;function yr(e,t){switch(e){case`compositionend`:return _r(t);case`keypress`:return t.which===32?(hr=!0,mr):null;case`textInput`:return e=t.data,e===mr&&hr?null:e;default:return null}}function br(e,t){if(vr)return e===`compositionend`||!ur&&gr(e,t)?(e=Mn(),jn=An=kn=null,vr=!1,e):null;switch(e){case`paste`:return null;case`keypress`:if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case`compositionend`:return pr&&t.locale!==`ko`?null:t.data;default:return null}}var xr={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Sr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t===`input`?!!xr[e.type]:t===`textarea`}function Cr(e,t,n,r){bn?xn?xn.push(r):xn=[r]:bn=r,t=qf(t,`onChange`),0<t.length&&(n=new Rn(`onChange`,`change`,null,n,r),e.push({event:n,listeners:t}))}var wr=null,Tr=null;function Er(e){Bf(e,0)}function Dr(e){if($t(Nt(e)))return e}function Or(e,t){if(e===`change`)return t}var kr=!1;if(En){var Ar;if(En){var jr=`oninput`in document;if(!jr){var Mr=document.createElement(`div`);Mr.setAttribute(`oninput`,`return;`),jr=typeof Mr.oninput==`function`}Ar=jr}else Ar=!1;kr=Ar&&(!document.documentMode||9<document.documentMode)}function Nr(){wr&&(wr.detachEvent(`onpropertychange`,Pr),Tr=wr=null)}function Pr(e){if(e.propertyName===`value`&&Dr(Tr)){var t=[];Cr(t,Tr,e,yn(e)),wn(Er,t)}}function Fr(e,t,n){e===`focusin`?(Nr(),wr=t,Tr=n,wr.attachEvent(`onpropertychange`,Pr)):e===`focusout`&&Nr()}function Ir(e){if(e===`selectionchange`||e===`keyup`||e===`keydown`)return Dr(Tr)}function Lr(e,t){if(e===`click`)return Dr(t)}function Rr(e,t){if(e===`input`||e===`change`)return Dr(t)}function zr(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var Br=typeof Object.is==`function`?Object.is:zr;function Vr(e,t){if(Br(e,t))return!0;if(typeof e!=`object`||!e||typeof t!=`object`||!t)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!Fe.call(t,i)||!Br(e[i],t[i]))return!1}return!0}function Hr(e){if(e||=typeof document<`u`?document:void 0,e===void 0)return null;try{return e.activeElement||e.body}catch{return e.body}}function Ur(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Wr(e,t){var n=Ur(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}a:{for(;n;){if(n.nextSibling){n=n.nextSibling;break a}n=n.parentNode}n=void 0}n=Ur(n)}}function Gr(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Gr(e,t.parentNode):`contains`in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function Kr(e){e=e!=null&&e.ownerDocument!=null&&e.ownerDocument.defaultView!=null?e.ownerDocument.defaultView:window;for(var t=Hr(e.document);t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href==`string`}catch{n=!1}if(n)e=t.contentWindow;else break;t=Hr(e.document)}return t}function qr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t===`input`&&(e.type===`text`||e.type===`search`||e.type===`tel`||e.type===`url`||e.type===`password`)||t===`textarea`||e.contentEditable===`true`)}var Jr=En&&`documentMode`in document&&11>=document.documentMode,Yr=null,Xr=null,Zr=null,Qr=!1;function $r(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Qr||Yr==null||Yr!==Hr(r)||(r=Yr,`selectionStart`in r&&qr(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Zr&&Vr(Zr,r)||(Zr=r,r=qf(Xr,`onSelect`),0<r.length&&(t=new Rn(`onSelect`,`select`,null,t,n),e.push({event:t,listeners:r}),t.target=Yr)))}function ei(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n[`Webkit`+e]=`webkit`+t,n[`Moz`+e]=`moz`+t,n}var ti={animationend:ei(`Animation`,`AnimationEnd`),animationiteration:ei(`Animation`,`AnimationIteration`),animationstart:ei(`Animation`,`AnimationStart`),transitionrun:ei(`Transition`,`TransitionRun`),transitionstart:ei(`Transition`,`TransitionStart`),transitioncancel:ei(`Transition`,`TransitionCancel`),transitionend:ei(`Transition`,`TransitionEnd`)},ni={},ri={};En&&(ri=document.createElement(`div`).style,`AnimationEvent`in window||(delete ti.animationend.animation,delete ti.animationiteration.animation,delete ti.animationstart.animation),`TransitionEvent`in window||delete ti.transitionend.transition);function ii(e){if(ni[e])return ni[e];if(!ti[e])return e;var t=ti[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in ri)return ni[e]=t[n];return e}var ai=ii(`animationend`),oi=ii(`animationiteration`),si=ii(`animationstart`),ci=ii(`transitionrun`),li=ii(`transitionstart`),ui=ii(`transitioncancel`),di=ii(`transitionend`),fi=new Map,pi=`abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel`.split(` `);pi.push(`scrollEnd`);function mi(e,t){fi.set(e,t),zt(t,[e])}var hi=0;function gi(e,t){if(e.name!=null&&e.name!==`auto`)return e.name;if(t.autoName!==null)return t.autoName;e=vd.identifierPrefix;var n=hi++;return e=`_`+e+`t_`+n.toString(32)+`_`,t.autoName=e}function _i(e){if(e==null||typeof e==`string`)return e;var t=null,n=Ed;if(n!==null)for(var r=0;r<n.length;r++){var i=e[n[r]];if(i!=null){if(i===`none`)return`none`;t=t==null?i:t+(` `+i)}}return t??e.default}function vi(e,t){return e=_i(e),t=_i(t),t==null?e===`auto`?null:e:t===`auto`?null:t}var yi=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)},bi=[],xi=0,Si=0;function Ci(){for(var e=xi,t=Si=xi=0;t<e;){var n=bi[t];bi[t++]=null;var r=bi[t];bi[t++]=null;var i=bi[t];bi[t++]=null;var a=bi[t];if(bi[t++]=null,r!==null&&i!==null){var o=r.pending;o===null?i.next=i:(i.next=o.next,o.next=i),r.pending=i}a!==0&&Di(n,i,a)}}function wi(e,t,n,r){bi[xi++]=e,bi[xi++]=t,bi[xi++]=n,bi[xi++]=r,Si|=r,e.lanes|=r,e=e.alternate,e!==null&&(e.lanes|=r)}function Ti(e,t,n,r){return wi(e,t,n,r),Oi(e)}function Ei(e,t){return wi(e,null,null,t),Oi(e)}function Di(e,t,n){e.lanes|=n;var r=e.alternate;r!==null&&(r.lanes|=n);for(var i=!1,a=e.return;a!==null;)a.childLanes|=n,r=a.alternate,r!==null&&(r.childLanes|=n),a.tag===22&&(e=a.stateNode,e===null||e._visibility&1||(i=!0)),e=a,a=a.return;return e.tag===3?(a=e.stateNode,i&&t!==null&&(i=31-Ze(n),e=a.hiddenUpdates,r=e[i],r===null?e[i]=[t]:r.push(t),t.lane=n|536870912),a):null}function Oi(e){if(50<Dd)throw Dd=0,Od=null,Error(o(185));for(var t=e.return;t!==null;)e=t,t=e.return;return e.tag===3?e.stateNode:null}var ki={};function Ai(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function ji(e,t,n,r){return new Ai(e,t,n,r)}function Mi(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Ni(e,t){var n=e.alternate;return n===null?(n=ji(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&1206910976,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n.refCleanup=e.refCleanup,n}function Pi(e,t){e.flags&=1206910978;var n=e.alternate;return n===null?(e.childLanes=0,e.lanes=t,e.child=null,e.subtreeFlags=0,e.memoizedProps=null,e.memoizedState=null,e.updateQueue=null,e.dependencies=null,e.stateNode=null):(e.childLanes=n.childLanes,e.lanes=n.lanes,e.child=n.child,e.subtreeFlags=0,e.deletions=null,e.memoizedProps=n.memoizedProps,e.memoizedState=n.memoizedState,e.updateQueue=n.updateQueue,e.type=n.type,t=n.dependencies,e.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),e}function Fi(e,t,n,r,i,a){var s=0;if(r=e,typeof r==`function`)Mi(r)&&(s=1);else if(typeof r==`string`)s=qm(e,n,xe.current)?26:e===`html`||e===`head`||e===`body`?27:5;else a:switch(r){case j:return e=ji(31,n,t,i),e.elementType=j,e.lanes=a,e;case ne:return Ii(n.children,i,a,t);case O:s=8,i|=24;break;case re:return e=ji(12,n,t,i|2),e.elementType=re,e.lanes=a,e;case oe:return e=ji(13,n,t,i),e.elementType=oe,e.lanes=a,e;case se:return e=ji(19,n,t,i),e.elementType=se,e.lanes=a,e;case M:case ue:return e=i|32,e=ji(30,n,t,e),e.elementType=ue,e.lanes=a,e.stateNode={autoName:null,paired:null,clones:null,ref:null},e;default:if(typeof r==`object`&&r)switch(r.$$typeof){case k:s=10;break a;case ie:s=9;break a;case ae:s=11;break a;case ce:s=14;break a;case A:s=16,r=null;break a}s=29,n=Error(o(130,e===null?`null`:typeof e,``)),r=null}return t=ji(s,n,t,i),t.elementType=e,t.type=r,t.lanes=a,t}function Ii(e,t,n,r){return e=ji(7,e,r,t),e.lanes=n,e}function Li(e,t,n){return e=ji(6,e,null,t),e.lanes=n,e}function Ri(e){var t=ji(18,null,null,0);return t.stateNode=e,t}function zi(e,t,n){return t=ji(4,e.children===null?[]:e.children,e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}var Bi=new WeakMap;function Vi(e,t){if(typeof e==`object`&&e){var n=Bi.get(e);return n===void 0?(t={value:e,source:t,stack:Pe(t)},Bi.set(e,t),t):n}return{value:e,source:t,stack:Pe(t)}}var Hi=[],Ui=0,Wi=null,Gi=0,Ki=[],qi=0,Ji=null,Yi=1,Xi=``;function Zi(e,t){Hi[Ui++]=Gi,Hi[Ui++]=Wi,Wi=e,Gi=t}function Qi(e,t,n){Ki[qi++]=Yi,Ki[qi++]=Xi,Ki[qi++]=Ji,Ji=e;var r=Yi;e=Xi;var i=32-Ze(r)-1;r&=~(1<<i),n+=1;var a=32-Ze(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Yi=1<<32-Ze(t)+i|n<<i|r,Xi=a+e}else Yi=1<<a|n<<i|r,Xi=e}function $i(e){e.return!==null&&(Zi(e,1),Qi(e,1,0))}function ea(e){for(;e===Wi;)Wi=Hi[--Ui],Hi[Ui]=null,Gi=Hi[--Ui],Hi[Ui]=null;for(;e===Ji;)Ji=Ki[--qi],Ki[qi]=null,Xi=Ki[--qi],Ki[qi]=null,Yi=Ki[--qi],Ki[qi]=null}function ta(e,t){Ki[qi++]=Yi,Ki[qi++]=Xi,Ki[qi++]=Ji,Yi=t.id,Xi=t.overflow,Ji=e}var na=null,ra=null,V=!1,ia=null,aa=!1,oa=Error(o(519));function sa(e){throw pa(Vi(Error(o(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?`text`:`HTML`,``)),e)),oa}function ca(e){var t=e.stateNode,n=e.type,r=e.memoizedProps;switch(t[xt]=e,t[St]=r,n){case`dialog`:$(`cancel`,t),$(`close`,t);break;case`iframe`:case`object`:case`embed`:$(`load`,t);break;case`video`:case`audio`:for(n=0;n<Rf.length;n++)$(Rf[n],t);break;case`source`:$(`error`,t);break;case`img`:case`image`:case`link`:$(`error`,t),$(`load`,t);break;case`details`:$(`toggle`,t);break;case`input`:$(`invalid`,t),rn(t,r.value,r.defaultValue,r.checked,r.defaultChecked,r.type,r.name,!0);break;case`select`:$(`invalid`,t);break;case`textarea`:$(`invalid`,t),cn(t,r.value,r.defaultValue,r.children)}n=r.children,typeof n!=`string`&&typeof n!=`number`&&typeof n!=`bigint`||t.textContent===``+n||!0===r.suppressHydrationWarning||$f(t.textContent,n)?(r.popover!=null&&($(`beforetoggle`,t),$(`toggle`,t)),r.onScroll!=null&&$(`scroll`,t),r.onScrollEnd!=null&&$(`scrollend`,t),r.onClick!=null&&(t.onclick=_n),t=!0):t=!1,t||sa(e,!0)}function la(e){for(na=e.return;na;)switch(na.tag){case 5:case 31:case 13:aa=!1;return;case 27:case 3:aa=!0;return;default:na=na.return}}function ua(e){if(e!==na)return!1;if(!V)return la(e),V=!0,!1;var t=e.tag,n;if((n=t!==3&&t!==27)&&((n=t===5)&&(n=e.type,n=n===`form`||n===`button`||pp(e.type,e.memoizedProps)),n=!n),n&&ra&&sa(e),la(e),t===13){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(317));ra=dm(e)}else if(t===31){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(317));ra=dm(e)}else t===27?(t=ra,Sp(e.type)?(e=um,um=null,ra=e):ra=t):ra=na?lm(e.stateNode.nextSibling):null;return!0}function da(){ra=na=null,V=!1}function fa(){var e=ia;return e!==null&&(dd===null?dd=e:dd.push.apply(dd,e),ia=null),e}function pa(e){ia===null?ia=[e]:ia.push(e)}var ma=be(null),ha=null,ga=null;function _a(e,t,n){I(ma,t._currentValue),t._currentValue=n}function va(e){e._currentValue=ma.current,F(ma)}function ya(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)===t?r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t):(e.childLanes|=t,r!==null&&(r.childLanes|=t)),e===n)break;e=e.return}}function ba(e,t,n,r){var i=e.child;for(i!==null&&(i.return=e);i!==null;){var a=i.dependencies;if(a!==null){var s=i.child;a=a.firstContext;a:for(;a!==null;){var c=a;a=i;for(var l=0;l<t.length;l++)if(c.context===t[l]){a.lanes|=n,c=a.alternate,c!==null&&(c.lanes|=n),ya(a.return,n,e),r||(s=null);break a}a=c.next}}else if(i.tag===18){if(s=i.return,s===null)throw Error(o(341));s.lanes|=n,a=s.alternate,a!==null&&(a.lanes|=n),ya(s,n,e),s=null}else i.tag===13&&i.memoizedState!==null&&i.memoizedState.dehydrated===null?(i.lanes|=n,s=i.alternate,s!==null&&(s.lanes|=n),ya(i.return,n,e),s=i.child,s=s===null?null:s.sibling):s=i.child;if(s!==null)s.return=i;else for(s=i;s!==null;){if(s===e){s=null;break}if(i=s.sibling,i!==null){i.return=s.return,s=i;break}s=s.return}i=s}}function xa(e,t,n,r){e=null;for(var i=t,a=!1;i!==null;){if(!a){if(i.flags&524288)a=!0;else if(i.flags&262144)break}if(i.tag===10){var s=i.alternate;if(s===null)throw Error(o(387));if(s=s.memoizedProps,s!==null){var c=i.type;Br(i.pendingProps.value,s.value)||(e===null?e=[c]:e.push(c))}}else if(i===we.current){if(s=i.alternate,s===null)throw Error(o(387));s.memoizedState.memoizedState!==i.memoizedState.memoizedState&&(e===null?e=[sh]:e.push(sh))}i=i.return}return e!==null&&ba(t,e,n,r),t.flags|=262144,e!==null}function Sa(e){for(e=e.firstContext;e!==null;){if(!Br(e.context._currentValue,e.memoizedValue))return!0;e=e.next}return!1}function Ca(e){ha=e,ga=null,e=e.dependencies,e!==null&&(e.firstContext=null)}function wa(e){return Ea(ha,e)}function Ta(e,t){return ha===null&&Ca(e),Ea(e,t)}function Ea(e,t){var n=t._currentValue;if(t={context:t,memoizedValue:n,next:null},ga===null){if(e===null)throw Error(o(308));ga=t,e.dependencies={lanes:0,firstContext:t},e.flags|=524288}else ga=ga.next=t;return n}var Da=typeof AbortController<`u`?AbortController:function(){var e=[],t=this.signal={aborted:!1,addEventListener:function(t,n){e.push(n)}};this.abort=function(){t.aborted=!0,e.forEach(function(e){return e()})}},Oa=n.unstable_scheduleCallback,ka=n.unstable_NormalPriority,Aa={$$typeof:k,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function ja(){return{controller:new Da,data:new Map,refCount:0}}function Ma(e){e.refCount--,e.refCount===0&&Oa(ka,function(){e.controller.abort()})}function Na(e,t){if(e.pendingLanes&4194048){var n=e.transitionTypes;for(n===null&&(n=e.transitionTypes=[]),e=0;e<t.length;e++){var r=t[e];n.indexOf(r)===-1&&n.push(r)}}}var Pa=null;function Fa(e){var t=e.transitionTypes;return e.transitionTypes=null,t}var Ia=null,La=0,Ra=0,za=null;function Ba(e,t){if(Ia===null){var n=Ia=[];La=0,Ra=Nf(),za={status:`pending`,value:void 0,then:function(e){n.push(e)}}}return La++,t.then(Va,Va),t}function Va(){if(--La===0&&(Pa=null,Ia!==null)){za!==null&&(za.status=`fulfilled`);var e=Ia;Ia=null,Ra=0,za=null;for(var t=0;t<e.length;t++)(0,e[t])()}}function Ha(e,t){var n=[],r={status:`pending`,value:null,reason:null,then:function(e){n.push(e)}};return e.then(function(){r.status=`fulfilled`,r.value=t;for(var e=0;e<n.length;e++)(0,n[e])(t)},function(e){for(r.status=`rejected`,r.reason=e,e=0;e<n.length;e++)(0,n[e])(void 0)}),r}var Ua=N.S;N.S=function(e,t){if(md=ze(),typeof t==`object`&&t&&typeof t.then==`function`&&Ba(e,t),Pa!==null)for(var n=yf;n!==null;)Na(n,Pa),n=n.next;if(n=e.types,n!==null){for(var r=yf;r!==null;)Na(r,n),r=r.next;if(Ra!==0){r=Pa,r===null&&(r=Pa=[]);for(var i=0;i<n.length;i++){var a=n[i];r.indexOf(a)===-1&&r.push(a)}}}Ua!==null&&Ua(e,t)};var Wa=be(null);function Ga(){var e=Wa.current;return e===null?$u.pooledCache:e}function Ka(e,t){t===null?I(Wa,Wa.current):I(Wa,t.pool)}function qa(){var e=Ga();return e===null?null:{parent:Aa._currentValue,pool:e}}var Ja=Error(o(460)),H=Error(o(474)),Ya=Error(o(542)),Xa={then:function(){}};function Za(e){return e=e.status,e===`fulfilled`||e===`rejected`}function Qa(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(_n,_n),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,no(e),e===void 0&&!(`reason`in t)?Error(o(600)):e;default:if(typeof t.status==`string`)t.then(_n,_n);else{if(e=$u,e!==null&&100<e.shellSuspendCounter)throw Error(o(482));e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})}switch(t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,no(e),e}throw eo=t,Ja}}function $a(e){try{var t=e._init;return t(e._payload)}catch(e){throw typeof e==`object`&&e&&typeof e.then==`function`?(eo=e,Ja):e}}var eo=null;function to(){if(eo===null)throw Error(o(459));var e=eo;return eo=null,e}function no(e){if(e===Ja||e===Ya)throw Error(o(483))}var ro=null,io=0;function ao(e){var t=io;return io+=1,ro===null&&(ro=[]),Qa(ro,e,t)}function oo(e,t){t=t.props.ref,e.ref=t===void 0?null:t}function so(e,t){throw t.$$typeof===ee?Error(o(525)):(e=Object.prototype.toString.call(t),Error(o(31,e===`[object Object]`?`object with keys {`+Object.keys(t).join(`, `)+`}`:e)))}function co(e){function t(t,n){if(e){var r=t.deletions;r===null?(t.deletions=[n],t.flags|=16):r.push(n)}}function n(n,r){if(!e)return null;for(;r!==null;)t(n,r),r=r.sibling;return null}function r(e){for(var t=new Map;e!==null;)e.key===null?t.set(e.index,e):t.set(e.key,e),e=e.sibling;return t}function i(e,t){return e=Ni(e,t),e.index=0,e.sibling=null,e}function a(t,n,r){return t.index=r,e?(r=t.alternate,r===null?(t.flags|=134217730,n):(r=r.index,r<n?(t.flags|=2,n):r)):(t.flags|=1048576,n)}function s(t){return e&&t.alternate===null&&(t.flags|=134217730),t}function c(e,t,n,r){return t===null||t.tag!==6?(t=Li(n,e.mode,r),t.return=e,t):(t=i(t,n),t.return=e,t)}function l(e,t,n,r){var a=n.type;return a===ne?(e=d(e,t,n.props.children,r,n.key),oo(e,n),e):t!==null&&(t.elementType===a||typeof a==`object`&&a&&a.$$typeof===A&&$a(a)===t.type)?(t=i(t,n.props),oo(t,n),t.return=e,t):(t=Fi(n.type,n.key,n.props,null,e.mode,r),oo(t,n),t.return=e,t)}function u(e,t,n,r){return t===null||t.tag!==4||t.stateNode.containerInfo!==n.containerInfo||t.stateNode.implementation!==n.implementation?(t=zi(n,e.mode,r),t.return=e,t):(t=i(t,n.children||[]),t.return=e,t)}function d(e,t,n,r,a){return t===null||t.tag!==7?(t=Ii(n,e.mode,r,a),t.return=e,t):(t=i(t,n),t.return=e,t)}function f(e,t,n){if(typeof t==`string`&&t!==``||typeof t==`number`||typeof t==`bigint`)return t=Li(``+t,e.mode,n),t.return=e,t;if(typeof t==`object`&&t){switch(t.$$typeof){case te:return n=Fi(t.type,t.key,t.props,null,e.mode,n),oo(n,t),n.return=e,n;case D:return t=zi(t,e.mode,n),t.return=e,t;case A:return t=$a(t),f(e,t,n)}if(ge(t)||pe(t))return t=Ii(t,e.mode,n,null),t.return=e,t;if(typeof t.then==`function`)return f(e,ao(t),n);if(t.$$typeof===k)return f(e,Ta(e,t),n);so(e,t)}return null}function p(e,t,n,r){var i=t===null?null:t.key;if(typeof n==`string`&&n!==``||typeof n==`number`||typeof n==`bigint`)return i===null?c(e,t,``+n,r):null;if(typeof n==`object`&&n){switch(n.$$typeof){case te:return n.key===i?l(e,t,n,r):null;case D:return n.key===i?u(e,t,n,r):null;case A:return n=$a(n),p(e,t,n,r)}if(ge(n)||pe(n))return i===null?d(e,t,n,r,null):null;if(typeof n.then==`function`)return p(e,t,ao(n),r);if(n.$$typeof===k)return p(e,t,Ta(e,n),r);so(e,n)}return null}function m(e,t,n,r,i){if(typeof r==`string`&&r!==``||typeof r==`number`||typeof r==`bigint`)return e=e.get(n)||null,c(t,e,``+r,i);if(typeof r==`object`&&r){switch(r.$$typeof){case te:return e=e.get(r.key===null?n:r.key)||null,l(t,e,r,i);case D:return e=e.get(r.key===null?n:r.key)||null,u(t,e,r,i);case A:return r=$a(r),m(e,t,n,r,i)}if(ge(r)||pe(r))return e=e.get(n)||null,d(t,e,r,i,null);if(typeof r.then==`function`)return m(e,t,n,ao(r),i);if(r.$$typeof===k)return m(e,t,n,Ta(t,r),i);so(t,r)}return null}function h(i,o,s,c){for(var l=null,u=null,d=o,h=o=0,g=null;d!==null&&h<s.length;h++){d.index>h?(g=d,d=null):g=d.sibling;var _=p(i,d,s[h],c);if(_===null){d===null&&(d=g);break}e&&d&&_.alternate===null&&t(i,d),o=a(_,o,h),u===null?l=_:u.sibling=_,u=_,d=g}if(h===s.length)return n(i,d),V&&Zi(i,h),l;if(d===null){for(;h<s.length;h++)d=f(i,s[h],c),d!==null&&(o=a(d,o,h),u===null?l=d:u.sibling=d,u=d);return V&&Zi(i,h),l}for(d=r(d);h<s.length;h++)g=m(d,i,h,s[h],c),g!==null&&(e&&(_=g.alternate,_!==null&&d.delete(_.key===null?h:_.key)),o=a(g,o,h),u===null?l=g:u.sibling=g,u=g);return e&&d.forEach(function(e){return t(i,e)}),V&&Zi(i,h),l}function g(i,s,c,l){if(c==null)throw Error(o(151));for(var u=null,d=null,h=s,g=s=0,_=null,v=c.next();h!==null&&!v.done;g++,v=c.next()){h.index>g?(_=h,h=null):_=h.sibling;var y=p(i,h,v.value,l);if(y===null){h===null&&(h=_);break}e&&h&&y.alternate===null&&t(i,h),s=a(y,s,g),d===null?u=y:d.sibling=y,d=y,h=_}if(v.done)return n(i,h),V&&Zi(i,g),u;if(h===null){for(;!v.done;g++,v=c.next())v=f(i,v.value,l),v!==null&&(s=a(v,s,g),d===null?u=v:d.sibling=v,d=v);return V&&Zi(i,g),u}for(h=r(h);!v.done;g++,v=c.next())v=m(h,i,g,v.value,l),v!==null&&(e&&(_=v.alternate,_!==null&&h.delete(_.key===null?g:_.key)),s=a(v,s,g),d===null?u=v:d.sibling=v,d=v);return e&&h.forEach(function(e){return t(i,e)}),V&&Zi(i,g),u}function _(e,r,a,c){if(typeof a==`object`&&a&&a.type===ne&&a.key===null&&a.props.ref===void 0&&(a=a.props.children),typeof a==`object`&&a){switch(a.$$typeof){case te:a:{for(var l=a.key;r!==null;){if(r.key===l){if(l=a.type,l===ne){if(r.tag===7){n(e,r.sibling),c=i(r,a.props.children),oo(c,a),c.return=e,e=c;break a}}else if(r.elementType===l||typeof l==`object`&&l&&l.$$typeof===A&&$a(l)===r.type){n(e,r.sibling),c=i(r,a.props),oo(c,a),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}a.type===ne?(c=Ii(a.props.children,e.mode,c,a.key),oo(c,a),c.return=e,e=c):(c=Fi(a.type,a.key,a.props,null,e.mode,c),oo(c,a),c.return=e,e=c)}return s(e);case D:a:{for(l=a.key;r!==null;){if(r.key===l){if(r.tag===4&&r.stateNode.containerInfo===a.containerInfo&&r.stateNode.implementation===a.implementation){n(e,r.sibling),c=i(r,a.children||[]),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}c=zi(a,e.mode,c),c.return=e,e=c}return s(e);case A:return a=$a(a),_(e,r,a,c)}if(ge(a))return h(e,r,a,c);if(pe(a)){if(l=pe(a),typeof l!=`function`)throw Error(o(150));return a=l.call(a),g(e,r,a,c)}if(typeof a.then==`function`)return _(e,r,ao(a),c);if(a.$$typeof===k)return _(e,r,Ta(e,a),c);so(e,a)}return typeof a==`string`&&a!==``||typeof a==`number`||typeof a==`bigint`?(a=``+a,r!==null&&r.tag===6?(n(e,r.sibling),c=i(r,a),c.return=e,e=c):(n(e,r),c=Li(a,e.mode,c),c.return=e,e=c),s(e)):n(e,r)}return function(e,t,n,r){try{io=0;var i=_(e,t,n,r);return ro=null,i}catch(t){if(t===Ja||t===Ya)throw t;var a=ji(29,t,null,e.mode);return a.lanes=r,a.return=e,a}}}var lo=co(!0),uo=co(!1),fo=!1;function po(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function mo(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,callbacks:null})}function ho(e){return{lane:e,tag:0,payload:null,callback:null,next:null}}function go(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,q&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,t=Oi(e),Di(e,null,n),t}return wi(e,r,t,n),Oi(e)}function _o(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,n&4194048)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,mt(e,n)}}function vo(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,callbacks:r.callbacks},e.updateQueue=n}else e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}var yo=!1;function bo(){if(yo){var e=za;if(e!==null)throw e}}function xo(e,t,n,r){yo=!1;var i=e.updateQueue;fo=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var c=s,l=c.next;c.next=null,o===null?a=l:o.next=l,o=c;var u=e.alternate;u!==null&&(u=u.updateQueue,s=u.lastBaseUpdate,s!==o&&(s===null?u.firstBaseUpdate=l:s.next=l,u.lastBaseUpdate=c))}if(a!==null){var d=i.baseState;o=0,u=l=c=null,s=a;do{var f=s.lane&-536870913,p=f!==s.lane;if(p?(Y&f)===f:(r&f)===f){f!==0&&f===Ra&&(yo=!0),u!==null&&(u=u.next={lane:0,tag:s.tag,payload:s.payload,callback:null,next:null});a:{var m=e,h=s;f=t;var g=n;switch(h.tag){case 1:if(m=h.payload,typeof m==`function`){d=m.call(g,d,f);break a}d=m;break a;case 3:m.flags=m.flags&-65537|128;case 0:if(m=h.payload,f=typeof m==`function`?m.call(g,d,f):m,f==null)break a;d=E({},d,f);break a;case 2:fo=!0}}f=s.callback,f!==null&&(e.flags|=64,p&&(e.flags|=8192),p=i.callbacks,p===null?i.callbacks=[f]:p.push(f))}else p={lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},u===null?(l=u=p,c=d):u=u.next=p,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(1);u===null&&(c=d),i.baseState=c,i.firstBaseUpdate=l,i.lastBaseUpdate=u,a===null&&(i.shared.lanes=0),Z|=o,e.lanes=o,e.memoizedState=d}}function So(e,t){if(typeof e!=`function`)throw Error(o(191,e));e.call(t)}function Co(e,t){var n=e.callbacks;if(n!==null)for(e.callbacks=null,e=0;e<n.length;e++)So(n[e],t)}var wo=be(null),To=be(0);function Eo(e,t){e=id,I(To,e),I(wo,t),id=e|t.baseLanes}function Do(){I(To,id),I(wo,wo.current)}function Oo(){id=To.current,F(wo),F(To)}var ko=be(null),Ao=null;function jo(e){var t=e.alternate;I(Fo,Fo.current&1),I(ko,e),Ao===null&&(t===null||wo.current!==null||t.memoizedState!==null)&&(Ao=e)}function Mo(e){I(Fo,Fo.current),I(ko,e),Ao===null&&(Ao=e)}function No(e){e.tag===22?(I(Fo,Fo.current),I(ko,e),Ao===null&&(Ao=e)):U()}function U(){I(Fo,Fo.current),I(ko,ko.current)}function Po(e){F(ko),Ao===e&&(Ao=null),F(Fo)}var Fo=be(0);function Io(e,t){I(ko,ko.current),I(Fo,t)}function Lo(e){F(Fo),F(ko),Ao===e&&(Ao=null)}function Ro(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||om(n)||sm(n)))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==`independent`){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var zo=0,W=null,G=null,Bo=null,Vo=!1,Ho=!1,Uo=!1,Wo=0,Go=0,Ko=null,qo=0;function Jo(){throw Error(o(321))}function Yo(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Br(e[n],t[n]))return!1;return!0}function Xo(e,t,n,r,i,a){return zo=a,W=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,N.H=e===null||e.memoizedState===null?pc:mc,Uo=!1,a=n(r,i),Uo=!1,Ho&&(a=Qo(t,n,r,i)),Zo(e),a}function Zo(e){N.H=fc;var t=G!==null&&G.next!==null;if(zo=0,Bo=G=W=null,Vo=!1,Go=0,Ko=null,t)throw Error(o(300));e===null||jc||(e=e.dependencies,e!==null&&Sa(e)&&(jc=!0))}function Qo(e,t,n,r){W=e;var i=0;do{if(Ho&&(Ko=null),Go=0,Ho=!1,25<=i)throw Error(o(301));if(i+=1,Bo=G=null,e.updateQueue!=null){var a=e.updateQueue;a.lastEffect=null,a.events=null,a.stores=null,a.memoCache!=null&&(a.memoCache.index=0)}N.H=hc,a=t(n,r)}while(Ho);return a}function $o(){var e=N.H,t=e.useState()[0];return t=typeof t.then==`function`?os(t):t,e=e.useState()[0],(G===null?null:G.memoizedState)!==e&&(W.flags|=1024),t}function es(){var e=Wo!==0;return Wo=0,e}function ts(e,t,n){t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~n}function ns(e){if(Vo){for(e=e.memoizedState;e!==null;){var t=e.queue;t!==null&&(t.pending=null),e=e.next}Vo=!1}zo=0,Bo=G=W=null,Ho=!1,Go=Wo=0,Ko=null}function rs(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Bo===null?W.memoizedState=Bo=e:Bo=Bo.next=e,Bo}function is(){if(G===null){var e=W.alternate;e=e===null?null:e.memoizedState}else e=G.next;var t=Bo===null?W.memoizedState:Bo.next;if(t!==null)Bo=t,G=e;else{if(e===null)throw W.alternate===null?Error(o(467)):Error(o(310));G=e,e={memoizedState:G.memoizedState,baseState:G.baseState,baseQueue:G.baseQueue,queue:G.queue,next:null},Bo===null?W.memoizedState=Bo=e:Bo=Bo.next=e}return Bo}function as(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function os(e){var t=Go;return Go+=1,Ko===null&&(Ko=[]),e=Qa(Ko,e,t),t=W,(Bo===null?t.memoizedState:Bo.next)===null&&(t=t.alternate,N.H=t===null||t.memoizedState===null?pc:mc),e}function ss(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return os(e);if(e.$$typeof===de)return;if(e.$$typeof===k)return wa(e)}throw Error(o(438,String(e)))}function cs(e){var t=null,n=W.updateQueue;if(n!==null&&(t=n.memoCache),t==null){var r=W.alternate;r!==null&&(r=r.updateQueue,r!==null&&(r=r.memoCache,r!=null&&(t={data:r.data.map(function(e){return e.slice()}),index:0})))}if(t??={data:[],index:0},n===null&&(n=as(),W.updateQueue=n),n.memoCache=t,n=t.data[t.index],n===void 0)for(n=t.data[t.index]=Array(e),r=0;r<e;r++)n[r]=le;return t.index++,n}function ls(e,t){return typeof t==`function`?t(e):t}function us(e){return ds(is(),G,e)}function ds(e,t,n){var r=e.queue;if(r===null)throw Error(o(311));r.lastRenderedReducer=n;var i=e.baseQueue,a=r.pending;if(a!==null){if(i!==null){var s=i.next;i.next=a.next,a.next=s}t.baseQueue=i=a,r.pending=null}if(a=e.baseState,i===null)e.memoizedState=a;else{t=i.next;var c=s=null,l=null,u=t,d=!1;do{var f=u.lane&-536870913;if(f===u.lane?(zo&f)===f:(Y&f)===f){var p=u.revertLane;if(p===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),f===Ra&&(d=!0);else if((zo&p)===p){u=u.next,p===Ra&&(d=!0);continue}else f={lane:0,revertLane:u.revertLane,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=f,s=a):l=l.next=f,W.lanes|=p,Z|=p;f=u.action,Uo&&n(a,f),a=u.hasEagerState?u.eagerState:n(a,f)}else p={lane:f,revertLane:u.revertLane,gesture:u.gesture,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=p,s=a):l=l.next=p,W.lanes|=f,Z|=f;u=u.next}while(u!==null&&u!==t);if(l===null?s=a:l.next=c,!Br(a,e.memoizedState)&&(jc=!0,d&&(n=za,n!==null)))throw n;e.memoizedState=a,e.baseState=s,e.baseQueue=l,r.lastRenderedState=a}return i===null&&(r.lanes=0),[e.memoizedState,r.dispatch]}function fs(e){var t=is(),n=t.queue;if(n===null)throw Error(o(311));n.lastRenderedReducer=e;var r=n.dispatch,i=n.pending,a=t.memoizedState;if(i!==null){n.pending=null;var s=i=i.next;do a=e(a,s.action),s=s.next;while(s!==i);Br(a,t.memoizedState)||(jc=!0),t.memoizedState=a,t.baseQueue===null&&(t.baseState=a),n.lastRenderedState=a}return[a,r]}function ps(e,t,n){var r=W,i=is(),a=V;if(a){if(n===void 0)throw Error(o(407));n=n()}else n=t();var s=!Br((G||i).memoizedState,n);if(s&&(i.memoizedState=n,jc=!0),i=i.queue,Rs(gs.bind(null,r,i,e),[e]),e=i.getSnapshot!==t||s||Bo!==null&&!!(Bo.memoizedState.tag&1),Ns(e?9:8,{destroy:void 0},hs.bind(null,r,i,n,t),null),e){if(r.flags|=2048,$u===null)throw Error(o(349));a||zo&127||ms(r,t,n)}return n}function ms(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=W.updateQueue,t===null?(t=as(),W.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function hs(e,t,n,r){t.value=n,t.getSnapshot=r,_s(t)&&vs(e)}function gs(e,t,n){return n(function(){_s(t)&&vs(e)})}function _s(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Br(e,n)}catch{return!0}}function vs(e){var t=Ei(e,2);t!==null&&Md(t,e,2)}function ys(e){var t=rs();if(typeof e==`function`){var n=e;if(e=n(),Uo){Xe(!0);try{n()}finally{Xe(!1)}}}return t.memoizedState=t.baseState=e,t.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:ls,lastRenderedState:e},t}function bs(e,t,n,r){return e.baseState=n,ds(e,G,typeof r==`function`?r:ls)}function xs(e,t,n,r,i){if(lc(e))throw Error(o(485));if(e=t.action,e!==null){var a={payload:i,action:e,next:null,isTransition:!0,status:`pending`,value:null,reason:null,listeners:[],then:function(e){a.listeners.push(e)}};N.T===null?a.isTransition=!1:n(!0),r(a),n=t.pending,n===null?(a.next=t.pending=a,Ss(t,a)):(a.next=n.next,t.pending=n.next=a)}}function Ss(e,t){var n=t.action,r=t.payload,i=e.state;if(t.isTransition){var a=N.T,o={};o.types=a===null?null:a.types,N.T=o;try{var s=n(i,r),c=N.S;c!==null&&c(o,s),Cs(e,t,s)}catch(n){Ts(e,t,n)}finally{a!==null&&o.types!==null&&(a.types=o.types),N.T=a}}else try{a=n(i,r),Cs(e,t,a)}catch(n){Ts(e,t,n)}}function Cs(e,t,n){typeof n==`object`&&n&&typeof n.then==`function`?n.then(function(n){ws(e,t,n)},function(n){return Ts(e,t,n)}):ws(e,t,n)}function ws(e,t,n){t.status=`fulfilled`,t.value=n,Es(t),e.state=n,t=e.pending,t!==null&&(n=t.next,n===t?e.pending=null:(n=n.next,t.next=n,Ss(e,n)))}function Ts(e,t,n){var r=e.pending;if(e.pending=null,r!==null){r=r.next;do t.status=`rejected`,t.reason=n,Es(t),t=t.next;while(t!==r)}e.action=null}function Es(e){e=e.listeners;for(var t=0;t<e.length;t++)(0,e[t])()}function Ds(e,t){return t}function Os(e,t){if(V){var n=$u.formState;if(n!==null){a:{var r=W;if(V){if(ra){b:{for(var i=ra,a=aa;i.nodeType!==8;)if(!a||(i=lm(i.nextSibling),i===null)){i=null;break b}a=i.data,i=a===`F!`||a===`F`?i:null}if(i){ra=lm(i.nextSibling),r=i.data===`F!`;break a}}sa(r)}r=!1}r&&(t=n[0])}}return n=rs(),n.memoizedState=n.baseState=t,r={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Ds,lastRenderedState:t},n.queue=r,n=oc.bind(null,W,r),r.dispatch=n,r=ys(!1),a=cc.bind(null,W,!1,r.queue),r=rs(),i={state:t,dispatch:null,action:e,pending:null},r.queue=i,n=xs.bind(null,W,i,a,n),i.dispatch=n,r.memoizedState=e,[t,n,!1]}function ks(e){return As(is(),G,e)}function As(e,t,n){if(t=ds(e,t,Ds)[0],e=us(ls)[0],typeof t==`object`&&t&&typeof t.then==`function`)try{var r=os(t)}catch(e){throw e===Ja?Ya:e}else r=t;t=is();var i=t.queue,a=i.dispatch;return n!==t.memoizedState&&(W.flags|=2048,Ns(9,{destroy:void 0},js.bind(null,i,n),null)),[r,a,e]}function js(e,t){e.action=t}function Ms(e){var t=is(),n=G;if(n!==null)return As(t,n,e);is(),t=t.memoizedState,n=is();var r=n.queue.dispatch;return n.memoizedState=e,[t,r,!1]}function Ns(e,t,n,r){return e={tag:e,create:n,deps:r,inst:t,next:null},t=W.updateQueue,t===null&&(t=as(),W.updateQueue=t),n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e),e}function Ps(){return is().memoizedState}function Fs(e,t,n,r){var i=rs();W.flags|=e,i.memoizedState=Ns(1|t,{destroy:void 0},n,r===void 0?null:r)}function Is(e,t,n,r){var i=is();r=r===void 0?null:r;var a=i.memoizedState.inst;G!==null&&r!==null&&Yo(r,G.memoizedState.deps)?i.memoizedState=Ns(t,a,n,r):(W.flags|=e,i.memoizedState=Ns(1|t,a,n,r))}function Ls(e,t){Fs(8390656,8,e,t)}function Rs(e,t){Is(2048,8,e,t)}function zs(e){W.flags|=4;var t=W.updateQueue;if(t===null)t=as(),W.updateQueue=t,t.events=[e];else{var n=t.events;n===null?t.events=[e]:n.push(e)}}function Bs(e){var t=is().memoizedState;return zs({ref:t,nextImpl:e}),function(){if(q&2)throw Error(o(440));return t.impl.apply(void 0,arguments)}}function Vs(e,t){return Is(4,2,e,t)}function Hs(e,t){return Is(4,4,e,t)}function Us(e,t){if(typeof t==`function`){e=e();var n=t(e);return function(){typeof n==`function`?n():t(null)}}if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Ws(e,t,n){n=n==null?null:n.concat([e]),Is(4,4,Us.bind(null,t,e),n)}function Gs(){}function Ks(e,t){var n=is();t=t===void 0?null:t;var r=n.memoizedState;return t!==null&&Yo(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function qs(e,t){var n=is();t=t===void 0?null:t;var r=n.memoizedState;if(t!==null&&Yo(t,r[1]))return r[0];if(r=e(),Uo){Xe(!0);try{e()}finally{Xe(!1)}}return n.memoizedState=[r,t],r}function Js(e,t,n){return n===void 0||zo&1073741824&&!(Y&261930)?e.memoizedState=t:(e.memoizedState=n,e=Ad(),W.lanes|=e,Z|=e,n)}function Ys(e,t,n,r){return Br(n,t)?n:wo.current===null?!(zo&106)||zo&1073741824&&!(Y&261930)?(jc=!0,e.memoizedState=n):(e=Ad(),W.lanes|=e,Z|=e,t):(e=Js(e,n,r),Br(e,t)||(jc=!0),e)}function Xs(e,t,n,r,i){var a=P.p;P.p=a!==0&&8>a?a:8;var o=N.T,s={};s.types=o===null?null:o.types,N.T=s,cc(e,!1,t,n);try{var c=i(),l=N.S;l!==null&&l(s,c),typeof c==`object`&&c&&typeof c.then==`function`?sc(e,t,Ha(c,r),kd(e)):sc(e,t,r,kd(e))}catch(n){sc(e,t,{then:function(){},status:`rejected`,reason:n},kd())}finally{P.p=a,o!==null&&s.types!==null&&(o.types=s.types),N.T=o}}function Zs(){}function Qs(e,t,n,r){if(e.tag!==5)throw Error(o(476));var i=$s(e).queue;Xs(e,i,t,_e,n===null?Zs:function(){return ec(e),n(r)})}function $s(e){var t=e.memoizedState;if(t!==null)return t;t={memoizedState:_e,baseState:_e,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ls,lastRenderedState:_e},next:null};var n={};return t.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ls,lastRenderedState:n},next:null},e.memoizedState=t,e=e.alternate,e!==null&&(e.memoizedState=t),t}function ec(e){var t=$s(e);t.next===null&&(t=e.alternate.memoizedState),sc(e,t.next.queue,{},kd())}function tc(){return wa(sh)}function nc(){return is().memoizedState}function rc(){return is().memoizedState}function ic(e){for(var t=e.return;t!==null;){switch(t.tag){case 24:case 3:var n=kd();e=ho(n);var r=go(t,e,n);r!==null&&(Md(r,t,n),_o(r,t,n)),t={cache:ja()},e.payload=t;return}t=t.return}}function ac(e,t,n){var r=kd();n={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},lc(e)?uc(t,n):(n=Ti(e,t,n,r),n!==null&&(Md(n,e,r),dc(n,t,r)))}function oc(e,t,n){sc(e,t,n,kd())}function sc(e,t,n,r){var i={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(lc(e))uc(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Br(s,o))return wi(e,t,i,0),$u===null&&Ci(),!1}catch{}if(n=Ti(e,t,i,r),n!==null)return Md(n,e,r),dc(n,t,r),!0}return!1}function cc(e,t,n,r){if(r={lane:2,revertLane:Nf(),gesture:null,action:r,hasEagerState:!1,eagerState:null,next:null},lc(e)){if(t)throw Error(o(479))}else t=Ti(e,n,r,2),t!==null&&Md(t,e,2)}function lc(e){var t=e.alternate;return e===W||t!==null&&t===W}function uc(e,t){Ho=Vo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function dc(e,t,n){if(n&4194048){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,mt(e,n)}}var fc={readContext:wa,use:ss,useCallback:Jo,useContext:Jo,useEffect:Jo,useImperativeHandle:Jo,useLayoutEffect:Jo,useInsertionEffect:Jo,useMemo:Jo,useReducer:Jo,useRef:Jo,useState:Jo,useDebugValue:Jo,useDeferredValue:Jo,useTransition:Jo,useSyncExternalStore:Jo,useId:Jo,useHostTransitionStatus:Jo,useFormState:Jo,useActionState:Jo,useOptimistic:Jo,useMemoCache:Jo,useCacheRefresh:Jo,useEffectEvent:Jo},pc={readContext:wa,use:ss,useCallback:function(e,t){return rs().memoizedState=[e,t===void 0?null:t],e},useContext:wa,useEffect:Ls,useImperativeHandle:function(e,t,n){n=n==null?null:n.concat([e]),Fs(4194308,4,Us.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Fs(4194308,4,e,t)},useInsertionEffect:function(e,t){Fs(4,2,e,t)},useMemo:function(e,t){var n=rs();t=t===void 0?null:t;var r=e();if(Uo){Xe(!0);try{e()}finally{Xe(!1)}}return n.memoizedState=[r,t],r},useReducer:function(e,t,n){var r=rs();if(n!==void 0){var i=n(t);if(Uo){Xe(!0);try{n(t)}finally{Xe(!1)}}}else i=t;return r.memoizedState=r.baseState=i,e={pending:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:i},r.queue=e,e=e.dispatch=ac.bind(null,W,e),[r.memoizedState,e]},useRef:function(e){var t=rs();return e={current:e},t.memoizedState=e},useState:function(e){e=ys(e);var t=e.queue,n=oc.bind(null,W,t);return t.dispatch=n,[e.memoizedState,n]},useDebugValue:Gs,useDeferredValue:function(e,t){return Js(rs(),e,t)},useTransition:function(){var e=ys(!1);return e=Xs.bind(null,W,e.queue,!0,!1),rs().memoizedState=e,[!1,e]},useSyncExternalStore:function(e,t,n){var r=W,i=rs();if(V){if(n===void 0)throw Error(o(407));n=n()}else{if(n=t(),$u===null)throw Error(o(349));Y&127||ms(r,t,n)}i.memoizedState=n;var a={value:n,getSnapshot:t};return i.queue=a,Ls(gs.bind(null,r,a,e),[e]),r.flags|=2048,Ns(9,{destroy:void 0},hs.bind(null,r,a,n,t),null),n},useId:function(){var e=rs(),t=$u.identifierPrefix;if(V){var n=Xi,r=Yi;n=(r&~(1<<32-Ze(r)-1)).toString(32)+n,t=`_`+t+`R_`+n,n=Wo++,0<n&&(t+=`H`+n.toString(32)),t+=`_`}else n=qo++,t=`_`+t+`r_`+n.toString(32)+`_`;return e.memoizedState=t},useHostTransitionStatus:tc,useFormState:Os,useActionState:Os,useOptimistic:function(e){var t=rs();t.memoizedState=t.baseState=e;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return t.queue=n,t=cc.bind(null,W,!0,n),n.dispatch=t,[e,t]},useMemoCache:cs,useCacheRefresh:function(){return rs().memoizedState=ic.bind(null,W)},useEffectEvent:function(e){var t=rs(),n={impl:e};return t.memoizedState=n,function(){if(q&2)throw Error(o(440));return n.impl.apply(void 0,arguments)}}},mc={readContext:wa,use:ss,useCallback:Ks,useContext:wa,useEffect:Rs,useImperativeHandle:Ws,useInsertionEffect:Vs,useLayoutEffect:Hs,useMemo:qs,useReducer:us,useRef:Ps,useState:function(){return us(ls)},useDebugValue:Gs,useDeferredValue:function(e,t){return Ys(is(),G.memoizedState,e,t)},useTransition:function(){var e=us(ls)[0],t=is().memoizedState;return[typeof e==`boolean`?e:os(e),t]},useSyncExternalStore:ps,useId:nc,useHostTransitionStatus:tc,useFormState:ks,useActionState:ks,useOptimistic:function(e,t){return bs(is(),G,e,t)},useMemoCache:cs,useCacheRefresh:rc,useEffectEvent:Bs},hc={readContext:wa,use:ss,useCallback:Ks,useContext:wa,useEffect:Rs,useImperativeHandle:Ws,useInsertionEffect:Vs,useLayoutEffect:Hs,useMemo:qs,useReducer:fs,useRef:Ps,useState:function(){return fs(ls)},useDebugValue:Gs,useDeferredValue:function(e,t){var n=is();return G===null?Js(n,e,t):Ys(n,G.memoizedState,e,t)},useTransition:function(){var e=fs(ls)[0],t=is().memoizedState;return[typeof e==`boolean`?e:os(e),t]},useSyncExternalStore:ps,useId:nc,useHostTransitionStatus:tc,useFormState:Ms,useActionState:Ms,useOptimistic:function(e,t){var n=is();return G===null?(n.baseState=e,[e,n.queue.dispatch]):bs(n,G,e,t)},useMemoCache:cs,useCacheRefresh:rc,useEffectEvent:Bs};function gc(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:E({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var _c={enqueueSetState:function(e,t,n){e=e._reactInternals;var r=kd(),i=ho(r);i.payload=t,n!=null&&(i.callback=n),t=go(e,i,r),t!==null&&(Md(t,e,r),_o(t,e,r))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=kd(),i=ho(r);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=go(e,i,r),t!==null&&(Md(t,e,r),_o(t,e,r))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=kd(),r=ho(n);r.tag=2,t!=null&&(r.callback=t),t=go(e,r,n),t!==null&&(Md(t,e,n),_o(t,e,n))}};function vc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate==`function`?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Vr(n,r)||!Vr(i,a):!0}function yc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps==`function`&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps==`function`&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&_c.enqueueReplaceState(t,t.state,null)}function bc(e,t){var n=t;if(`ref`in t)for(var r in n={},t)r!==`ref`&&(n[r]=t[r]);if(e=e.defaultProps)for(var i in n===t&&(n=E({},n)),e)n[i]===void 0&&(n[i]=e[i]);return n}function xc(e){yi(e)}function Sc(e){console.error(e)}function Cc(e){yi(e)}function wc(e,t){try{var n=e.onUncaughtError;n(t.value,{componentStack:t.stack})}catch(e){setTimeout(function(){throw e})}}function Tc(e,t,n){try{var r=e.onCaughtError;r(n.value,{componentStack:n.stack,errorBoundary:t.tag===1?t.stateNode:null})}catch(e){setTimeout(function(){throw e})}}function Ec(e,t,n){return n=ho(n),n.tag=3,n.payload={element:null},n.callback=function(){wc(e,t)},n}function Dc(e){return e=ho(e),e.tag=3,e}function Oc(e,t,n,r){var i=n.type.getDerivedStateFromError;if(typeof i==`function`){var a=r.value;e.payload=function(){return i(a)},e.callback=function(){Tc(t,n,r)}}var o=n.stateNode;o!==null&&typeof o.componentDidCatch==`function`&&(e.callback=function(){Tc(t,n,r),typeof i!=`function`&&(gd===null?gd=new Set([this]):gd.add(this));var e=r.stack;this.componentDidCatch(r.value,{componentStack:e===null?``:e})})}function kc(e,t,n,r,i){if(n.flags|=32768,typeof r==`object`&&r&&typeof r.then==`function`){if(t=n.alternate,t!==null&&xa(t,n,i,!0),n=ko.current,n!==null){switch(n.tag){case 31:case 13:case 19:return Ao===null?Wd():n.alternate===null&&ad===0&&(ad=3),n.flags&=-257,n.flags|=65536,n.lanes=i,r===Xa?n.flags|=16384:(t=n.updateQueue,t===null?n.updateQueue=new Set([r]):t.add(r),pf(e,r,i)),!1;case 22:return n.flags|=65536,r===Xa?n.flags|=16384:(t=n.updateQueue,t===null?(t={transitions:null,markerInstances:null,retryQueue:new Set([r])},n.updateQueue=t):(n=t.retryQueue,n===null?t.retryQueue=new Set([r]):n.add(r)),pf(e,r,i)),!1}throw Error(o(435,n.tag))}return pf(e,r,i),Wd(),!1}if(V)return t=ko.current,t===null?(r!==oa&&(t=Error(o(423),{cause:r}),pa(Vi(t,n))),e=e.current.alternate,e.flags|=65536,i&=-i,e.lanes|=i,r=Vi(r,n),i=Ec(e.stateNode,r,i),vo(e,i),ad!==4&&(ad=2)):(!(t.flags&65536)&&(t.flags|=256),t.flags|=65536,t.lanes=i,r!==oa&&(e=Error(o(422),{cause:r}),pa(Vi(e,n)))),!1;var a=Error(o(520),{cause:r});if(a=Vi(a,n),ud===null?ud=[a]:ud.push(a),ad!==4&&(ad=2),t===null)return!0;r=Vi(r,n),n=t;do{switch(n.tag){case 3:return n.flags|=65536,e=i&-i,n.lanes|=e,e=Ec(n.stateNode,r,e),vo(n,e),!1;case 1:if(t=n.type,a=n.stateNode,!(n.flags&128)&&(typeof t.getDerivedStateFromError==`function`||a!==null&&typeof a.componentDidCatch==`function`&&(gd===null||!gd.has(a))))return n.flags|=65536,i&=-i,n.lanes|=i,i=Dc(i),Oc(i,e,n,r),vo(n,i),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var Ac=Error(o(461)),jc=!1;function Mc(e,t,n,r){t.child=e===null?uo(t,null,n,r):lo(t,e.child,n,r)}function Nc(e,t,n,r,i){n=n.render;var a=t.ref;if(`ref`in r){var o={};for(var s in r)s!==`ref`&&(o[s]=r[s])}else o=r;return Ca(t),r=Xo(e,t,n,o,a,i),s=es(),e!==null&&!jc?(ts(e,t,i),sl(e,t,i)):(V&&s&&$i(t),t.flags|=1,Mc(e,t,r,i),t.child)}function Pc(e,t,n,r,i){if(e===null){var a=n.type;return typeof a==`function`&&!Mi(a)&&a.defaultProps===void 0&&n.compare===null?(t.tag=15,t.type=a,Fc(e,t,a,r,i)):(e=Fi(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!cl(e,i)){var o=a.memoizedProps;if(n=n.compare,n=n===null?Vr:n,n(o,r)&&e.ref===t.ref)return sl(e,t,i)}return t.flags|=1,e=Ni(a,r),e.ref=t.ref,e.return=t,t.child=e}function Fc(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Vr(a,r)&&e.ref===t.ref){if(jc=!1,t.pendingProps=r=a,cl(e,i))e.flags&131072&&(jc=!0);else return t.lanes=e.lanes,sl(e,t,i)}}return Uc(e,t,n,r,i)}function Ic(e,t,n,r){var i=r.children,a=e===null?null:e.memoizedState;if(e===null&&t.stateNode===null&&(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),r.mode===`hidden`){if(t.flags&128){if(a=a===null?n:a.baseLanes|n,e!==null){for(r=t.child=e.child,i=0;r!==null;)i=i|r.lanes|r.childLanes,r=r.sibling;r=i&~a}else r=0,t.child=null;return Rc(e,t,a,n,r)}if(n&536870912)t.memoizedState={baseLanes:0,cachePool:null},e!==null&&Ka(t,a===null?null:a.cachePool),a===null?Do():Eo(t,a),No(t);else return r=t.lanes=536870912,Rc(e,t,a===null?n:a.baseLanes|n,n,r)}else a===null?(e!==null&&Ka(t,null),Do(),U()):(Ka(t,a.cachePool),Eo(t,a),U(),t.memoizedState=null);return Mc(e,t,i,n),t.child}function Lc(e,t){return e!==null&&e.tag===22||t.stateNode!==null||(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),t.sibling}function Rc(e,t,n,r,i){var a=Ga();return a=a===null?null:{parent:Aa._currentValue,pool:a},t.memoizedState={baseLanes:n,cachePool:a},e!==null&&Ka(t,null),Do(),No(t),e!==null&&xa(e,t,r,!0),t.childLanes=i,null}function zc(e,t){return t=Qc({mode:t.mode,children:t.children},e.mode),t.ref=e.ref,e.child=t,t.return=e,t}function Bc(e,t,n){return lo(t,e.child,null,n),e=zc(t,t.pendingProps),e.flags|=2,Po(t),t.memoizedState=null,e}function Vc(e,t,n){var r=t.pendingProps,i=!!(t.flags&128);if(t.flags&=-129,e===null){if(V){if(r.mode===`hidden`)return e=zc(t,r),t.lanes=536870912,e.memoizedState={baseLanes:0,cachePool:null},Lc(null,e);if(Mo(t),(e=ra)?(e=am(e,aa),e=e!==null&&e.data===`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Ji===null?null:{id:Yi,overflow:Xi},retryLane:536870912,hydrationErrors:null},n=Ri(e),n.return=t,t.child=n,na=t,ra=null)):e=null,e===null)throw sa(t);return t.lanes=536870912,null}return zc(t,r)}var a=e.memoizedState;if(a!==null){var s=a.dehydrated;if(Mo(t),i){if(t.flags&256)t.flags&=-257,t=Bc(e,t,n);else if(t.memoizedState!==null)t.child=e.child,t.flags|=128,t=null;else throw Error(o(558))}else if(jc||xa(e,t,n,!1),i=(n&e.childLanes)!==0,jc||i){if(wo.current===null){if(r=$u,r!==null&&(s=ht(r,n),s!==0&&s!==a.retryLane))throw a.retryLane=s,Ei(e,s),Md(r,e,s),Ac;Wd()}t=Bc(e,t,n)}else e=a.treeContext,ra=lm(s.nextSibling),na=t,V=!0,ia=null,aa=!1,e!==null&&ta(t,e),t=zc(t,r),t.flags|=134221824;return t}return e=Ni(e.child,{mode:r.mode,children:r.children}),e.ref=t.ref,t.child=e,e.return=t,e}function Hc(e,t){var n=t.ref;if(n===null)e!==null&&e.ref!==null&&(t.flags|=4194816);else{if(typeof n!=`function`&&typeof n!=`object`)throw Error(o(284));(e===null||e.ref!==n)&&(t.flags|=4194816)}}function Uc(e,t,n,r,i){return Ca(t),n=Xo(e,t,n,r,void 0,i),r=es(),e!==null&&!jc?(ts(e,t,i),sl(e,t,i)):(V&&r&&$i(t),t.flags|=1,Mc(e,t,n,i),t.child)}function Wc(e,t,n,r,i,a){return Ca(t),t.updateQueue=null,n=Qo(t,r,n,i),Zo(e),r=es(),e!==null&&!jc?(ts(e,t,a),sl(e,t,a)):(V&&r&&$i(t),t.flags|=1,Mc(e,t,n,a),t.child)}function Gc(e,t,n,r,i){if(Ca(t),t.stateNode===null){var a=ki,o=n.contextType;typeof o==`object`&&o&&(a=wa(o)),a=new n(r,a),t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,a.updater=_c,t.stateNode=a,a._reactInternals=t,a=t.stateNode,a.props=r,a.state=t.memoizedState,a.refs={},po(t),o=n.contextType,a.context=typeof o==`object`&&o?wa(o):ki,a.state=t.memoizedState,o=n.getDerivedStateFromProps,typeof o==`function`&&(gc(t,n,o,r),a.state=t.memoizedState),typeof n.getDerivedStateFromProps==`function`||typeof a.getSnapshotBeforeUpdate==`function`||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(o=a.state,typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount(),o!==a.state&&_c.enqueueReplaceState(a,a.state,null),xo(t,r,a,i),bo(),a.state=t.memoizedState),typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!0}else if(e===null){a=t.stateNode;var s=t.memoizedProps,c=bc(n,s);a.props=c;var l=a.context,u=n.contextType;o=ki,typeof u==`object`&&u&&(o=wa(u));var d=n.getDerivedStateFromProps;u=typeof d==`function`||typeof a.getSnapshotBeforeUpdate==`function`,s=t.pendingProps!==s,u||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(s||l!==o)&&yc(t,a,r,o),fo=!1;var f=t.memoizedState;a.state=f,xo(t,r,a,i),bo(),l=t.memoizedState,s||f!==l||fo?(typeof d==`function`&&(gc(t,n,d,r),l=t.memoizedState),(c=fo||vc(t,n,c,r,f,l,o))?(u||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount==`function`&&(t.flags|=4194308)):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=o,r=c):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,mo(e,t),o=t.memoizedProps,u=bc(n,o),a.props=u,d=t.pendingProps,f=a.context,l=n.contextType,c=ki,typeof l==`object`&&l&&(c=wa(l)),s=n.getDerivedStateFromProps,(l=typeof s==`function`||typeof a.getSnapshotBeforeUpdate==`function`)||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(o!==d||f!==c)&&yc(t,a,r,c),fo=!1,f=t.memoizedState,a.state=f,xo(t,r,a,i),bo();var p=t.memoizedState;o!==d||f!==p||fo||e!==null&&e.dependencies!==null&&Sa(e.dependencies)?(typeof s==`function`&&(gc(t,n,s,r),p=t.memoizedState),(u=fo||vc(t,n,u,r,f,p,c)||e!==null&&e.dependencies!==null&&Sa(e.dependencies))?(l||typeof a.UNSAFE_componentWillUpdate!=`function`&&typeof a.componentWillUpdate!=`function`||(typeof a.componentWillUpdate==`function`&&a.componentWillUpdate(r,p,c),typeof a.UNSAFE_componentWillUpdate==`function`&&a.UNSAFE_componentWillUpdate(r,p,c)),typeof a.componentDidUpdate==`function`&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate==`function`&&(t.flags|=1024)):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=p),a.props=r,a.state=p,a.context=c,r=u):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return a=r,Hc(e,t),r=!!(t.flags&128),a||r?(a=t.stateNode,n=r&&typeof n.getDerivedStateFromError!=`function`?null:a.render(),t.flags|=1,e!==null&&r?(t.child=lo(t,e.child,null,i),t.child=lo(t,null,n,i)):Mc(e,t,n,i),t.memoizedState=a.state,e=t.child):e=sl(e,t,i),e}function Kc(e,t,n,r){return da(),t.flags|=256,Mc(e,t,n,r),t.child}var qc={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function Jc(e){return{baseLanes:e,cachePool:qa()}}function Yc(e,t,n){return e=e===null?0:e.childLanes&~n,t&&(e|=cd),e}function Xc(e,t,n){var r=t.pendingProps,i=!1,a=!!(t.flags&128),o;if((o=a)||(o=e!==null&&e.memoizedState===null?!1:!!(Fo.current&2)),o&&(i=!0,t.flags&=-129),o=!!(t.flags&32),t.flags&=-33,e===null){if(V){if(i?jo(t):U(),(e=ra)?(e=am(e,aa),e=e!==null&&e.data!==`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Ji===null?null:{id:Yi,overflow:Xi},retryLane:536870912,hydrationErrors:null},n=Ri(e),n.return=t,t.child=n,na=t,ra=null)):e=null,e===null)throw sa(t);return t.lanes=sm(e)?32:536870912,null}return a=r.children,r=r.fallback,i?(U(),i=t.mode,a=Qc({mode:`hidden`,children:a},i),r=Ii(r,i,n,null),a.return=t,r.return=t,a.sibling=r,t.child=a,r=t.child,r.memoizedState=Jc(n),r.childLanes=Yc(e,o,n),t.memoizedState=qc,Lc(null,r)):(jo(t),Zc(t,a))}var s=e.memoizedState;if(s!==null){var c=s.dehydrated;if(c!==null)return el(e,t,a,o,r,c,s,n)}return i?(U(),i=r.fallback,a=t.mode,s=e.child,c=s.sibling,r=Ni(s,{mode:`hidden`,children:r.children}),r.subtreeFlags=s.subtreeFlags&1206910976,c===null?(i=Ii(i,a,n,null),i.flags|=2):i=Ni(c,i),i.return=t,r.return=t,r.sibling=i,t.child=r,Lc(null,r),r=t.child,i=e.child.memoizedState,i===null?i=Jc(n):(a=i.cachePool,a===null?a=qa():(s=Aa._currentValue,a=a.parent===s?a:{parent:s,pool:s}),i={baseLanes:i.baseLanes|n,cachePool:a}),r.memoizedState=i,r.childLanes=Yc(e,o,n),t.memoizedState=qc,Lc(e.child,r)):(jo(t),n=e.child,e=n.sibling,n=Ni(n,{mode:`visible`,children:r.children}),n.return=t,n.sibling=null,e!==null&&(o=t.deletions,o===null?(t.deletions=[e],t.flags|=16):o.push(e)),t.child=n,t.memoizedState=null,n)}function Zc(e,t){return t=Qc({mode:`visible`,children:t},e.mode),t.return=e,e.child=t}function Qc(e,t){return e=ji(22,e,null,t),e.lanes=0,e}function $c(e,t,n){return lo(t,e.child,null,n),e=Zc(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function el(e,t,n,r,i,a,s,c){if(n)return t.flags&256?(jo(t),t.flags&=-257,$c(e,t,c)):t.memoizedState===null?(U(),a=i.fallback,s=t.mode,i=Qc({mode:`visible`,children:i.children},s),a=Ii(a,s,c,null),a.flags|=2,i.return=t,a.return=t,i.sibling=a,t.child=i,lo(t,e.child,null,c),i=t.child,i.memoizedState=Jc(c),i.childLanes=Yc(e,r,c),t.memoizedState=qc,Lc(null,i)):(U(),t.child=e.child,t.flags|=128,null);if(jo(t),sm(a)){if(r=a.nextSibling&&a.nextSibling.dataset,r)var l=r.dgst;return r=l,r!==``&&(i=Error(o(419)),i.stack=``,i.digest=r,pa({value:i,source:null,stack:null})),$c(e,t,c)}if(jc||xa(e,t,c,!1),r=(c&e.childLanes)!==0,jc||r){if(wo.current!==null)return $c(e,t,c);if(r=$u,r!==null&&(i=ht(r,c),i!==0&&i!==s.retryLane))throw s.retryLane=i,Ei(e,i),Md(r,e,i),Ac;return om(a)||Wd(),$c(e,t,c)}return om(a)?(t.flags|=192,t.child=e.child,null):(e=s.treeContext,ra=lm(a.nextSibling),na=t,V=!0,ia=null,aa=!1,e!==null&&ta(t,e),t=Zc(t,i.children),t.flags|=134221824,t)}function tl(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),ya(e.return,t,n)}function nl(e){for(var t=null;e!==null;){var n=e.alternate;n!==null&&Ro(n)===null&&(t=e),e=e.sibling}return t}function rl(e,t,n,r,i,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i,treeForkCount:a}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i,o.treeForkCount=a)}function il(e){var t=e.child;for(e.child=null;t!==null;){var n=t.sibling;t.sibling=e.child,e.child=t,t=n}}function al(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;r=r.children;var o=Fo.current;if(t.flags&128)return Io(t,o),null;var s=!!(o&2);if(s?(o=o&1|2,t.flags|=128):o&=1,Io(t,o),i===`backwards`&&e!==null?(il(e),Mc(e,t,r,n),il(e)):Mc(e,t,r,n),r=V?Gi:0,!s&&e!==null&&e.flags&128)a:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&tl(e,n,t);else if(e.tag===19)tl(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break a;for(;e.sibling===null;){if(e.return===null||e.return===t)break a;e=e.return}e.sibling.return=e.return,e=e.sibling}switch(i){case`backwards`:n=nl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null,il(t)),rl(t,!0,i,null,a,r);break;case`unstable_legacy-backwards`:for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Ro(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}rl(t,!0,n,null,a,r);break;case`together`:rl(t,!1,null,null,void 0,r);break;case`independent`:t.memoizedState=null;break;default:n=nl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),rl(t,!1,i,n,a,r)}return t.child}function ol(e,t,n){var r=t.pendingProps;return _a(t,t.type,r.value),Mc(e,t,r.children,n),t.child}function sl(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),Z|=t.lanes,(n&t.childLanes)===0){if(e!==null){if(xa(e,t,n,!1),(n&t.childLanes)===0)return null}else return null}if(e!==null&&t.child!==e.child)throw Error(o(153));if(t.child!==null){for(e=t.child,n=Ni(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=Ni(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function cl(e,t){return(e.lanes&t)!==0||(e=e.dependencies,!!(e!==null&&Sa(e)))}function ll(e,t,n){switch(t.tag){case 3:Te(t,t.stateNode.containerInfo),_a(t,Aa,e.memoizedState.cache),da();break;case 27:case 5:De(t);break;case 4:Te(t,t.stateNode.containerInfo);break;case 10:_a(t,t.type,t.memoizedProps.value);break;case 31:if(t.memoizedState!==null)return t.flags|=128,Mo(t),null;break;case 13:var r=t.memoizedState;if(r!==null){if(r.dehydrated!==null)return jo(t),t.flags|=128,null;r=xa(e,t,n,!1);var i=t.child.childLanes;return r||(n&i)!==0?Xc(e,t,n):(jo(t),e=sl(e,t,n),e===null?null:e.sibling)}jo(t);break;case 19:if(t.flags&128)return al(e,t,n);if(i=!!(e.flags&128),r=(n&t.childLanes)!==0,r||=(xa(e,t,n,!1),(n&t.childLanes)!==0),i){if(r)return al(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),Io(t,Fo.current),r)break;return null;case 22:return t.lanes=0,Ic(e,t,n,t.pendingProps);case 24:_a(t,Aa,e.memoizedState.cache)}return sl(e,t,n)}function ul(e,t,n){if(e!==null){if(e.memoizedProps!==t.pendingProps)jc=!0;else{if(!cl(e,n)&&!(t.flags&128))return jc=!1,ll(e,t,n);jc=!!(e.flags&131072)}}else jc=!1,V&&t.flags&1048576&&Qi(t,Gi,t.index);switch(t.lanes=0,t.tag){case 16:a:{var r=t.pendingProps;if(e=$a(t.elementType),t.type=e,typeof e==`function`)Mi(e)?(r=bc(e,r),t.tag=1,t=Gc(null,t,e,r,n)):(t.tag=0,t=Uc(null,t,e,r,n));else{if(e!=null){var i=e.$$typeof;if(i===ae){t.tag=11,t=Nc(null,t,e,r,n);break a}if(i===ce){t.tag=14,t=Pc(null,t,e,r,n);break a}if(i===k){t.tag=10,t.type=e,t=ol(null,t,n);break a}}throw t=he(e)||e,Error(o(306,t,``))}}return t;case 0:return Uc(e,t,t.type,t.pendingProps,n);case 1:return r=t.type,i=bc(r,t.pendingProps),Gc(e,t,r,i,n);case 3:a:{if(Te(t,t.stateNode.containerInfo),e===null)throw Error(o(387));r=t.pendingProps;var a=t.memoizedState;i=a.element,mo(e,t),xo(t,r,null,n);var s=t.memoizedState;if(r=s.cache,_a(t,Aa,r),r!==a.cache&&ba(t,[Aa],n,!0),bo(),r=s.element,a.isDehydrated){if(a={element:r,isDehydrated:!1,cache:s.cache},t.updateQueue.baseState=a,t.memoizedState=a,t.flags&256){t=Kc(e,t,r,n);break a}if(r!==i){i=Vi(Error(o(424)),t),pa(i),t=Kc(e,t,r,n);break a}switch(e=t.stateNode.containerInfo,e.nodeType){case 9:e=e.body;break;default:e=e.nodeName===`HTML`?e.ownerDocument.body:e}for(ra=lm(e.firstChild),na=t,V=!0,ia=null,aa=!0,n=uo(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling}else{if(da(),r===i){t=sl(e,t,n);break a}Mc(e,t,r,n)}t=t.child}return t;case 26:return Hc(e,t),e===null?(n=Nm(t.type,null,t.pendingProps,null))?t.memoizedState=n:V||(t.stateNode=fp(t.type,t.pendingProps,Ce.current,t)):t.memoizedState=Nm(t.type,e.memoizedProps,t.pendingProps,e.memoizedState),null;case 27:return De(t),e===null&&V&&(r=t.stateNode=hm(t.type,t.pendingProps,Ce.current),na=t,aa=!0,i=ra,Sp(t.type)?(um=i,ra=lm(r.firstChild)):ra=i),Mc(e,t,t.pendingProps.children,n),Hc(e,t),e===null&&(t.flags|=4194304),t.child;case 5:return e===null&&V&&((i=r=ra)&&(r=rm(r,t.type,t.pendingProps,aa),r===null?i=!1:(t.stateNode=r,na=t,ra=lm(r.firstChild),aa=!1,i=!0)),i||sa(t)),De(t),i=t.type,a=t.pendingProps,s=e===null?null:e.memoizedProps,r=a.children,pp(i,a)?r=null:s!==null&&pp(i,s)&&(t.flags|=32),t.memoizedState!==null&&(i=Xo(e,t,$o,null,null,n),sh._currentValue=i),Hc(e,t),Mc(e,t,r,n),t.child;case 6:return e===null&&V&&((e=n=ra)&&(n=im(n,t.pendingProps,aa),n===null?e=!1:(t.stateNode=n,na=t,ra=null,e=!0)),e||sa(t)),null;case 13:return Xc(e,t,n);case 4:return Te(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=lo(t,null,r,n):Mc(e,t,r,n),t.child;case 11:return Nc(e,t,t.type,t.pendingProps,n);case 7:return r=t.pendingProps,Hc(e,t),Mc(e,t,r,n),t.child;case 8:return Mc(e,t,t.pendingProps.children,n),t.child;case 12:return Mc(e,t,t.pendingProps.children,n),t.child;case 10:return ol(e,t,n);case 9:return i=t.type._context,r=t.pendingProps.children,Ca(t),i=wa(i),r=r(i),t.flags|=1,Mc(e,t,r,n),t.child;case 14:return Pc(e,t,t.type,t.pendingProps,n);case 15:return Fc(e,t,t.type,t.pendingProps,n);case 19:return al(e,t,n);case 31:return Vc(e,t,n);case 22:return Ic(e,t,n,t.pendingProps);case 24:return Ca(t),r=wa(Aa),e===null?(i=Ga(),i===null&&(i=$u,a=ja(),i.pooledCache=a,a.refCount++,a!==null&&(i.pooledCacheLanes|=n),i=a),t.memoizedState={parent:r,cache:i},po(t),_a(t,Aa,i)):((e.lanes&n)!==0&&(mo(e,t),xo(t,null,null,n),bo()),i=e.memoizedState,a=t.memoizedState,i.parent===r?(r=a.cache,_a(t,Aa,r),r!==i.cache&&ba(t,[Aa],n,!0)):(i={parent:r,cache:r},t.memoizedState=i,t.lanes===0&&(t.memoizedState=t.updateQueue.baseState=i),_a(t,Aa,r))),Mc(e,t,t.pendingProps.children,n),t.child;case 30:return t.stateNode===null&&(t.stateNode={autoName:null,paired:null,clones:null,ref:null}),r=t.pendingProps,r.name!=null&&r.name!==`auto`?t.flags|=e===null?18882560:18874368:V&&$i(t),e!==null&&e.memoizedProps.name!==r.name?t.flags|=4194816:Hc(e,t),Mc(e,t,r.children,n),t.child;case 29:throw t.pendingProps}throw Error(o(156,t.tag))}function dl(e){e.flags|=4}function fl(e,t,n,r,i){var a;if((a=!!(e.mode&32))&&(a=n===null?Jm(t,r):Jm(t,r)&&(r.src!==n.src||r.srcSet!==n.srcSet)),a){if(e.flags|=16777216,(i&335544128)===i){if(e.stateNode.complete)e.flags|=8192;else if(Vd())e.flags|=8192;else throw eo=Xa,H}}else e.flags&=-16777217}function pl(e,t){if(t.type!==`stylesheet`||t.state.loading&4)e.flags&=-16777217;else if(e.flags|=16777216,!Ym(t)){if(Vd())e.flags|=8192;else throw eo=Xa,H}}function ml(e,t){t!==null&&(e.flags|=4),e.flags&16384&&(t=e.tag===22?536870912:lt(),e.lanes|=t,ld|=t)}function hl(e,t){if(!V)switch(e.tailMode){case`visible`:break;case`collapsed`:for(var n=e.tail,r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null;break;default:for(t=e.tail,n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null}}function gl(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&1206910976,r|=i.flags&1206910976,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function _l(e,t,n){var r=t.pendingProps;switch(ea(t),t.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return gl(t),null;case 1:return gl(t),null;case 3:return n=t.stateNode,r=null,e!==null&&(r=e.memoizedState.cache),t.memoizedState.cache!==r&&(t.flags|=2048),va(Aa),Ee(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(e===null||e.child===null)&&(ua(t)?dl(t):e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,fa())),gl(t),null;case 26:var i=t.type,a=t.memoizedState;return e===null?(dl(t),a===null?(gl(t),fl(t,i,null,r,n)):(gl(t),pl(t,a))):a?a===e.memoizedState?(gl(t),t.flags&=-16777217):(dl(t),gl(t),pl(t,a)):(e=e.memoizedProps,e!==r&&dl(t),gl(t),fl(t,i,e,r,n)),null;case 27:if(Oe(t),n=Ce.current,i=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&dl(t);else{if(!r){if(t.stateNode===null)throw Error(o(166));return gl(t),t.subtreeFlags&=-33554433,null}e=xe.current,ua(t)?ca(t,e):(e=hm(i,r,n),t.stateNode=e,dl(t))}return gl(t),t.subtreeFlags&=-33554433,null;case 5:if(Oe(t),i=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&dl(t);else{if(!r){if(t.stateNode===null)throw Error(o(166));return gl(t),t.subtreeFlags&=-33554433,null}if(a=xe.current,ua(t))ca(t,a);else{var s=lp(Ce.current);switch(a){case 1:a=s.createElementNS(`http://www.w3.org/2000/svg`,i);break;case 2:a=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,i);break;default:switch(i){case`svg`:a=s.createElementNS(`http://www.w3.org/2000/svg`,i);break;case`math`:a=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,i);break;case`script`:a=s.createElement(`div`),a.innerHTML=`<script><\/script>`,a=a.removeChild(a.firstChild);break;case`select`:a=typeof r.is==`string`?s.createElement(`select`,{is:r.is}):s.createElement(`select`),r.multiple?a.multiple=!0:r.size&&(a.size=r.size);break;default:a=typeof r.is==`string`?s.createElement(i,{is:r.is}):s.createElement(i)}}a[xt]=t,a[St]=r;a:for(s=t.child;s!==null;){if(s.tag===5||s.tag===6)a.appendChild(s.stateNode);else if(s.tag!==4&&s.tag!==27&&s.child!==null){s.child.return=s,s=s.child;continue}if(s===t)break a;for(;s.sibling===null;){if(s.return===null||s.return===t)break a;s=s.return}s.sibling.return=s.return,s=s.sibling}t.stateNode=a;a:switch(np(a,i,r),i){case`button`:case`input`:case`select`:case`textarea`:r=!!r.autoFocus;break a;case`img`:r=!0;break a;default:r=!1}r&&dl(t)}}return gl(t),t.subtreeFlags&=-33554433,fl(t,t.type,e===null?null:e.memoizedProps,t.pendingProps,n),null;case 6:if(e&&t.stateNode!=null)e.memoizedProps!==r&&dl(t);else{if(typeof r!=`string`&&t.stateNode===null)throw Error(o(166));if(e=Ce.current,ua(t)){if(e=t.stateNode,n=t.memoizedProps,r=null,i=na,i!==null)switch(i.tag){case 27:case 5:r=i.memoizedProps}e[xt]=t,e=!!(e.nodeValue===n||r!==null&&!0===r.suppressHydrationWarning||$f(e.nodeValue,n)),e||sa(t,!0)}else e=lp(e).createTextNode(r),e[xt]=t,t.stateNode=e}return gl(t),null;case 31:if(n=t.memoizedState,e===null||e.memoizedState!==null){if(r=ua(t),n!==null){if(e===null){if(!r)throw Error(o(318));if(e=t.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(o(557));e[xt]=t}else da(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;gl(t),e=!1}else n=fa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=n),e=!0;if(!e)return t.flags&256?(Po(t),t):(Po(t),null);if(t.flags&128)throw Error(o(558))}return gl(t),null;case 13:if(r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(i=ua(t),r!==null&&r.dehydrated!==null){if(e===null){if(!i)throw Error(o(318));if(i=t.memoizedState,i=i===null?null:i.dehydrated,!i)throw Error(o(317));i[xt]=t}else da(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;gl(t),i=!1}else i=fa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=i),i=!0;if(!i)return t.flags&256?(Po(t),t):(Po(t),null)}return Po(t),t.flags&128?(t.lanes=n,t):(n=r!==null,e=e!==null&&e.memoizedState!==null,n&&(r=t.child,i=null,r.alternate!==null&&r.alternate.memoizedState!==null&&r.alternate.memoizedState.cachePool!==null&&(i=r.alternate.memoizedState.cachePool.pool),a=null,r.memoizedState!==null&&r.memoizedState.cachePool!==null&&(a=r.memoizedState.cachePool.pool),a!==i&&(r.flags|=2048)),n!==e&&n&&(t.child.flags|=8192),ml(t,t.updateQueue),gl(t),null);case 4:return Ee(),e===null&&Uf(t.stateNode.containerInfo),t.flags|=67108864,gl(t),null;case 10:return va(t.type),gl(t),null;case 19:if(Lo(t),r=t.memoizedState,r===null)return gl(t),null;if(i=!!(t.flags&128),a=r.rendering,a===null){if(i)hl(r,!1);else{if(ad!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(a=Ro(e),a!==null){for(t.flags|=128,hl(r,!1),e=a.updateQueue,t.updateQueue=e,ml(t,e),t.subtreeFlags=0,e=n,n=t.child;n!==null;)Pi(n,e),n=n.sibling;return Io(t,Fo.current&1|2),V&&Zi(t,r.treeForkCount),t.child}e=e.sibling}r.tail!==null&&ze()>Q&&(t.flags|=128,i=!0,hl(r,!1),t.lanes=4194304)}}else{if(!i){if(e=Ro(a),e!==null){if(t.flags|=128,i=!0,e=e.updateQueue,t.updateQueue=e,ml(t,e),hl(r,!0),r.tail===null&&r.tailMode!==`collapsed`&&r.tailMode!==`visible`&&!a.alternate&&!V)return gl(t),null}else 2*ze()-r.renderingStartTime>Q&&n!==536870912&&(t.flags|=128,i=!0,hl(r,!1),t.lanes=4194304)}r.isBackwards?(a.sibling=t.child,t.child=a):(e=r.last,e===null?t.child=a:e.sibling=a,r.last=a)}if(r.tail!==null){e=r.tail;a:{for(n=e;n!==null;){if(n.alternate!==null){n=!1;break a}n=n.sibling}n=!0}return r.rendering=e,r.tail=e.sibling,r.renderingStartTime=ze(),e.sibling=null,a=Fo.current,a=i?a&1|2:a&1,r.tailMode===`visible`||r.tailMode===`collapsed`||!n||V?Io(t,a):(n=a,I(ko,t),I(Fo,n),Ao===null&&(Ao=t)),V&&Zi(t,r.treeForkCount),e}return gl(t),null;case 22:case 23:return Po(t),Oo(),r=t.memoizedState!==null,e===null?r&&(t.flags|=8192):e.memoizedState!==null!==r&&(t.flags|=8192),r?n&536870912&&!(t.flags&128)&&(gl(t),t.subtreeFlags&6&&(t.flags|=8192)):gl(t),n=t.updateQueue,n!==null&&ml(t,n.retryQueue),n=null,e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),r=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(r=t.memoizedState.cachePool.pool),r!==n&&(t.flags|=2048),e!==null&&F(Wa),null;case 24:return n=null,e!==null&&(n=e.memoizedState.cache),t.memoizedState.cache!==n&&(t.flags|=2048),va(Aa),gl(t),null;case 25:return null;case 30:return t.flags|=33554432,gl(t),null}throw Error(o(156,t.tag))}function vl(e,t){switch(ea(t),t.tag){case 1:return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return va(Aa),Ee(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 26:case 27:case 5:return Oe(t),null;case 31:if(t.memoizedState!==null){if(Po(t),t.alternate===null)throw Error(o(340));da()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 13:if(Po(t),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(o(340));da()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return Lo(t),e=t.flags,e&65536?(t.flags=e&-65537|128,e=t.memoizedState,e!==null&&(e.rendering=null,e.tail=null),t.flags|=4,t):null;case 4:return Ee(),null;case 10:return va(t.type),null;case 22:case 23:return Po(t),Oo(),e!==null&&F(Wa),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 24:return va(Aa),null;case 25:return null;default:return null}}function yl(e,t){switch(ea(t),t.tag){case 3:va(Aa),Ee();break;case 26:case 27:case 5:Oe(t);break;case 4:Ee();break;case 31:t.memoizedState!==null&&Po(t);break;case 13:Po(t);break;case 19:Lo(t);break;case 10:va(t.type);break;case 22:case 23:Po(t),Oo(),e!==null&&F(Wa);break;case 24:va(Aa)}}function bl(e,t){try{var n=t.updateQueue,r=n===null?null:n.lastEffect;if(r!==null){var i=r.next;n=i;do{if((n.tag&e)===e){r=void 0;var a=n.create,o=n.inst;r=a(),o.destroy=r}n=n.next}while(n!==i)}}catch(e){ff(t,t.return,e)}}function xl(e,t,n){try{var r=t.updateQueue,i=r===null?null:r.lastEffect;if(i!==null){var a=i.next;r=a;do{if((r.tag&e)===e){var o=r.inst,s=o.destroy;if(s!==void 0){o.destroy=void 0,i=t;var c=n,l=s;try{l()}catch(e){ff(i,c,e)}}}r=r.next}while(r!==a)}}catch(e){ff(t,t.return,e)}}function Sl(e){var t=e.updateQueue;if(t!==null){var n=e.stateNode;try{Co(t,n)}catch(t){ff(e,e.return,t)}}}function Cl(e,t,n){n.props=bc(e.type,e.memoizedProps),n.state=e.memoizedState;try{n.componentWillUnmount()}catch(n){ff(e,t,n)}}function wl(e,t){try{var n=e.ref;if(n!==null){switch(e.tag){case 26:case 27:case 5:var r=e.stateNode;break;case 30:var i=e.stateNode,a=gi(e.memoizedProps,i);(i.ref===null||i.ref.name!==a)&&(i.ref=Pp(a)),r=i.ref;break;case 7:if(e.stateNode===null){var o=new Fp(e);m(e.child,!1,Qp,o,void 0,void 0),e.stateNode=o}r=e.stateNode;break;default:r=e.stateNode}typeof n==`function`?e.refCleanup=n(r):n.current=r}}catch(n){ff(e,t,n)}}function Tl(e,t){var n=e.ref,r=e.refCleanup;if(n!==null){if(typeof r==`function`)try{r()}catch(n){ff(e,t,n)}finally{e.refCleanup=null,e=e.alternate,e!=null&&(e.refCleanup=null)}else if(typeof n==`function`)try{n(null)}catch(n){ff(e,t,n)}else n.current=null}}function El(e,t){if((e.tag===5||e.tag===27||e.tag===6)&&e.alternate===null&&t!==null)for(var n=0;n<t.length;n++)em(e.stateNode,t[n])}function Dl(e){for(var t=e.return;t!==null&&(Al(t)&&em(e.stateNode,t.stateNode),!kl(t));)t=t.return}function Ol(e){for(var t=e.return;t!==null&&(Al(t)&&tm(e.stateNode,t.stateNode),!kl(t));)t=t.return}function kl(e){return e.tag===5||e.tag===3||e.tag===27}function Al(e){return e&&e.tag===7&&e.stateNode!==null}function jl(e){var t=e.type,n=e.memoizedProps,r=e.stateNode;try{a:switch(t){case`button`:case`input`:case`select`:case`textarea`:n.autoFocus&&r.focus();break a;case`img`:n.src?r.src=n.src:n.srcSet&&(r.srcset=n.srcSet)}}catch(t){ff(e,e.return,t)}}function Ml(e,t,n){try{var r=e.stateNode;ip(r,e.type,n,t),r[St]=t}catch(t){ff(e,e.return,t)}}function Nl(e){return e.tag===5||e.tag===3||e.tag===26||e.tag===27&&Sp(e.type)||e.tag===4}function Pl(e){a:for(;;){for(;e.sibling===null;){if(e.return===null||Nl(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.tag===27&&Sp(e.type)||e.flags&2||e.child===null||e.tag===4)continue a;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Fl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?(n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n).insertBefore(i,t):(t=n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n,t.appendChild(i),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=_n)),El(e,r),B=!0;else if(i!==4&&(i===27&&(El(e,r),r=null,Sp(e.type)&&(n=e.stateNode,t=null)),e=e.child,e!==null))for(Fl(e,t,n,r),e=e.sibling;e!==null;)Fl(e,t,n,r),e=e.sibling}function Il(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?n.insertBefore(i,t):n.appendChild(i),El(e,r),B=!0;else if(i!==4&&(i===27&&(El(e,r),r=null,Sp(e.type)&&(n=e.stateNode)),e=e.child,e!==null))for(Il(e,t,n,r),e=e.sibling;e!==null;)Il(e,t,n,r),e=e.sibling}function Ll(e){var t=e.stateNode,n=e.memoizedProps;try{for(var r=e.type,i=t.attributes;i.length;)t.removeAttributeNode(i[0]);np(t,r,n),t[xt]=e,t[St]=n}catch(t){ff(e,e.return,t)}}var Rl=!1,zl=null;function Bl(e){(e.tag===30||e.subtreeFlags&33554432)&&(Rl=!0)}var Vl=null;function Hl(){var e=Vl;return Vl=null,e}var Ul=0;function Wl(e,t,n,r,i){return Ul=0,Gl(e.child,t,n,r,i)}function Gl(e,t,n,r,i){for(var a=!1;e!==null;){if(e.tag===5){var o=e.stateNode;if(r!==null){var s=Op(o);r.push(s),s.view&&(a=!0)}else a||Op(o).view&&(a=!0);Rl=!0,Tp(o,Ul===0?t:t+`_`+Ul,n),Ul++}else e.tag===22&&e.memoizedState!==null||e.tag===30&&i||Gl(e.child,t,n,r,i)&&(a=!0);e=e.sibling}return a}function Kl(e,t){for(;e!==null;)e.tag===5?Ep(e.stateNode,e.memoizedProps):e.tag===22&&e.memoizedState!==null||e.tag===30&&t||Kl(e.child,t),e=e.sibling}function ql(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if((e.tag!==22||e.memoizedState===null)&&(ql(e),e.tag===30&&e.flags&18874368&&e.stateNode.paired)){var t=e.memoizedProps;if(t.name==null||t.name===`auto`)throw Error(o(544));var n=t.name;t=vi(t.default,t.share),t!==`none`&&(Wl(e,n,t,null,!1)||Kl(e.child,!1))}e=e.sibling}}function Jl(e,t){if(e.tag===30){var n=e.stateNode,r=e.memoizedProps,i=gi(r,n),a=vi(r.default,n.paired?r.share:r.enter);a===`none`?ql(e):Wl(e,i,a,null,!1)?(ql(e),n.paired||t||jd(e,r.onEnter)):Kl(e.child,!1)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Jl(e,t),e=e.sibling;else ql(e)}function Yl(e){if(zl!==null&&zl.size!==0){var t=zl;if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var n=e.memoizedProps,r=n.name;if(r!=null&&r!==`auto`){var i=t.get(r);if(i!==void 0){var a=vi(n.default,n.share);if(a!==`none`&&(Wl(e,r,a,null,!1)?(a=e.stateNode,i.paired=a,a.paired=i,jd(e,n.onShare)):Kl(e.child,!1)),t.delete(r),t.size===0)break}}}Yl(e)}e=e.sibling}}}function Xl(e){if(e.tag===30){var t=e.memoizedProps,n=gi(t,e.stateNode),r=zl===null?void 0:zl.get(n),i=vi(t.default,r===void 0?t.exit:t.share);i!==`none`&&(Wl(e,n,i,null,!1)?r===void 0?jd(e,t.onExit):(i=e.stateNode,r.paired=i,i.paired=r,zl.delete(n),jd(e,t.onShare)):Kl(e.child,!1)),zl!==null&&Yl(e)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Xl(e),e=e.sibling;else zl!==null&&Yl(e)}function Zl(e){for(e=e.child;e!==null;){if(e.tag===30){var t=e.memoizedProps,n=gi(t,e.stateNode);t=vi(t.default,t.update),e.flags&=-5,t!==`none`&&Wl(e,n,t,e.memoizedState=[],!1)}else e.subtreeFlags&33554432&&Zl(e);e=e.sibling}}function Ql(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var t=e.stateNode;t.paired!==null&&(t.paired=null,Kl(e.child,!1))}Ql(e)}e=e.sibling}}function $l(e){if(e.tag===30)e.stateNode.paired=null,Kl(e.child,!1),Ql(e);else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)$l(e),e=e.sibling;else Ql(e)}function eu(e){for(e=e.child;e!==null;)e.tag===30?Kl(e.child,!1):e.subtreeFlags&33554432&&eu(e),e=e.sibling}function tu(e,t,n,r,i,a,o){for(var s=!1;t!==null;){if(t.tag===5){var c=t.stateNode;if(a!==null&&Ul<a.length){var l=a[Ul],u=Op(c);(l.view||u.view)&&(s=!0);var d;if(d=!(e.flags&4)){if(u.clip)d=!0;else{d=l.rect;var f=u.rect;d=d.y!==f.y||d.x!==f.x||d.height!==f.height||d.width!==f.width}}d&&(e.flags|=4),u.abs?u=!l.abs:(l=l.rect,u=u.rect,u=l.height!==u.height||l.width!==u.width),u&&(e.flags|=32)}else e.flags|=32;e.flags&4&&Tp(c,Ul===0?n:n+`_`+Ul,i),s&&e.flags&4||(Vl===null&&(Vl=[]),Vl.push(c,Ul===0?r:r+`_`+Ul,t.memoizedProps)),Ul++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&o?e.flags|=t.flags&32:tu(e,t.child,n,r,i,a,o)&&(s=!0));t=t.sibling}return s}function nu(e,t){for(e=e.child;e!==null;){if(e.tag===30){var n=e.memoizedProps,r=e.stateNode,i=gi(n,r),a=vi(n.default,n.update);if(t){r=r.clones;var o=r===null?null:r.map(kp)}else o=e.memoizedState,e.memoizedState=null;r=e;var s=e.child;Ul=0,i=tu(r,s,i,i,a,o,!1),e.flags&4&&i&&(t||jd(e,n.onUpdate))}else e.subtreeFlags&33554432&&nu(e,t);e=e.sibling}}var ru=!1,K=!1,iu=!1,au=!1,ou=typeof WeakSet==`function`?WeakSet:Set,su=null,cu=!1,lu=!1,uu=!1,du=!1;function fu(e,t,n){if(e=e.containerInfo,sp=gh,e=Kr(e),qr(e)){if(`selectionStart`in e)var r={start:e.selectionStart,end:e.selectionEnd};else a:{r=(r=e.ownerDocument)&&r.defaultView||window;var i=r.getSelection&&r.getSelection();if(i&&i.rangeCount!==0){r=i.anchorNode;var a=i.anchorOffset,o=i.focusNode;i=i.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break a}var s=0,c=-1,l=-1,u=0,d=0,f=e,p=null;b:for(;;){for(var m;f!==r||a!==0&&f.nodeType!==3||(c=s+a),f!==o||i!==0&&f.nodeType!==3||(l=s+i),f.nodeType===3&&(s+=f.nodeValue.length),(m=f.firstChild)!==null;)p=f,f=m;for(;;){if(f===e)break b;if(p===r&&++u===a&&(c=s),p===o&&++d===i&&(l=s),(m=f.nextSibling)!==null)break;f=p,p=f.parentNode}f=m}r=c===-1||l===-1?null:{start:c,end:l}}else r=null}r||={start:0,end:0}}else r=null;for(cp={focusedElem:e,selectionRange:r},gh=!1,n=(n&335544064)===n,su=t,t=n?9270:1024;su!==null;){if(e=su,n&&(r=e.deletions,r!==null))for(a=0;a<r.length;a++)n&&Xl(r[a]);if(e.alternate===null&&e.flags&2)n&&Bl(e),pu(n);else{if(e.tag===22){if(r=e.alternate,e.memoizedState!==null){r!==null&&r.memoizedState===null&&n&&Xl(r),pu(n);continue}if(r!==null&&r.memoizedState!==null){n&&Bl(e),pu(n);continue}}r=e.child,(e.subtreeFlags&t)!==0&&r!==null?(r.return=e,su=r):(n&&Zl(e),pu(n))}}zl=null}function pu(e){for(;su!==null;){var t=su,n=e,r=t.alternate,i=t.flags;switch(t.tag){case 0:case 11:case 15:break;case 1:if(i&1024&&r!==null){n=void 0,i=r.memoizedProps,r=r.memoizedState;var a=t.stateNode;try{var s=bc(t.type,i);n=a.getSnapshotBeforeUpdate(s,r),a.__reactInternalSnapshotBeforeUpdate=n}catch(e){ff(t,t.return,e)}}break;case 3:if(i&1024){if(r=t.stateNode.containerInfo,n=r.nodeType,n===9)nm(r);else if(n===1)switch(r.nodeName){case`HEAD`:case`HTML`:case`BODY`:nm(r);break;default:r.textContent=``}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&r!==null&&(n=gi(r.memoizedProps,r.stateNode),i=t.memoizedProps,i=vi(i.default,i.update),i!==`none`&&Wl(r,n,i,r.memoizedState=[],!0));break;default:if(i&1024)throw Error(o(163))}if(r=t.sibling,r!==null){r.return=t.return,su=r;break}su=t.return}}function mu(e,t,n){var r=n.flags;switch(n.tag){case 0:case 11:case 15:Pu(e,n),r&4&&bl(5,n);break;case 1:if(Pu(e,n),r&4){if(e=n.stateNode,t===null)try{e.componentDidMount()}catch(e){ff(n,n.return,e)}else{var i=bc(n.type,t.memoizedProps);t=t.memoizedState;try{e.componentDidUpdate(i,t,e.__reactInternalSnapshotBeforeUpdate)}catch(e){ff(n,n.return,e)}}}r&64&&Sl(n),r&512&&wl(n,n.return);break;case 3:if(Pu(e,n),r&64&&(e=n.updateQueue,e!==null)){if(t=null,n.child!==null)switch(n.child.tag){case 27:case 5:t=n.child.stateNode;break;case 1:t=n.child.stateNode}try{Co(e,t)}catch(e){ff(n,n.return,e)}}break;case 27:t===null&&r&4&&Ll(n);case 26:case 5:Pu(e,n),t===null&&r&4&&jl(n),r&512&&wl(n,n.return);break;case 12:Pu(e,n);break;case 31:Pu(e,n),r&4&&Cu(e,n);break;case 13:Pu(e,n),r&4&&wu(e,n),r&64&&(e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null&&(n=gf.bind(null,n),cm(e,n))));break;case 22:if(r=n.memoizedState!==null||ru,!r){var a=t!==null&&t.memoizedState!==null||K;t=ru,i=K,ru=r,(K=a)&&!i?(r=2,n.subtreeFlags&8772&&(r|=1),Iu(e,n,r)):Pu(e,n),ru=t,K=i}break;case 30:Pu(e,n),r&512&&wl(n,n.return);break;case 7:r&512&&wl(n,n.return);default:Pu(e,n)}}function hu(e,t){for(e=e.child;e!==null;)gu(e,t),e=e.sibling}function gu(e,t){switch(e.tag){case 5:case 26:try{var n=e.stateNode;if(t){var r=n.style;typeof r.setProperty==`function`?r.setProperty(`display`,`none`,`important`):r.display=`none`}else{var i=e.stateNode,a=e.memoizedProps.style,o=a!=null&&a.hasOwnProperty(`display`)?a.display:null;i.style.display=o==null||typeof o==`boolean`?``:(``+o).trim()}}catch(t){ff(e,e.return,t)}_u(e,t);break;case 6:try{e.stateNode.nodeValue=t?``:e.memoizedProps,B=!0}catch(t){ff(e,e.return,t)}break;case 18:try{var s=e.stateNode;t?wp(s,!0):wp(e.stateNode,!1)}catch(t){ff(e,e.return,t)}break;case 22:case 23:e.memoizedState===null&&hu(e,t);break;default:hu(e,t)}}function _u(e,t){if(e.subtreeFlags&67108864)for(e=e.child;e!==null;){a:{var n=e,r=t;switch(n.tag){case 4:gu(n,r);break a;case 22:n.memoizedState===null&&_u(n,r);break a;default:_u(n,r)}}e=e.sibling}}function vu(e){var t=e.alternate;t!==null&&(e.alternate=null,vu(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&At(t)),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}var yu=null,bu=!1;function xu(e,t,n){for(n=n.child;n!==null;)Su(e,t,n),n=n.sibling}function Su(e,t,n){if(Ye&&typeof Ye.onCommitFiberUnmount==`function`)try{Ye.onCommitFiberUnmount(Je,n)}catch{}switch(n.tag){case 26:K||Tl(n,t),xu(e,t,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!K&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:K||Tl(n,t),Ol(n);var r=yu,i=bu;Sp(n.type)&&(yu=n.stateNode,bu=!1),xu(e,t,n),gm(n.stateNode,n.type,n.memoizedProps),yu=r,bu=i;break;case 5:K||Tl(n,t),Ol(n);case 6:if(n.tag===6&&Ol(n),r=yu,i=bu,yu=null,xu(e,t,n),yu=r,bu=i,yu!==null){if(bu)try{(yu.nodeType===9?yu.body:yu.nodeName===`HTML`?yu.ownerDocument.body:yu).removeChild(n.stateNode),B=!0}catch(e){ff(n,t,e)}else try{yu.removeChild(n.stateNode),B=!0}catch(e){ff(n,t,e)}}break;case 18:yu!==null&&(bu?(e=yu,Cp(e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,n.stateNode),Hh(e)):Cp(yu,n.stateNode));break;case 4:r=yu,i=bu,yu=n.stateNode.containerInfo,bu=!0,xu(e,t,n),yu=r,bu=i;break;case 0:case 11:case 14:case 15:xl(2,n,t),K||xl(4,n,t),xu(e,t,n);break;case 1:K||(Tl(n,t),r=n.stateNode,typeof r.componentWillUnmount==`function`&&Cl(n,t,r)),xu(e,t,n);break;case 21:xu(e,t,n);break;case 22:K=(r=K)||n.memoizedState!==null,xu(e,t,n),K=r;break;case 30:Tl(n,t),xu(e,t,n);break;case 7:K||Tl(n,t),xu(e,t,n);break;default:xu(e,t,n)}}function Cu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null))){e=e.dehydrated;try{Hh(e)}catch(e){ff(t,t.return,e)}}}function wu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null&&(e=e.dehydrated,e!==null))))try{Hh(e)}catch(e){ff(t,t.return,e)}}function Tu(e){switch(e.tag){case 31:case 13:case 19:var t=e.stateNode;return t===null&&(t=e.stateNode=new ou),t;case 22:return e=e.stateNode,t=e._retryCache,t===null&&(t=e._retryCache=new ou),t;default:throw Error(o(435,e.tag))}}function Eu(e,t){var n=Tu(e);t.forEach(function(t){if(!n.has(t)){n.add(t);var r=_f.bind(null,e,t);t.then(r,r)}})}function Du(e,t,n){var r=t.deletions;if(r!==null)for(var i=0;i<r.length;i++){var a=r[i],s=e,c=t,l=c;a:for(;l!==null;){switch(l.tag){case 27:if(Sp(l.type)){yu=l.stateNode,bu=!1;break a}break;case 5:yu=l.stateNode,bu=!1;break a;case 3:case 4:yu=l.stateNode.containerInfo,bu=!0;break a}l=l.return}if(yu===null)throw Error(o(160));Su(s,c,a),yu=null,bu=!1,s=a.alternate,s!==null&&(s.return=null),a.return=null}if(t.subtreeFlags&13886)for(t=t.child;t!==null;)ku(t,e,n),t=t.sibling}var Ou=null;function ku(e,t,n){var r=e.alternate,i=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(i&4&&(r=e.updateQueue,r=r===null?null:r.events,r!==null))for(var a=0;a<r.length;a++){var s=r[a];s.ref.impl=s.nextImpl}Du(t,e,n),Au(e),i&4&&(xl(3,e,e.return),bl(3,e),xl(5,e,e.return));break;case 1:Du(t,e,n),Au(e),i&512&&(K||r===null||Tl(r,r.return)),i&64&&ru&&(e=e.updateQueue,e!==null&&(t=e.callbacks,t!==null&&(n=e.shared.hiddenCallbacks,e.shared.hiddenCallbacks=n===null?t:n.concat(t))));break;case 26:if(a=Ou,Du(t,e,n),Au(e),i&512&&(K||r===null||Tl(r,r.return)),i&4){if(i=r===null?null:r.memoizedState,n=e.memoizedState,r===null){if(n===null){if(e.stateNode===null){if(ru)e.stateNode=fp(e.type,e.memoizedProps,t.containerInfo,e);else{a:{t=e.type,n=e.memoizedProps,i=a.ownerDocument||a;b:switch(t){case`title`:r=i.getElementsByTagName(`title`)[0],(!r||r[Ot]||r[xt]||r.namespaceURI===`http://www.w3.org/2000/svg`||r.hasAttribute(`itemprop`))&&(r=i.createElement(t),i.head.insertBefore(r,i.querySelector(`head > title`))),np(r,t,n),r[xt]=e,Ft(r),t=r;break a;case`link`:if(a=Gm(`link`,`href`,i).get(t+(n.href||``))){for(s=0;s<a.length;s++)if(r=a[s],r.getAttribute(`href`)===(n.href==null||n.href===``?null:n.href)&&r.getAttribute(`rel`)===(n.rel==null?null:n.rel)&&r.getAttribute(`title`)===(n.title==null?null:n.title)&&r.getAttribute(`crossorigin`)===(n.crossOrigin==null?null:n.crossOrigin)){a.splice(s,1);break b}}r=i.createElement(t),np(r,t,n),i.head.appendChild(r);break;case`meta`:if(a=Gm(`meta`,`content`,i).get(t+(n.content||``))){for(s=0;s<a.length;s++)if(r=a[s],r.getAttribute(`content`)===(n.content==null?null:``+n.content)&&r.getAttribute(`name`)===(n.name==null?null:n.name)&&r.getAttribute(`property`)===(n.property==null?null:n.property)&&r.getAttribute(`http-equiv`)===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute(`charset`)===(n.charSet==null?null:n.charSet)){a.splice(s,1);break b}}r=i.createElement(t),np(r,t,n),i.head.appendChild(r);break;default:throw Error(o(468,t))}r[xt]=e,Ft(r),t=r}e.stateNode=t}}else ru||Km(a,e.type,e.stateNode)}else e.stateNode=Bm(a,n,e.memoizedProps)}else i===n?n===null&&e.stateNode!==null&&Ml(e,e.memoizedProps,r.memoizedProps):(i===null?(t=r.stateNode,t===null||K||t.parentNode.removeChild(t)):i.count--,n===null?ru||Km(a,e.type,e.stateNode):Bm(a,n,e.memoizedProps))}break;case 27:Du(t,e,n),Au(e),i&512&&(K||r===null||Tl(r,r.return)),r!==null&&i&4&&Ml(e,e.memoizedProps,r.memoizedProps);break;case 5:if(a=iu,iu=!1,Du(t,e,n),iu=a,Au(e),i&512&&(K||r===null||Tl(r,r.return)),e.flags&32){t=e.stateNode;try{ln(t,``),B=!0}catch(t){ff(e,e.return,t)}}i&4&&e.stateNode!=null&&(t=e.memoizedProps,Ml(e,t,r===null?t:r.memoizedProps)),i&1024&&(au=!0);break;case 6:if(Du(t,e,n),Au(e),i&4){if(e.stateNode===null)throw Error(o(162));t=e.memoizedProps,n=e.stateNode;try{n.nodeValue=t,B=!0}catch(t){ff(e,e.return,t)}}break;case 3:if(B=!1,Wm=null,a=Ou,Ou=bm(t.containerInfo),Du(t,e,n),Ou=a,Au(e),i&4&&r!==null&&r.memoizedState.isDehydrated)try{Hh(t.containerInfo)}catch(t){ff(e,e.return,t)}au&&(au=!1,ju(e)),B=!1;break;case 4:i=iu,iu=ru,r=Gt(),a=Ou,Ou=bm(e.stateNode.containerInfo),Du(t,e,n),Au(e),Ou=a,B&&lu&&(uu=!0),B=r,iu=i;break;case 12:Du(t,e,n),Au(e);break;case 31:Du(t,e,n),Au(e),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Eu(e,t)));break;case 13:Du(t,e,n),Au(e),e.child.flags&8192&&e.memoizedState!==null!=(r!==null&&r.memoizedState!==null)&&(pd=ze()),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Eu(e,t)));break;case 22:a=e.memoizedState!==null,s=r!==null&&r.memoizedState!==null;var c=ru,l=K,u=iu;ru=c||a,iu=u||a,K=l||s,Du(t,e,n),K=l,iu=u,ru=c,Au(e),i&8192&&(t=e.stateNode,t._visibility=a?t._visibility&-2:t._visibility|1,!a||r===null||s||ru||K||(t=s||K,n=ru,r=K,ru=a||ru,K=t,Fu(e,2),ru=n,K=r),!a&&iu||hu(e,a)),i&4&&(t=e.updateQueue,t!==null&&(n=t.retryQueue,n!==null&&(t.retryQueue=null,Eu(e,n))));break;case 19:Du(t,e,n),Au(e),i&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Eu(e,t)));break;case 30:i&512&&(K||r===null||Tl(r,r.return)),i=Gt(),a=lu,s=(n&335544064)===n,c=e.memoizedProps,lu=s&&vi(c.default,c.update)!==`none`,Du(t,e,n),Au(e),s&&r!==null&&B&&(e.flags|=4),lu=a,B=i;break;case 21:break;case 7:i&512&&(K||r===null||Tl(r,r.return)),r&&r.stateNode!==null&&(r.stateNode._fragmentFiber=e);default:Du(t,e,n),Au(e)}}function Au(e){var t=e.flags;if(t&2){try{for(var n,r=e.return;r!==null;){if(Nl(r)){n=r;break}r=r.return}r=null;for(var i=e.return;i!==null;){if(Al(i)){var a=i.stateNode;r===null?r=[a]:r.push(a)}if(kl(i))break;i=i.return}var s=r;if(n==null)throw Error(o(160));switch(n.tag){case 27:var c=n.stateNode;Il(e,Pl(e),c,s);break;case 5:var l=n.stateNode;n.flags&32&&(ln(l,``),n.flags&=-33),Il(e,Pl(e),l,s);break;case 3:case 4:var u=n.stateNode.containerInfo;Fl(e,Pl(e),u,s);break;default:throw Error(o(161))}}catch(t){ff(e,e.return,t)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function ju(e){if(e.subtreeFlags&1024)for(e=e.child;e!==null;){var t=e;ju(t),t.tag===5&&t.flags&1024&&(t=t.stateNode,gh=!0,t.reset(),gh=!1),e=e.sibling}}function Mu(e,t){if(t.subtreeFlags&9270)for(t=t.child;t!==null;)Nu(t,e),t=t.sibling;else nu(t,!1)}function Nu(e,t){var n=e.alternate;if(n===null)Jl(e,!1);else switch(e.tag){case 3:if(du=cu=!1,Hl(),Mu(t,e),!cu&&!uu){if(e=Vl,e!==null)for(var r=0;r<e.length;r+=3){n=e[r];var i=e[r+1];Ep(n,e[r+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(`+i+`)`})}e=t.containerInfo,e=e.nodeType===9?e.documentElement:e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===``&&(e.style.viewTransitionName=`none`,e.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(root)`}),e.animate({width:[0,0],height:[0,0]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition`})),du=!0}Vl=null;break;case 5:Mu(t,e);break;case 4:r=cu,cu=!1,Mu(t,e),cu&&(uu=!0),cu=r;break;case 22:e.memoizedState===null&&(n.memoizedState===null?Mu(t,e):Jl(e,!1));break;case 30:r=cu,i=Hl(),cu=!1,Mu(t,e),cu&&(e.flags|=4);var a=e.memoizedProps,o=e.stateNode;t=gi(a,o),o=gi(n.memoizedProps,o);var s=vi(a.default,a.update);s===`none`?t=!1:(a=n.memoizedState,n.memoizedState=null,n=e.child,Ul=0,t=tu(e,n,t,o,s,a,!0),Ul!==(a===null?0:a.length)&&(e.flags|=32)),e.flags&4&&t?(jd(e,e.memoizedProps.onUpdate),Vl=i):i!==null&&(i.push.apply(i,Vl),Vl=i),cu=e.flags&32?!0:r;break;default:Mu(t,e)}}function Pu(e,t){if(t.subtreeFlags&8772)for(t=t.child;t!==null;)mu(e,t.alternate,t),t=t.sibling}function Fu(e,t){for(e=e.child;e!==null;){var n=e,r=t;switch(n.tag){case 0:case 11:case 14:case 15:xl(4,n,n.return),Fu(n,r);break;case 1:Tl(n,n.return);var i=n.stateNode;typeof i.componentWillUnmount==`function`&&Cl(n,n.return,i),Fu(n,r);break;case 27:r&2&&gm(n.stateNode,n.type,n.memoizedProps);case 5:Tl(n,n.return),n.tag!==5&&n.tag!==27||Ol(n),Fu(n,r);break;case 6:Ol(n);break;case 26:Tl(n,n.return),i=n.stateNode,n.memoizedState!==null||i===null||K||i.parentNode.removeChild(i),Fu(n,r);break;case 22:n.memoizedState===null&&Fu(n,r);break;case 30:Tl(n,n.return),Fu(n,r);break;case 7:Tl(n,n.return);default:Fu(n,r)}e=e.sibling}}function Iu(e,t,n){for(n=t.subtreeFlags&8772?n:n&-2,t=t.child;t!==null;){var r=t.alternate,i=e,a=t,o=a.flags,s=!!(n&1);switch(a.tag){case 0:case 11:case 15:Iu(i,a,n),bl(4,a);break;case 1:if(Iu(i,a,n),r=a,i=r.stateNode,typeof i.componentDidMount==`function`)try{i.componentDidMount()}catch(e){ff(r,r.return,e)}if(r=a,i=r.updateQueue,i!==null){var c=r.stateNode;try{var l=i.shared.hiddenCallbacks;if(l!==null)for(i.shared.hiddenCallbacks=null,i=0;i<l.length;i++)So(l[i],c)}catch(e){ff(r,r.return,e)}}s&&o&64&&Sl(a),wl(a,a.return);break;case 27:n&2&&Ll(a);case 5:a.tag!==5&&a.tag!==27||Dl(a),Iu(i,a,n),s&&r===null&&o&4&&jl(a),wl(a,a.return);break;case 6:Dl(a);break;case 26:c=a.stateNode,a.memoizedState!==null||c===null||ru||Km(bm(c.ownerDocument),a.type,c),Iu(i,a,n),s&&r===null&&o&4&&jl(a),wl(a,a.return);break;case 12:Iu(i,a,n);break;case 31:Iu(i,a,n),s&&o&4&&Cu(i,a);break;case 13:Iu(i,a,n),s&&o&4&&wu(i,a);break;case 22:a.memoizedState===null&&Iu(i,a,n),wl(a,a.return);break;case 30:Iu(i,a,n),wl(a,a.return);break;case 7:wl(a,a.return);default:Iu(i,a,n)}t=t.sibling}}function Lu(e,t){var n=null;e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),e=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(e=t.memoizedState.cachePool.pool),e!==n&&(e!=null&&e.refCount++,n!=null&&Ma(n))}function Ru(e,t){e=null,t.alternate!==null&&(e=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==e&&(t.refCount++,e!=null&&Ma(e))}function zu(e,t,n,r){var i=(n&335544064)===n;if(t.subtreeFlags&(i?10262:10256))for(t=t.child;t!==null;)Bu(e,t,n,r),t=t.sibling;else i&&eu(t)}function Bu(e,t,n,r){var i=(n&335544064)===n;i&&t.alternate===null&&t.return!==null&&t.return.alternate!==null&&$l(t);var a=t.flags;switch(t.tag){case 0:case 11:case 15:zu(e,t,n,r),a&2048&&bl(9,t);break;case 1:zu(e,t,n,r);break;case 3:zu(e,t,n,r),i&&du&&(e=e.containerInfo,e=e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,e.style.viewTransitionName===`root`&&(e.style.viewTransitionName=``),e=e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===`none`&&(e.style.viewTransitionName=``)),a&2048&&(a=null,t.alternate!==null&&(a=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==a&&(t.refCount++,a!=null&&Ma(a)));break;case 12:if(a&2048){zu(e,t,n,r),a=t.stateNode;try{var o=t.memoizedProps,s=o.id,c=o.onPostCommit;typeof c==`function`&&c(s,t.alternate===null?`mount`:`update`,a.passiveEffectDuration,-0)}catch(e){ff(t,t.return,e)}}else zu(e,t,n,r);break;case 31:zu(e,t,n,r);break;case 13:zu(e,t,n,r);break;case 23:break;case 22:o=t.stateNode,s=t.alternate,t.memoizedState===null?(i&&s!==null&&s.memoizedState!==null&&$l(t),o._visibility&2?zu(e,t,n,r):(o._visibility|=2,Vu(e,t,n,r,!!(t.subtreeFlags&10256)||!1))):(i&&s!==null&&s.memoizedState===null&&$l(s),o._visibility&2?zu(e,t,n,r):Hu(e,t)),a&2048&&Lu(s,t);break;case 24:zu(e,t,n,r),a&2048&&Ru(t.alternate,t);break;case 30:i&&(a=t.alternate,a!==null&&(Kl(a.child,!0),Kl(t.child,!0))),zu(e,t,n,r);break;default:zu(e,t,n,r)}}function Vu(e,t,n,r,i){for(i&&=!!(t.subtreeFlags&10256)||!1,t=t.child;t!==null;){var a=e,o=t,s=n,c=r,l=o.flags;switch(o.tag){case 0:case 11:case 15:Vu(a,o,s,c,i),bl(8,o);break;case 23:break;case 22:var u=o.stateNode;o.memoizedState===null?(u._visibility|=2,Vu(a,o,s,c,i)):u._visibility&2?Vu(a,o,s,c,i):Hu(a,o),i&&l&2048&&Lu(o.alternate,o);break;case 24:Vu(a,o,s,c,i),i&&l&2048&&Ru(o.alternate,o);break;default:Vu(a,o,s,c,i)}t=t.sibling}}function Hu(e,t){if(t.subtreeFlags&10256)for(t=t.child;t!==null;){var n=e,r=t,i=r.flags;switch(r.tag){case 22:Hu(n,r),i&2048&&Lu(r.alternate,r);break;case 24:Hu(n,r),i&2048&&Ru(r.alternate,r);break;default:Hu(n,r)}t=t.sibling}}var Uu=8192;function Wu(e,t,n){if(e.subtreeFlags&Uu)for(e=e.child;e!==null;)Gu(e,t,n),e=e.sibling}function Gu(e,t,n){switch(e.tag){case 26:Wu(e,t,n),e.flags&Uu&&(e.memoizedState===null?(e=e.stateNode,(t&335544128)===t&&Zm(n,e)):Qm(n,Ou,e.memoizedState,e.memoizedProps));break;case 5:Wu(e,t,n),e.flags&Uu&&(e=e.stateNode,(t&335544128)===t&&Zm(n,e));break;case 3:case 4:var r=Ou;Ou=bm(e.stateNode.containerInfo),Wu(e,t,n),Ou=r;break;case 22:e.memoizedState===null&&(r=e.alternate,r!==null&&r.memoizedState!==null?(r=Uu,Uu=16777216,Wu(e,t,n),Uu=r):Wu(e,t,n));break;case 30:if((e.flags&Uu)!==0&&(r=e.memoizedProps.name,r!=null&&r!==`auto`)){var i=e.stateNode;i.paired=null,zl===null&&(zl=new Map),zl.set(r,i)}Wu(e,t,n);break;default:Wu(e,t,n)}}function Ku(e){var t=e.alternate;if(t!==null&&(e=t.child,e!==null)){t.child=null;do t=e.sibling,e.sibling=null,e=t;while(e!==null)}}function qu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];su=r,Xu(r,e)}Ku(e)}if(e.subtreeFlags&10256)for(e=e.child;e!==null;)Ju(e),e=e.sibling}function Ju(e){switch(e.tag){case 0:case 11:case 15:qu(e),e.flags&2048&&xl(9,e,e.return);break;case 3:qu(e);break;case 12:qu(e);break;case 22:var t=e.stateNode;e.memoizedState!==null&&t._visibility&2&&(e.return===null||e.return.tag!==13)?(t._visibility&=-3,Yu(e)):qu(e);break;default:qu(e)}}function Yu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];su=r,Xu(r,e)}Ku(e)}for(e=e.child;e!==null;){switch(t=e,t.tag){case 0:case 11:case 15:xl(8,t,t.return),Yu(t);break;case 22:n=t.stateNode,n._visibility&2&&(n._visibility&=-3,Yu(t));break;default:Yu(t)}e=e.sibling}}function Xu(e,t){for(;su!==null;){var n=su;switch(n.tag){case 0:case 11:case 15:xl(8,n,t);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var r=n.memoizedState.cachePool.pool;r!=null&&r.refCount++}break;case 24:Ma(n.memoizedState.cache)}if(r=n.child,r!==null)r.return=n,su=r;else a:for(n=e;su!==null;){r=su;var i=r.sibling,a=r.return;if(vu(r),r===n){su=null;break a}if(i!==null){i.return=a,su=i;break a}su=a}}}var Zu={getCacheForType:function(e){var t=wa(Aa),n=t.data.get(e);return n===void 0&&(n=e(),t.data.set(e,n)),n},cacheSignal:function(){return wa(Aa).controller.signal}},Qu=typeof WeakMap==`function`?WeakMap:Map,q=0,$u=null,J=null,Y=0,X=0,ed=null,td=!1,nd=!1,rd=!1,id=0,ad=0,Z=0,od=0,sd=0,cd=0,ld=0,ud=null,dd=null,fd=!1,pd=0,md=0,Q=1/0,hd=null,gd=null,_d=0,vd=null,yd=null,bd=0,xd=0,Sd=null,Cd=null,wd=null,Td=null,Ed=null,Dd=0,Od=null;function kd(){return q&2&&Y!==0?Y&-Y:N.T===null?vt():Nf()}function Ad(){if(cd===0){if(!(Y&536870912)||V){var e=nt;nt<<=1,!(nt&3932160)&&(nt=262144),cd=e}else cd=536870912}return e=ko.current,e!==null&&(e.flags|=32),cd}function jd(e,t){if(t!=null){var n=e.stateNode,r=n.ref;r===null&&(r=n.ref=Pp(gi(e.memoizedProps,n))),Td===null&&(Td=[]),Td.push(t.bind(null,r))}}function Md(e,t,n){(e===$u&&(X===2||X===9)||e.cancelPendingCommit!==null)&&(zd(e,0),Id(e,Y,cd,!1)),dt(e,n),q&2&&e===$u||(e===$u&&(!(q&2)&&(od|=n),ad===4&&Id(e,Y,cd,!1)),Tf(e))}function Nd(e,t,n){if(q&6)throw Error(o(327));var r=!n&&!(t&127)&&(t&e.expiredLanes)===0||ot(e,t),i=r?qd(e,t):Gd(e,t,!0),a=r;do{if(i===0){nd&&!r&&Id(e,t,0,!1);break}if(n=e.current.alternate,a&&!Fd(n))i=Gd(e,t,!1),a=!1;else{if(i===2){if(a=t,e.errorRecoveryDisabledLanes&a)var s=0;else s=e.pendingLanes&-536870913,s=s===0?s&536870912?536870912:0:s;if(s!==0){t=s;a:{var c=e;i=ud;var l=c.current.memoizedState.isDehydrated;if(l&&(zd(c,s).flags|=256),s=Gd(c,s,!1),s!==2&&s!==6){if(rd&&!l){c.errorRecoveryDisabledLanes|=a,od|=a,i=4;break a}a=dd,dd=i,a!==null&&(dd===null?dd=a:dd.push.apply(dd,a))}i=s}if(a=!1,i!==2)continue}}if(i===1){zd(e,0),Id(e,t,0,!0);break}a:{switch(r=e,a=i,a){case 0:case 1:throw Error(o(345));case 4:if((t&4194048)!==t&&(t&62914560)!==t)break;case 6:Id(r,t,cd,!td);break a;case 2:dd=null;break;case 3:case 5:break;default:throw Error(o(329))}if((t&62914560)===t&&(i=pd+300-ze(),10<i)){if(Id(r,t,cd,!td),at(r,0,!0)!==0)break a;bd=t,r.timeoutHandle=gp(Pd.bind(null,r,n,dd,hd,fd,t,cd,od,ld,td,a,`Throttled`,-0,0),i);break a}Pd(r,n,dd,hd,fd,t,cd,od,ld,td,a,null,-0,0)}break}}while(1);Tf(e)}function Pd(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.timeoutHandle=-1;var m=t.subtreeFlags,h=(a&335544064)===a;d=null,(h||m&8192||(m&16785408)==16785408)&&(d={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:_n},zl=null,Gu(t,a,d),h&&(m=d,h=e.containerInfo,h=(h.nodeType===9?h:h.ownerDocument).__reactViewTransition,h!=null&&(m.count++,m.waitingForViewTransition=!0,m=nh.bind(m),h.finished.then(m,m))),m=(a&62914560)===a?pd-ze():(a&4194048)===a?md-ze():0,m=eh(d,m),m!==null)?(bd=a,e.cancelPendingCommit=m(ef.bind(null,e,t,a,n,r,i,o,s,c,l,u,d,null,f,p)),Id(e,a,o,!l)):ef(e,t,a,n,r,i,o,s,c,l,u,d)}function Fd(e){for(var t=e;;){var n=t.tag;if((n===0||n===11||n===15)&&t.flags&16384&&(n=t.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Br(a(),i))return!1}catch{return!1}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Id(e,t,n,r){t=st(e,t),t&=~sd,t&=~od,e.suspendedLanes|=t,e.pingedLanes&=~t,r&&(e.warmLanes|=t),r=e.expirationTimes;for(var i=t;0<i;){var a=31-Ze(i),o=1<<a;r[a]=-1,i&=~o}n!==0&&pt(e,n,t)}function Ld(){return q&6?!0:(Ef(0,!1),!1)}function Rd(){if(J!==null){if(X===0)var e=J.return;else e=J,ga=ha=null,ns(e),ro=null,io=0,e=J;for(;e!==null;)yl(e.alternate,e),e=e.return;J=null}}function zd(e,t){var n=e.timeoutHandle;return n!==-1&&(e.timeoutHandle=-1,_p(n)),n=e.cancelPendingCommit,n!==null&&(e.cancelPendingCommit=null,n()),bd=0,Rd(),$u=e,J=n=Ni(e.current,null),Y=t,X=0,ed=null,td=!1,nd=ot(e,t),rd=!1,ld=cd=sd=od=Z=ad=0,dd=ud=null,fd=!1,id=st(e,t),Ci(),n}function Bd(e,t){W=null,N.H=fc,t===Ja||t===Ya?(t=to(),X=3):t===H?(t=to(),X=4):X=t===Ac?8:typeof t==`object`&&t&&typeof t.then==`function`?6:1,ed=t,J===null&&(ad=1,wc(e,Vi(t,e.current)))}function Vd(){var e=ko.current;return e===null?!0:(Y&4194048)===Y?Ao===null:(Y&62914560)===Y||Y&536870912?e===Ao:!1}function Hd(){var e=N.H;return N.H=fc,e===null?fc:e}function Ud(){var e=N.A;return N.A=Zu,e}function Wd(){ad=4,td||(Y&4194048)!==Y&&ko.current!==null||(nd=!0),!(Z&134217727)&&!(od&134217727)||$u===null||Id($u,Y,cd,!1)}function Gd(e,t,n){var r=q;q|=2;var i=Hd(),a=Ud();($u!==e||Y!==t)&&(hd=null,zd(e,t)),t=!1;var o=ad;a:do try{if(X!==0&&J!==null){var s=J,c=ed;switch(X){case 8:Rd(),o=6;break a;case 3:case 2:case 9:case 6:ko.current===null&&(t=!0);var l=X;if(X=0,ed=null,Zd(e,s,c,l),n&&nd){o=0;break a}break;default:l=X,X=0,ed=null,Zd(e,s,c,l)}}Kd(),o=ad;break}catch(t){Bd(e,t)}while(1);return t&&e.shellSuspendCounter++,ga=ha=null,q=r,N.H=i,N.A=a,J===null&&($u=null,Y=0,Ci()),o}function Kd(){for(;J!==null;)Yd(J)}function qd(e,t){var n=q;q|=2;var r=Hd(),i=Ud();$u!==e||Y!==t?(hd=null,Q=ze()+500,zd(e,t)):nd=ot(e,t);a:do try{if(X!==0&&J!==null){t=J;var a=ed;b:switch(X){case 1:X=0,ed=null,Zd(e,t,a,1);break;case 2:case 9:if(Za(a)){X=0,ed=null,Xd(t);break}t=function(){X!==2&&X!==9||$u!==e||(X=7),Tf(e)},a.then(t,t);break a;case 3:X=7;break a;case 4:X=5;break a;case 7:Za(a)?(X=0,ed=null,Xd(t)):(X=0,ed=null,Zd(e,t,a,7));break;case 5:var s=null;switch(J.tag){case 26:s=J.memoizedState;case 5:case 27:var c=J;if(s?Ym(s):c.stateNode.complete){X=0,ed=null;var l=c.sibling;if(l!==null)J=l;else{var u=c.return;u===null?J=null:(J=u,Qd(u))}break b}}X=0,ed=null,Zd(e,t,a,5);break;case 6:X=0,ed=null,Zd(e,t,a,6);break;case 8:Rd(),ad=6;break a;default:throw Error(o(462))}}Jd();break}catch(t){Bd(e,t)}while(1);return ga=ha=null,N.H=r,N.A=i,q=n,J===null?($u=null,Y=0,Ci(),ad):0}function Jd(){for(;J!==null&&!Le();)Yd(J)}function Yd(e){var t=ul(e.alternate,e,id);e.memoizedProps=e.pendingProps,t===null?Qd(e):J=t}function Xd(e){var t=e,n=t.alternate;switch(t.tag){case 15:case 0:t=Wc(n,t,t.pendingProps,t.type,void 0,Y);break;case 11:t=Wc(n,t,t.pendingProps,t.type.render,t.ref,Y);break;case 5:ns(t);var r=t;r===na&&(V?(la(r),r.tag===5&&r.stateNode!=null&&(ra=r.stateNode)):(la(r),V=!0));default:yl(n,t),t=J=Pi(t,id),t=ul(n,t,id)}e.memoizedProps=e.pendingProps,t===null?Qd(e):J=t}function Zd(e,t,n,r){ga=ha=null,ns(t),ro=null,io=0;var i=t.return;try{if(kc(e,i,t,n,Y)){ad=1,wc(e,Vi(n,e.current)),J=null;return}}catch(t){if(i!==null)throw J=i,t;ad=1,wc(e,Vi(n,e.current)),J=null;return}t.flags&32768?(V||r===1?e=!0:nd||Y&536870912?e=!1:(td=e=!0,(r===2||r===9||r===3||r===6)&&(r=ko.current,r!==null&&r.tag===13&&(r.flags|=16384))),$d(t,e)):Qd(t)}function Qd(e){var t=e;do{if(t.flags&32768){$d(t,td);return}e=t.return;var n=_l(t.alternate,t,id);if(n!==null){J=n;return}if(t=t.sibling,t!==null){J=t;return}J=t=e}while(t!==null);ad===0&&(ad=5)}function $d(e,t){do{var n=vl(e.alternate,e);if(n!==null){n.flags&=32767,J=n;return}if(n=e.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!t&&(e=e.sibling,e!==null)){J=e;return}J=e=n}while(e!==null);ad=6,J=null}function ef(e,t,n,r,i,a,s,c,l,u,d,f){e.cancelPendingCommit=null;do lf();while(_d!==0);if(q&6)throw Error(o(327));if(t!==null){if(t===e.current)throw Error(o(177));e===$u&&(J=$u=null,Y=0),yd=t,vd=e,bd=n,Sd=i,Cd=r,tf(e,t,n,s,c,l,f)}}function tf(e,t,n,r,i,a,o){var s=t.lanes|t.childLanes;if(xd=s,s|=Si,ft(e,n,s,r,i,a),Td=null,(n&335544064)===n?(Ed=Fa(e),r=10262):(Ed=null,r=10256),(t.subtreeFlags&r)!==0||(t.flags&r)!==0?(e.callbackNode=null,e.callbackPriority=0,vf(Ue,function(){return uf(),null})):(e.callbackNode=null,e.callbackPriority=0),Rl=!1,r=!!(t.flags&13878),t.subtreeFlags&13878||r){r=N.T,N.T=null,i=P.p,P.p=2,a=q,q|=4;try{fu(e,t,n)}finally{q=a,P.p=i,N.T=r}}_d=1,Rl?wd=Mp(o,e.containerInfo,Ed,af,of,rf,sf,uf,nf,null,null):(af(),of(),sf())}function nf(e){if(_d!==0){var t=vd.onRecoverableError;t(e,{componentStack:null})}}function rf(){_d===3&&(_d=0,Nu(yd,vd),_d=4)}function af(){if(_d===1){_d=0;var e=vd,t=yd,n=bd,r=!!(t.flags&13878);if(t.subtreeFlags&13878||r){r=N.T,N.T=null;var i=P.p;P.p=2;var a=q;q|=4;try{lu=uu=!1,ku(t,e,n),n=cp;var o=Kr(e.containerInfo),s=n.focusedElem,c=n.selectionRange;if(o!==s&&s&&s.ownerDocument&&Gr(s.ownerDocument.documentElement,s)){if(c!==null&&qr(s)){var l=c.start,u=c.end;if(u===void 0&&(u=l),`selectionStart`in s)s.selectionStart=l,s.selectionEnd=Math.min(u,s.value.length);else{var d=s.ownerDocument||document,f=d&&d.defaultView||window;if(f.getSelection){var p=f.getSelection(),m=s.textContent.length,h=Math.min(c.start,m),g=c.end===void 0?h:Math.min(c.end,m);!p.extend&&h>g&&(o=g,g=h,h=o);var _=Wr(s,h),v=Wr(s,g);if(_&&v&&(p.rangeCount!==1||p.anchorNode!==_.node||p.anchorOffset!==_.offset||p.focusNode!==v.node||p.focusOffset!==v.offset)){var y=d.createRange();y.setStart(_.node,_.offset),p.removeAllRanges(),h>g?(p.addRange(y),p.extend(v.node,v.offset)):(y.setEnd(v.node,v.offset),p.addRange(y))}}}}for(d=[],p=s;p=p.parentNode;)p.nodeType===1&&d.push({element:p,left:p.scrollLeft,top:p.scrollTop});for(typeof s.focus==`function`&&s.focus(),s=0;s<d.length;s++){var b=d[s];b.element.scrollLeft=b.left,b.element.scrollTop=b.top}}gh=!!sp,cp=sp=null}finally{q=a,P.p=i,N.T=r}}e.current=t,_d=2}}function of(){if(_d===2){_d=0;var e=vd,t=yd,n=!!(t.flags&8772);if(t.subtreeFlags&8772||n){n=N.T,N.T=null;var r=P.p;P.p=2;var i=q;q|=4;try{mu(e,t.alternate,t)}finally{q=i,P.p=r,N.T=n}}_d=3}}function sf(){if(_d===4||_d===3){_d=0;var e=wd;wd=null,Re();var t=vd,n=yd,r=bd,i=Cd,a=(r&335544064)===r?10262:10256;if((n.subtreeFlags&a)!==0||(n.flags&a)!==0?_d=5:(_d=0,yd=vd=null,cf(t,t.pendingLanes)),a=t.pendingLanes,a===0&&(gd=null),_t(r),n=n.stateNode,Ye&&typeof Ye.onCommitFiberRoot==`function`)try{Ye.onCommitFiberRoot(Je,n,void 0,(n.current.flags&128)==128)}catch{}if(i!==null){n=N.T,a=P.p,P.p=2,N.T=null;try{for(var o=t.onRecoverableError,s=0;s<i.length;s++){var c=i[s];o(c.value,{componentStack:c.stack})}}finally{N.T=n,P.p=a}}if(i=Td,o=Ed,Ed=null,i!==null&&(Td=null,o===null&&(o=[]),e!==null))for(c=0;c<i.length;c++)n=(0,i[c])(o),n!==void 0&&e.finished.finally(n);bd&3&&lf(),Tf(t),a=t.pendingLanes,r&261930&&a&42?t===Od?Dd++:(Dd=0,Od=t):(Dd=0,Od=null),Ef(0,!1)}}function cf(e,t){(e.pooledCacheLanes&=t)===0&&(t=e.pooledCache,t!=null&&(e.pooledCache=null,Ma(t)))}function lf(){return wd!==null&&(wd.skipTransition(),wd=null),af(),of(),sf(),uf()}function uf(){if(_d!==5)return!1;var e=vd,t=xd;xd=0;var n=_t(bd),r=N.T,i=P.p;try{P.p=32>n?32:n,N.T=null,n=Sd,Sd=null;var a=vd,s=bd;if(_d=0,yd=vd=null,bd=0,q&6)throw Error(o(331));var c=q;if(q|=4,Ju(a.current),Bu(a,a.current,s,n),q=c,Ef(0,!1),Ye&&typeof Ye.onPostCommitFiberRoot==`function`)try{Ye.onPostCommitFiberRoot(Je,a)}catch{}return!0}finally{P.p=i,N.T=r,cf(e,t)}}function df(e,t,n){t=Vi(n,t),t=Ec(e.stateNode,t,2),e=go(e,t,2),e!==null&&(dt(e,2),Tf(e))}function ff(e,t,n){if(e.tag===3)df(e,e,n);else for(;t!==null;){if(t.tag===3){df(t,e,n);break}if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError==`function`||typeof r.componentDidCatch==`function`&&(gd===null||!gd.has(r))){e=Vi(n,e),n=Dc(2),r=go(t,n,2),r!==null&&(Oc(n,r,t,e),dt(r,2),Tf(r));break}}t=t.return}}function pf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Qu;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(rd=!0,i.add(n),e=mf.bind(null,e,t,n),t.then(e,e))}function mf(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),e.pingedLanes|=e.suspendedLanes&n,e.warmLanes&=~n,$u===e&&(Y&n)===n&&(ad===4||ad===3&&(Y&62914560)===Y&&300>ze()-pd?q&2?sd|=n:zd(e,0):sd|=n,ld===Y&&(ld=0)),Tf(e)}function hf(e,t){t===0&&(t=lt()),e=Ei(e,t),e!==null&&(dt(e,t),Tf(e))}function gf(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),hf(e,n)}function _f(e,t){var n=0;switch(e.tag){case 31:case 13:var r=e.stateNode,i=e.memoizedState;i!==null&&(n=i.retryLane);break;case 19:r=e.stateNode;break;case 22:r=e.stateNode._retryCache;break;default:throw Error(o(314))}r!==null&&r.delete(t),hf(e,n)}function vf(e,t){return R(e,t)}var yf=null,bf=null,xf=!1,Sf=!1,Cf=!1,wf=0;function Tf(e){e!==bf&&e.next===null&&(bf===null?yf=bf=e:bf=bf.next=e),Sf=!0,xf||(xf=!0,Mf())}function Ef(e,t){if(!Cf&&Sf){Cf=!0;do for(var n=!1,r=yf;r!==null;){if(!t){if(e!==0){var i=r.pendingLanes;if(i===0)var a=0;else{var o=r.suspendedLanes,s=r.pingedLanes;a=(1<<31-Ze(42|e)+1)-1,a&=i&~(o&~s),a=a&201326741?a&201326741|1:a?a|2:0}a!==0&&(n=!0,jf(r,a))}else a=Y,a=at(r,r===$u?a:0,r.cancelPendingCommit!==null||r.timeoutHandle!==-1),!(a&3)||ot(r,a)||(n=!0,jf(r,a))}r=r.next}while(n);Cf=!1}}function Df(){Of()}function Of(){Sf=xf=!1;var e=0;wf!==0&&hp()&&(e=wf);for(var t=ze(),n=null,r=yf;r!==null;){var i=r.next,a=kf(r,t);a===0?(r.next=null,n===null?yf=i:n.next=i,i===null&&(bf=n)):(n=r,(e!==0||a&3)&&(Sf=!0)),r=i}_d!==0&&_d!==5||Ef(e,!1),wf!==0&&(wf=0)}function kf(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes&-62914561;0<a;){var o=31-Ze(a),s=1<<o,c=i[o];c===-1?((s&n)===0||(s&r)!==0)&&(i[o]=ct(s,t)):c<=t&&(e.expiredLanes|=s),a&=~s}if(t=$u,n=Y,n=at(e,e===t?n:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r=e.callbackNode,n===0||e===t&&(X===2||X===9)||e.cancelPendingCommit!==null)return r!==null&&r!==null&&Ie(r),e.callbackNode=null,e.callbackPriority=0;if(!(n&3)||ot(e,n)){if(t=n&-n,t===e.callbackPriority)return t;switch(r!==null&&Ie(r),_t(n)){case 2:case 8:n=He;break;case 32:n=Ue;break;case 268435456:n=Ge;break;default:n=Ue}return r=Af.bind(null,e),n=R(n,r),e.callbackPriority=t,e.callbackNode=n,t}return r!==null&&r!==null&&Ie(r),e.callbackPriority=2,e.callbackNode=null,2}function Af(e,t){if(_d!==0&&_d!==5)return e.callbackNode=null,e.callbackPriority=0,null;var n=e.callbackNode;if(lf()&&e.callbackNode!==n)return null;var r=Y;return r=at(e,e===$u?r:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r===0?null:(Nd(e,r,t),kf(e,ze()),e.callbackNode!=null&&e.callbackNode===n?Af.bind(null,e):null)}function jf(e,t){if(lf())return null;Nd(e,t,!0)}function Mf(){bp(function(){q&6?R(Ve,Df):Of()})}function Nf(){if(wf===0){var e=Ra;e===0&&(e=tt,tt<<=1,!(tt&261888)&&(tt=256)),wf=e}return wf}function Pf(e){return e==null||typeof e==`symbol`||typeof e==`boolean`?null:typeof e==`function`?e:gn(e)}function Ff(e,t,n,r,i){if(t===`submit`&&n&&n.stateNode===i){var a=Pf((i[St]||null).action),o=r.submitter;o&&(t=(t=o[St]||null)?Pf(t.formAction):o.getAttribute(`formAction`),t!==null&&(a=t,o=null));var s=new Rn(`action`,`action`,null,r,i);e.push({event:s,listeners:[{instance:null,listener:function(){if(r.defaultPrevented){if(wf!==0){var e=new FormData(i,o);Qs(n,{pending:!0,data:e,method:i.method,action:a},null,e)}}else typeof a==`function`&&(s.preventDefault(),e=new FormData(i,o),Qs(n,{pending:!0,data:e,method:i.method,action:a},a,e))},currentTarget:i}]})}}for(var If=0;If<pi.length;If++){var Lf=pi[If];mi(Lf.toLowerCase(),`on`+(Lf[0].toUpperCase()+Lf.slice(1)))}mi(ai,`onAnimationEnd`),mi(oi,`onAnimationIteration`),mi(si,`onAnimationStart`),mi(`dblclick`,`onDoubleClick`),mi(`focusin`,`onFocus`),mi(`focusout`,`onBlur`),mi(ci,`onTransitionRun`),mi(li,`onTransitionStart`),mi(ui,`onTransitionCancel`),mi(di,`onTransitionEnd`),Bt(`onMouseEnter`,[`mouseout`,`mouseover`]),Bt(`onMouseLeave`,[`mouseout`,`mouseover`]),Bt(`onPointerEnter`,[`pointerout`,`pointerover`]),Bt(`onPointerLeave`,[`pointerout`,`pointerover`]),zt(`onChange`,`change click focusin focusout input keydown keyup selectionchange`.split(` `)),zt(`onSelect`,`focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange`.split(` `)),zt(`onBeforeInput`,[`compositionend`,`keypress`,`textInput`,`paste`]),zt(`onCompositionEnd`,`compositionend focusout keydown keypress keyup mousedown`.split(` `)),zt(`onCompositionStart`,`compositionstart focusout keydown keypress keyup mousedown`.split(` `)),zt(`onCompositionUpdate`,`compositionupdate focusout keydown keypress keyup mousedown`.split(` `));var Rf=`abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting`.split(` `),zf=new Set(`beforetoggle cancel close invalid load scroll scrollend toggle`.split(` `).concat(Rf));function Bf(e,t){t=!!(t&4);for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;a:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],c=s.instance,l=s.currentTarget;if(s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){yi(e)}i.currentTarget=null,a=c}else for(o=0;o<r.length;o++){if(s=r[o],c=s.instance,l=s.currentTarget,s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){yi(e)}i.currentTarget=null,a=c}}}}function $(e,t){var n=t[wt];n===void 0&&(n=t[wt]=new Set);var r=e+`__bubble`;n.has(r)||(Wf(t,e,2,!1),n.add(r))}function Vf(e,t,n){var r=0;t&&(r|=4),Wf(n,e,r,t)}var Hf=`_reactListening`+Math.random().toString(36).slice(2);function Uf(e){if(!e[Hf]){e[Hf]=!0,Lt.forEach(function(t){t!==`selectionchange`&&(zf.has(t)||Vf(t,!1,e),Vf(t,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Hf]||(t[Hf]=!0,Vf(`selectionchange`,!1,t))}}function Wf(e,t,n,r){switch(Ch(t)){case 2:var i=_h;break;case 8:i=vh;break;default:i=yh}n=i.bind(null,t,n,e),i=void 0,!Dn||t!==`touchstart`&&t!==`touchmove`&&t!==`wheel`||(i=!0),r?i===void 0?e.addEventListener(t,n,!0):e.addEventListener(t,n,{capture:!0,passive:i}):i===void 0?e.addEventListener(t,n,!1):e.addEventListener(t,n,{passive:i})}function Gf(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)a:for(;;){if(r===null)return;var o=r.tag;if(o===3||o===4){var s=r.stateNode.containerInfo;if(s===i)break;if(o===4)for(o=r.return;o!==null;){var l=o.tag;if((l===3||l===4)&&o.stateNode.containerInfo===i)return;o=o.return}for(;s!==null;){if(o=jt(s),o===null)return;if(l=o.tag,l===5||l===6||l===26||l===27){r=a=o;continue a}s=s.parentNode}}r=r.return}wn(function(){var r=a,i=yn(n),o=[];a:{var s=fi.get(e);if(s!==void 0){var l=Rn,u=e;switch(e){case`keypress`:if(Nn(n)===0)break a;case`keydown`:case`keyup`:l=nr;break;case`focusin`:u=`focus`,l=qn;break;case`focusout`:u=`blur`,l=qn;break;case`beforeblur`:case`afterblur`:l=qn;break;case`click`:if(n.button===2)break a;case`auxclick`:case`dblclick`:case`mousedown`:case`mousemove`:case`mouseup`:case`mouseout`:case`mouseover`:case`contextmenu`:l=Gn;break;case`drag`:case`dragend`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`dragstart`:case`drop`:l=Kn;break;case`touchcancel`:case`touchend`:case`touchmove`:case`touchstart`:l=ar;break;case ai:case oi:case si:l=Jn;break;case di:l=or;break;case`scroll`:case`scrollend`:l=Bn;break;case`wheel`:l=sr;break;case`copy`:case`cut`:case`paste`:l=Yn;break;case`gotpointercapture`:case`lostpointercapture`:case`pointercancel`:case`pointerdown`:case`pointermove`:case`pointerout`:case`pointerover`:case`pointerup`:l=rr;break;case`submit`:l=ir;break;case`toggle`:case`beforetoggle`:l=cr}var d=!!(t&4),f=!d&&(e===`scroll`||e===`scrollend`),p=d?s===null?null:s+`Capture`:s;d=[];for(var m=r,h;m!==null;){var g=m;if(h=g.stateNode,g=g.tag,g!==5&&g!==26&&g!==27||h===null||p===null||(g=Tn(m,p),g!=null&&d.push(Kf(m,g,h))),f)break;m=m.return}0<d.length&&(s=new l(s,u,null,n,i),o.push({event:s,listeners:d}))}}if(!(t&7)){a:{if(l=e===`mouseover`||e===`pointerover`,s=e===`mouseout`||e===`pointerout`,l&&n!==vn&&(u=n.relatedTarget||n.fromElement)&&(jt(u)||u[Ct]))break a;(s||l)&&(u=i.window===i?i:(l=i.ownerDocument)?l.defaultView||l.parentWindow:window,s?(l=n.relatedTarget||n.toElement,s=r,l=l?jt(l):null,l!==null&&(f=c(l),d=l.tag,l!==f||d!==5&&d!==27&&d!==6)&&(l=null)):(s=null,l=r),s!==l&&(d=Gn,g=`onMouseLeave`,p=`onMouseEnter`,m=`mouse`,(e===`pointerout`||e===`pointerover`)&&(d=rr,g=`onPointerLeave`,p=`onPointerEnter`,m=`pointer`),f=s==null?u:Nt(s),h=l==null?u:Nt(l),u=new d(g,m+`leave`,s,n,i),u.target=f,u.relatedTarget=h,g=null,jt(i)===r&&(d=new d(p,m+`enter`,l,n,i),d.target=h,d.relatedTarget=f,g=d),f=g,d=s&&l?T(s,l,Jf):null,s!==null&&Yf(o,u,s,d,!1),l!==null&&f!==null&&Yf(o,f,l,d,!0)))}a:{if(s=r?Nt(r):window,l=s.nodeName&&s.nodeName.toLowerCase(),l===`select`||l===`input`&&s.type===`file`)var _=Or;else if(Sr(s)){if(kr)_=Rr;else{_=Ir;var v=Fr}}else l=s.nodeName,!l||l.toLowerCase()!==`input`||s.type!==`checkbox`&&s.type!==`radio`?r&&pn(r.elementType)&&(_=Or):_=Lr;if(_&&=_(e,r)){Cr(o,_,n,i);break a}v&&v(e,s,r)}switch(v=r?Nt(r):window,e){case`focusin`:(Sr(v)||v.contentEditable===`true`)&&(Yr=v,Xr=r,Zr=null);break;case`focusout`:Zr=Xr=Yr=null;break;case`mousedown`:Qr=!0;break;case`contextmenu`:case`mouseup`:case`dragend`:Qr=!1,$r(o,n,i);break;case`selectionchange`:if(Jr)break;case`keydown`:case`keyup`:$r(o,n,i)}var y;if(ur)b:{switch(e){case`compositionstart`:var b=`onCompositionStart`;break b;case`compositionend`:b=`onCompositionEnd`;break b;case`compositionupdate`:b=`onCompositionUpdate`;break b}b=void 0}else vr?gr(e,n)&&(b=`onCompositionEnd`):e===`keydown`&&n.keyCode===229&&(b=`onCompositionStart`);b&&(pr&&n.locale!==`ko`&&(vr||b!==`onCompositionStart`?b===`onCompositionEnd`&&vr&&(y=Mn()):(kn=i,An=`value`in kn?kn.value:kn.textContent,vr=!0)),v=qf(r,b),0<v.length&&(b=new Xn(b,e,null,n,i),o.push({event:b,listeners:v}),y?b.data=y:(y=_r(n),y!==null&&(b.data=y)))),(y=fr?yr(e,n):br(e,n))&&(b=qf(r,`onBeforeInput`),0<b.length&&(v=new Xn(`onBeforeInput`,`beforeinput`,null,n,i),o.push({event:v,listeners:b}),v.data=y)),Ff(o,e,r,n,i)}Bf(o,t)})}function Kf(e,t,n){return{instance:e,listener:t,currentTarget:n}}function qf(e,t){for(var n=t+`Capture`,r=[];e!==null;){var i=e,a=i.stateNode;if(i=i.tag,i!==5&&i!==26&&i!==27||a===null||(i=Tn(e,n),i!=null&&r.unshift(Kf(e,i,a)),i=Tn(e,t),i!=null&&r.push(Kf(e,i,a))),e.tag===3)return r;e=e.return}return[]}function Jf(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5&&e.tag!==27);return e||null}function Yf(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,c=s.alternate,l=s.stateNode;if(s=s.tag,c!==null&&c===r)break;s!==5&&s!==26&&s!==27||l===null||(c=l,i?(l=Tn(n,a),l!=null&&o.unshift(Kf(n,l,c))):i||(l=Tn(n,a),l!=null&&o.push(Kf(n,l,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Xf=/\r\n?/g,Zf=/\u0000|\uFFFD/g;function Qf(e){return(typeof e==`string`?e:``+e).replace(Xf,`
`).replace(Zf,``)}function $f(e,t){return t=Qf(t),Qf(e)===t}function ep(e,t,n,r,i,a){switch(n){case`children`:if(typeof r==`string`)t===`body`||t===`textarea`&&r===``||ln(e,r);else if(typeof r==`number`||typeof r==`bigint`)t!==`body`&&ln(e,``+r);else return;break;case`className`:qt(e,`class`,r);break;case`tabIndex`:qt(e,`tabindex`,r);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:qt(e,n,r);break;case`style`:fn(e,r,a);return;case`data`:if(t!==`object`){qt(e,`data`,r);break}case`src`:case`href`:if(r===``&&(t!==`a`||n!==`href`)||r==null||typeof r==`function`||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=gn(r),e.setAttribute(n,r);break;case`action`:case`formAction`:if(typeof r==`function`){e.setAttribute(n,`javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')`);break}if(typeof a==`function`&&(n===`formAction`?(t!==`input`&&ep(e,t,`name`,i.name,i,null),ep(e,t,`formEncType`,i.formEncType,i,null),ep(e,t,`formMethod`,i.formMethod,i,null),ep(e,t,`formTarget`,i.formTarget,i,null)):(ep(e,t,`encType`,i.encType,i,null),ep(e,t,`method`,i.method,i,null),ep(e,t,`target`,i.target,i,null))),r==null||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=gn(r),e.setAttribute(n,r);break;case`onClick`:r!=null&&(e.onclick=_n);return;case`onScroll`:r!=null&&$(`scroll`,e);return;case`onScrollEnd`:r!=null&&$(`scrollend`,e);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(o(61));if(n=r.__html,n!=null){if(i.children!=null)throw Error(o(60));a?.__html!==n&&(e.innerHTML=n)}}break;case`multiple`:e.multiple=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`muted`:e.muted=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`ref`:break;case`autoFocus`:break;case`xlinkHref`:if(r==null||typeof r==`function`||typeof r==`boolean`||typeof r==`symbol`){e.removeAttribute(`xlink:href`);break}n=gn(r),e.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,n);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`credentialless`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:r&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,``):e.removeAttribute(n);break;case`capture`:case`download`:!0===r?e.setAttribute(n,``):!1!==r&&r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`cols`:case`rows`:case`size`:case`span`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`&&!isNaN(r)&&1<=r?e.setAttribute(n,r):e.removeAttribute(n);break;case`rowSpan`:case`start`:r==null||typeof r==`function`||typeof r==`symbol`||isNaN(r)?e.removeAttribute(n):e.setAttribute(n,r);break;case`popover`:$(`beforetoggle`,e),$(`toggle`,e),Kt(e,`popover`,r);break;case`xlinkActuate`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:actuate`,r);break;case`xlinkArcrole`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:arcrole`,r);break;case`xlinkRole`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:role`,r);break;case`xlinkShow`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:show`,r);break;case`xlinkTitle`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:title`,r);break;case`xlinkType`:Jt(e,`http://www.w3.org/1999/xlink`,`xlink:type`,r);break;case`xmlBase`:Jt(e,`http://www.w3.org/XML/1998/namespace`,`xml:base`,r);break;case`xmlLang`:Jt(e,`http://www.w3.org/XML/1998/namespace`,`xml:lang`,r);break;case`xmlSpace`:Jt(e,`http://www.w3.org/XML/1998/namespace`,`xml:space`,r);break;case`is`:Kt(e,`is`,r);break;case`innerText`:case`textContent`:return;default:if(!(2<n.length)||n[0]!==`o`&&n[0]!==`O`||n[1]!==`n`&&n[1]!==`N`)n=mn.get(n)||n,Kt(e,n,r);else return}B=!0}function tp(e,t,n,r,i,a){switch(n){case`style`:fn(e,r,a);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(o(61));if(n=r.__html,n!=null){if(i.children!=null)throw Error(o(60));a?.__html!==n&&(e.innerHTML=n)}}break;case`children`:if(typeof r==`string`)ln(e,r);else if(typeof r==`number`||typeof r==`bigint`)ln(e,``+r);else return;break;case`onScroll`:r!=null&&$(`scroll`,e);return;case`onScrollEnd`:r!=null&&$(`scrollend`,e);return;case`onClick`:r!=null&&(e.onclick=_n);return;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`innerHTML`:case`ref`:return;case`innerText`:case`textContent`:return;default:if(!Rt.hasOwnProperty(n))a:{if(n[0]===`o`&&n[1]===`n`&&(i=n.endsWith(`Capture`),a=n.slice(2,i?n.length-7:void 0),t=e[St]||null,t=t==null?null:t[n],typeof t==`function`&&e.removeEventListener(a,t,i),typeof r==`function`)){typeof t!=`function`&&t!==null&&(n in e?e[n]=null:e.hasAttribute(n)&&e.removeAttribute(n)),e.addEventListener(a,r,i);break a}B=!0,n in e?e[n]=r:!0===r?e.setAttribute(n,``):Kt(e,n,r)}return}B=!0}function np(e,t,n){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`img`:$(`error`,e),$(`load`,e);var r=!1,i=!1,a;for(a in n)if(n.hasOwnProperty(a)){var s=n[a];if(s!=null)switch(a){case`src`:r=!0;break;case`srcSet`:i=!0;break;case`children`:case`dangerouslySetInnerHTML`:throw Error(o(137,t));default:ep(e,t,a,s,n,null)}}i&&ep(e,t,`srcSet`,n.srcSet,n,null),r&&ep(e,t,`src`,n.src,n,null);return;case`input`:$(`invalid`,e);var c=a=s=i=null,l=null,u=null;for(r in n)if(n.hasOwnProperty(r)){var d=n[r];if(d!=null)switch(r){case`name`:i=d;break;case`type`:s=d;break;case`checked`:l=d;break;case`defaultChecked`:u=d;break;case`value`:a=d;break;case`defaultValue`:c=d;break;case`children`:case`dangerouslySetInnerHTML`:if(d!=null)throw Error(o(137,t));break;default:ep(e,t,r,d,n,null)}}rn(e,a,c,l,u,s,i,!1);return;case`select`:for(i in $(`invalid`,e),r=s=a=null,n)if(n.hasOwnProperty(i)&&(c=n[i],c!=null))switch(i){case`value`:a=c;break;case`defaultValue`:s=c;break;case`multiple`:r=c;default:ep(e,t,i,c,n,null)}t=a,n=s,e.multiple=!!r,t==null?n!=null&&on(e,!!r,n,!0):on(e,!!r,t,!1);return;case`textarea`:for(s in $(`invalid`,e),a=i=r=null,n)if(n.hasOwnProperty(s)&&(c=n[s],c!=null))switch(s){case`value`:r=c;break;case`defaultValue`:i=c;break;case`children`:a=c;break;case`dangerouslySetInnerHTML`:if(c!=null)throw Error(o(91));break;default:ep(e,t,s,c,n,null)}cn(e,r,i,a);return;case`option`:for(l in n)if(n.hasOwnProperty(l)&&(r=n[l],r!=null))switch(l){case`selected`:e.selected=r&&typeof r!=`function`&&typeof r!=`symbol`;break;default:ep(e,t,l,r,n,null)}return;case`dialog`:$(`beforetoggle`,e),$(`toggle`,e),$(`cancel`,e),$(`close`,e);break;case`iframe`:case`object`:$(`load`,e);break;case`video`:case`audio`:for(r=0;r<Rf.length;r++)$(Rf[r],e);break;case`image`:$(`error`,e),$(`load`,e);break;case`details`:$(`toggle`,e);break;case`embed`:case`source`:case`link`:$(`error`,e),$(`load`,e);case`area`:case`base`:case`br`:case`col`:case`hr`:case`keygen`:case`meta`:case`param`:case`track`:case`wbr`:case`menuitem`:for(u in n)if(n.hasOwnProperty(u)&&(r=n[u],r!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:throw Error(o(137,t));default:ep(e,t,u,r,n,null)}return;default:if(pn(t)){for(d in n)n.hasOwnProperty(d)&&(r=n[d],r!==void 0&&tp(e,t,d,r,n,void 0));return}}for(c in n)n.hasOwnProperty(c)&&(r=n[c],r!=null&&ep(e,t,c,r,n,null))}var rp={};function ip(e,t,n,r){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`input`:var i=null,a=null,s=null,c=null,l=null,u=null,d=null;for(m in n){var f=n[m];if(n.hasOwnProperty(m)&&f!=null)switch(m){case`checked`:break;case`value`:break;case`defaultValue`:l=f;default:r.hasOwnProperty(m)||ep(e,t,m,null,r,f)}}for(var p in r){var m=r[p];if(f=n[p],r.hasOwnProperty(p)&&(m!=null||f!=null))switch(p){case`type`:m!==f&&(B=!0),a=m;break;case`name`:m!==f&&(B=!0),i=m;break;case`checked`:m!==f&&(B=!0),u=m;break;case`defaultChecked`:m!==f&&(B=!0),d=m;break;case`value`:m!==f&&(B=!0),s=m;break;case`defaultValue`:m!==f&&(B=!0),c=m;break;case`children`:case`dangerouslySetInnerHTML`:if(m!=null)throw Error(o(137,t));break;default:m!==f&&ep(e,t,p,m,r,f)}}nn(e,s,c,l,u,d,a,i);return;case`select`:for(a in m=s=c=p=null,n)if(l=n[a],n.hasOwnProperty(a)&&l!=null)switch(a){case`value`:break;case`multiple`:m=l;default:r.hasOwnProperty(a)||ep(e,t,a,null,r,l)}for(i in r)if(a=r[i],l=n[i],r.hasOwnProperty(i)&&(a!=null||l!=null))switch(i){case`value`:a!==l&&(B=!0),p=a;break;case`defaultValue`:a!==l&&(B=!0),c=a;break;case`multiple`:a!==l&&(B=!0),s=a;default:a!==l&&ep(e,t,i,a,r,l)}t=c,n=s,r=m,p==null?!!r!=!!n&&(t==null?on(e,!!n,n?[]:``,!1):on(e,!!n,t,!0)):on(e,!!n,p,!1);return;case`textarea`:for(c in m=p=null,n)if(i=n[c],n.hasOwnProperty(c)&&i!=null&&!r.hasOwnProperty(c))switch(c){case`value`:break;case`children`:break;default:ep(e,t,c,null,r,i)}for(s in r)if(i=r[s],a=n[s],r.hasOwnProperty(s)&&(i!=null||a!=null))switch(s){case`value`:i!==a&&(B=!0),p=i;break;case`defaultValue`:i!==a&&(B=!0),m=i;break;case`children`:break;case`dangerouslySetInnerHTML`:if(i!=null)throw Error(o(91));break;default:i!==a&&ep(e,t,s,i,r,a)}sn(e,p,m);return;case`option`:for(var h in n)if(p=n[h],n.hasOwnProperty(h)&&p!=null&&!r.hasOwnProperty(h))switch(h){case`selected`:e.selected=!1;break;default:ep(e,t,h,null,r,p)}for(l in r)if(p=r[l],m=n[l],r.hasOwnProperty(l)&&p!==m&&(p!=null||m!=null))switch(l){case`selected`:p!==m&&(B=!0),e.selected=p&&typeof p!=`function`&&typeof p!=`symbol`;break;default:ep(e,t,l,p,r,m)}return;case`img`:case`link`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:case`menuitem`:for(var g in n)p=n[g],n.hasOwnProperty(g)&&p!=null&&!r.hasOwnProperty(g)&&ep(e,t,g,null,r,p);for(u in r)if(p=r[u],m=n[u],r.hasOwnProperty(u)&&p!==m&&(p!=null||m!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:if(p!=null)throw Error(o(137,t));break;default:ep(e,t,u,p,r,m)}return;default:if(pn(t)){for(var _ in n)p=n[_],n.hasOwnProperty(_)&&p!==void 0&&!r.hasOwnProperty(_)&&tp(e,t,_,void 0,r,p);for(d in r)p=r[d],m=n[d],!r.hasOwnProperty(d)||p===m||p===void 0&&m===void 0||tp(e,t,d,p,r,m);return}}for(var v in n)p=n[v],n.hasOwnProperty(v)&&p!=null&&!r.hasOwnProperty(v)&&ep(e,t,v,null,r,p);for(f in r)p=r[f],m=n[f],!r.hasOwnProperty(f)||p===m||p==null&&m==null||ep(e,t,f,p,r,m)}function ap(e){switch(e){case`css`:case`script`:case`font`:case`img`:case`image`:case`input`:case`link`:return!0;default:return!1}}function op(){if(typeof performance.getEntriesByType==`function`){for(var e=0,t=0,n=performance.getEntriesByType(`resource`),r=0;r<n.length;r++){var i=n[r],a=i.transferSize,o=i.initiatorType,s=i.duration;if(a&&s&&ap(o)){for(o=0,s=i.responseEnd,r+=1;r<n.length;r++){var c=n[r],l=c.startTime;if(l>s)break;var u=c.transferSize,d=c.initiatorType;u&&ap(d)&&(c=c.responseEnd,o+=u*(c<s?1:(s-l)/(c-l)))}if(--r,t+=8*(a+o)/(i.duration/1e3),e++,10<e)break}}if(0<e)return t/e/1e6}return navigator.connection&&(e=navigator.connection.downlink,typeof e==`number`)?e:5}var sp=null,cp=null;function lp(e){return e.nodeType===9?e:e.ownerDocument}function up(e){switch(e){case`http://www.w3.org/2000/svg`:return 1;case`http://www.w3.org/1998/Math/MathML`:return 2;default:return 0}}function dp(e,t){if(e===0)switch(t){case`svg`:return 1;case`math`:return 2;default:return 0}return e===1&&t===`foreignObject`?0:e}function fp(e,t,n,r){return n=lp(n).createElement(e),n[xt]=r,n[St]=t,np(n,e,t),Ft(n),n}function pp(e,t){return e===`textarea`||e===`noscript`||typeof t.children==`string`||typeof t.children==`number`||typeof t.children==`bigint`||typeof t.dangerouslySetInnerHTML==`object`&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var mp=null;function hp(){var e=window.event;return e&&e.type===`popstate`?e!==mp&&(mp=e,!0):(mp=null,!1)}var gp=typeof setTimeout==`function`?setTimeout:void 0,_p=typeof clearTimeout==`function`?clearTimeout:void 0,vp=typeof Promise==`function`?Promise:void 0,yp=typeof requestAnimationFrame==`function`?requestAnimationFrame:gp,bp=typeof queueMicrotask==`function`?queueMicrotask:vp===void 0?gp:function(e){return vp.resolve(null).then(e).catch(xp)};function xp(e){setTimeout(function(){throw e})}function Sp(e){return e===`head`}function Cp(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8){if(n=i.data,n===`/$`||n===`/&`){if(r===0){e.removeChild(i),Hh(t);return}r--}else if(n===`$`||n===`$?`||n===`$~`||n===`$!`||n===`&`)r++;else if(n===`html`)_m(e.ownerDocument.documentElement);else if(n===`head`){n=e.ownerDocument.head,_m(n);for(var a=n.firstChild;a;){var o=a.nextSibling,s=a.nodeName;a[Ot]||s===`SCRIPT`||s===`STYLE`||s===`LINK`&&a.rel.toLowerCase()===`stylesheet`||n.removeChild(a),a=o}}else n===`body`&&_m(e.ownerDocument.body)}n=i}while(n);Hh(t)}function wp(e,t){var n=e;e=0;do{var r=n.nextSibling;if(n.nodeType===1?t?(n._stashedDisplay=n.style.display,n.style.display=`none`):(n.style.display=n._stashedDisplay||``,n.getAttribute(`style`)===``&&n.removeAttribute(`style`)):n.nodeType===3&&(t?(n._stashedText=n.nodeValue,n.nodeValue=``):n.nodeValue=n._stashedText||``),r&&r.nodeType===8){if(n=r.data,n===`/$`){if(e===0)break;e--}else n!==`$`&&n!==`$?`&&n!==`$~`&&n!==`$!`||e++}n=r}while(n)}function Tp(e,t,n){if(t=CSS.escape(t)===t?t:`r-`+btoa(t).replace(/=/g,``),e.style.viewTransitionName=t,n!=null&&(e.style.viewTransitionClass=n),n=getComputedStyle(e),n.display===`inline`){if(t=e.getClientRects(),t.length===1)var r=1;else for(var i=r=0;i<t.length;i++){var a=t[i];0<a.width&&0<a.height&&r++}r===1&&(e=e.style,e.display=t.length===1?`inline-block`:`block`,e.marginTop=`-`+n.paddingTop,e.marginBottom=`-`+n.paddingBottom)}}function Ep(e,t){e=e.style,t=t.style;var n=t==null?null:t.hasOwnProperty(`viewTransitionName`)?t.viewTransitionName:t.hasOwnProperty(`view-transition-name`)?t[`view-transition-name`]:null;e.viewTransitionName=n==null||typeof n==`boolean`?``:(``+n).trim(),n=t==null?null:t.hasOwnProperty(`viewTransitionClass`)?t.viewTransitionClass:t.hasOwnProperty(`view-transition-class`)?t[`view-transition-class`]:null,e.viewTransitionClass=n==null||typeof n==`boolean`?``:(``+n).trim(),e.display===`inline-block`&&(t==null?e.display=e.margin=``:(n=t.display,e.display=n==null||typeof n==`boolean`?``:n,n=t.margin,n==null?(n=t.hasOwnProperty(`marginTop`)?t.marginTop:t[`margin-top`],e.marginTop=n==null||typeof n==`boolean`?``:n,t=t.hasOwnProperty(`marginBottom`)?t.marginBottom:t[`margin-bottom`],e.marginBottom=t==null||typeof t==`boolean`?``:t):e.margin=n))}function Dp(e,t,n){return n=n.ownerDocument.defaultView,{rect:e,abs:t.position===`absolute`||t.position===`fixed`,clip:t.clipPath!==`none`||t.overflow!==`visible`||t.filter!==`none`||t.mask!==`none`||t.mask!==`none`||t.borderRadius!==`0px`,view:0<=e.bottom&&0<=e.right&&e.top<=n.innerHeight&&e.left<=n.innerWidth}}function Op(e){return Dp(e.getBoundingClientRect(),getComputedStyle(e),e)}function kp(e){var t=e.getBoundingClientRect();t=new DOMRect(t.x+2e4,t.y+2e4,t.width,t.height);var n=getComputedStyle(e);return Dp(t,n,e)}function Ap(e){return e.documentElement.clientHeight}function jp(e){this.addEventListener(`load`,e),this.addEventListener(`error`,e)}function Mp(e,t,n,r,i,a,o,s,c){var l=t.nodeType===9?t:t.ownerDocument;try{var u=l.startViewTransition({update:function(){var t=l.defaultView,n=t.navigation&&t.navigation.transition,o=l.fonts.status;r();var s=[];if(o===`loaded`&&(Ap(l),l.fonts.status===`loading`&&s.push(l.fonts.ready)),o=s.length,e!==null)for(var c=e.suspenseyImages,u=0,d=0;d<c.length;d++){var f=c[d];if(!f.complete){var p=f.getBoundingClientRect();if(0<p.bottom&&0<p.right&&p.top<t.innerHeight&&p.left<t.innerWidth){if(u+=Xm(f),u>$m){s.length=o;break}f=new Promise(jp.bind(f)),s.push(f)}}}if(0<s.length)return t=Promise.race([Promise.all(s),new Promise(function(e){return setTimeout(e,500)})]).then(i,i),(n?Promise.allSettled([n.finished,t]):t).then(a,a);if(i(),n)return n.finished.then(a,a);a()},types:n});l.__reactViewTransition=u;var d=[];return u.ready.then(function(){for(var e=l.documentElement.getAnimations({subtree:!0}),t=0;t<e.length;t++){var n=e[t],r=n.effect,i=r.pseudoElement;if(i!=null&&i.startsWith(`::view-transition`)){d.push(n),n=r.getKeyframes();for(var a=i=void 0,s=!0,c=0;c<n.length;c++){var u=n[c],f=u.width;if(i===void 0)i=f;else if(i!==f){s=!1;break}if(f=u.height,a===void 0)a=f;else if(a!==f){s=!1;break}delete u.width,delete u.height,u.transform===`none`&&delete u.transform}s&&i!==void 0&&a!==void 0&&(r.setKeyframes(n),s=getComputedStyle(r.target,r.pseudoElement),s.width!==i||s.height!==a)&&(s=n[0],s.width=i,s.height=a,s=n[n.length-1],s.width=i,s.height=a,r.setKeyframes(n))}}o()},function(e){l.__reactViewTransition===u&&(l.__reactViewTransition=null);try{if(typeof e==`object`&&e)switch(e.name){case`InvalidStateError`:(e.message===`View transition was skipped because document visibility state is hidden.`||e.message===`Skipping view transition because document visibility state has become hidden.`||e.message===`Skipping view transition because viewport size changed.`||e.message===`Transition was aborted because of invalid state`)&&(e=null)}e!==null&&c(e)}finally{r(),i(),o()}}),u.finished.finally(function(){for(var e=0;e<d.length;e++)d[e].cancel();l.__reactViewTransition===u&&(l.__reactViewTransition=null),s()}),u}catch{return r(),i(),o(),null}}function Np(e,t){this._scope=document.documentElement,this._selector=`::view-transition-`+e+`(`+t+`)`}Np.prototype.animate=function(e,t){return t=typeof t==`number`?{duration:t}:E({},t),t.pseudoElement=this._selector,this._scope.animate(e,t)},Np.prototype.getAnimations=function(){for(var e=this._scope,t=this._selector,n=e.getAnimations({subtree:!0}),r=[],i=0;i<n.length;i++){var a=n[i].effect;a!==null&&a.target===e&&a.pseudoElement===t&&r.push(n[i])}return r},Np.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function Pp(e){return{name:e,group:new Np(`group`,e),imagePair:new Np(`image-pair`,e),old:new Np(`old`,e),new:new Np(`new`,e)}}function Fp(e){this._fragmentFiber=e,this._observers=this._eventListeners=null}Fp.prototype.addEventListener=function(e,t,n){var r=null,i=null;if(!(n!=null&&typeof n!=`boolean`&&(r=n.signal||null,r!==null&&r.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var a=this._eventListeners;if(Bp(a,e,t,n)===-1){var o=this,s=t;n!=null&&typeof n!=`boolean`&&!0===n.once&&(s=function(r){o.removeEventListener(e,t,n),typeof t==`function`?t.call(this,r):t.handleEvent(r)}),r!==null&&(i=o.removeEventListener.bind(o,e,t,n),r.addEventListener(`abort`,i,{once:!0}),i=r.removeEventListener.bind(r,`abort`,i)),r=Rp(n),a.push({type:e,listener:t,optionsOrUseCapture:n,attachedListener:s,cleanup:i}),m(this._fragmentFiber.child,!1,Ip,e,s,r)}this._eventListeners=a}};function Ip(e,t,n,r){return y(e).addEventListener(t,n,r),!1}Fp.prototype.removeEventListener=function(e,t,n){var r=this._eventListeners;if(r!==null&&(t=Bp(r,e,t,n),t!==-1)){var i=r[t];n=i.attachedListener;var a=i.cleanup;i=Rp(i.optionsOrUseCapture),m(this._fragmentFiber.child,!1,Lp,e,n,i),r.splice(t,1),a!==null&&a()}};function Lp(e,t,n,r){return y(e).removeEventListener(t,n,r),!1}function Rp(e){return e!=null&&typeof e!=`boolean`&&(!0===e.once||e.signal instanceof AbortSignal)?{capture:e.capture,passive:e.passive}:e}function zp(e){return e==null?`c=0`:typeof e==`boolean`?`c=`+(e?`1`:`0`):`c=`+(e.capture?`1`:`0`)}function Bp(e,t,n,r){if(e.length===0)return-1;r=zp(r);for(var i=0;i<e.length;i++){var a=e[i];if(a.type===t&&a.listener===n&&zp(a.optionsOrUseCapture)===r)return i}return-1}Fp.prototype.dispatchEvent=function(e){var t=h(this._fragmentFiber);if(t===null)return!0;t=y(t);var n=this._eventListeners;if(n!==null&&0<n.length||!e.bubbles){var r=t.nodeType===9?t.createComment(``):document.createTextNode(``);if(n)for(var i=0;i<n.length;i++){var a=n[i];r.addEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture))}if(t.appendChild(r),e=r.dispatchEvent(e),n)for(i=0;i<n.length;i++)a=n[i],r.removeEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture));return t.removeChild(r),e}return t.dispatchEvent(e)},Fp.prototype.focus=function(e){m(this._fragmentFiber.child,!0,Vp,e,void 0,void 0)};function Vp(e,t){return e.tag!==6&&(e=y(e),pm(e,t))}Fp.prototype.focusLast=function(e){var t=[];m(this._fragmentFiber.child,!0,Hp,t,void 0,void 0);for(var n=t.length-1;0<=n&&!Vp(t[n],e);n--);};function Hp(e,t){return t.push(e),!1}Fp.prototype.blur=function(){var e=h(this._fragmentFiber);e!==null&&(e=y(e),e=lp(e).activeElement,e!==null&&m(this._fragmentFiber.child,!1,Up,e,void 0,void 0))};function Up(e,t){return e.tag!==6&&(e=y(e),e===t||e.contains(t)?(t.blur(),!0):!1)}Fp.prototype.observeUsing=function(e){this._observers===null&&(this._observers=new Set),this._observers.add(e),m(this._fragmentFiber.child,!1,Wp,e,void 0,void 0)};function Wp(e,t){return e.tag===6||(e=y(e),t.observe(e)),!1}Fp.prototype.unobserveUsing=function(e){var t=this._observers;if(t!==null&&t.has(e)){t.delete(e),m(this._fragmentFiber.child,!1,Gp,e,void 0,void 0);for(var n=t=0;n<Kp.length;n++){var r=Kp[n];r.fragmentInstance===this&&r.observer===e?e.unobserve(r.instance):Kp[t++]=r}Kp.length=t}};function Gp(e,t){return e.tag===6||(e=y(e),t.unobserve(e)),!1}var Kp=[],qp=!1;function Jp(e,t,n){Kp.push({fragmentInstance:e,observer:t,instance:n}),qp||(qp=!0,mm(function(){qp=!1;var e=Kp;Kp=[];for(var t=0;t<e.length;t++){var n=e[t];n.observer.unobserve(n.instance)}}))}Fp.prototype.getClientRects=function(){var e=[];return m(this._fragmentFiber.child,!1,Yp,e,void 0,void 0),e};function Yp(e,t){if(e.tag===6){e=e.stateNode;var n=e.ownerDocument.createRange();n.selectNodeContents(e),t.push.apply(t,n.getClientRects())}else e=y(e),t.push.apply(t,e.getClientRects());return!1}Fp.prototype.getRootNode=function(e){var t=h(this._fragmentFiber);return t===null?this:y(t).getRootNode(e)},Fp.prototype.compareDocumentPosition=function(e){var t=h(this._fragmentFiber);if(t===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];m(this._fragmentFiber.child,!1,Hp,n,void 0,void 0);var r=y(t);if(n.length===0){if(n=r,g(this._fragmentFiber)){a:{for(t=this._fragmentFiber.return;t!==null;){if(t.tag===4){t=t.stateNode.containerInfo;break a}if(t.tag===3||t.tag===5||t.tag===27)break;t=t.return}t=null}t!=null&&(n=t)}t=this._fragmentFiber;var i=r=n.compareDocumentPosition(e);return n===e?i=Node.DOCUMENT_POSITION_CONTAINS:r&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=_(t)[1],n===null?i=Node.DOCUMENT_POSITION_PRECEDING:(e=y(n).compareDocumentPosition(e),i=e===0||e&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),i|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}t=y(n[0]),i=y(n[n.length-1]);var a=g(this._fragmentFiber)?t.parentElement:r;if(a==null)return Node.DOCUMENT_POSITION_DISCONNECTED;r=a.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_CONTAINED_BY,a=a.compareDocumentPosition(i)&Node.DOCUMENT_POSITION_CONTAINED_BY;var o=t.compareDocumentPosition(e),s=i.compareDocumentPosition(e),c=o&Node.DOCUMENT_POSITION_CONTAINED_BY||s&Node.DOCUMENT_POSITION_CONTAINED_BY;return s=r&&a&&o&Node.DOCUMENT_POSITION_FOLLOWING&&s&Node.DOCUMENT_POSITION_PRECEDING,t=r&&t===e||a&&i===e||c||s?Node.DOCUMENT_POSITION_CONTAINED_BY:!r&&t===e||!a&&i===e?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:o,t&Node.DOCUMENT_POSITION_DISCONNECTED||t&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Xp(t,this._fragmentFiber,n[0],n[n.length-1],e)?t:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Xp(e,t,n,r,i){var a=jt(i);if(e&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!a)a:{for(;a!==null;){if(a.tag===7&&(a===t||a.alternate===t)){n=!0;break a}a=a.return}n=!1}return n}if(e&Node.DOCUMENT_POSITION_CONTAINS){if(a===null)return a=i.ownerDocument,i===a||i===a.documentElement||i===a.body;a:{for(a=t,t=h(t);a!==null;){if(!(a.tag!==5&&a.tag!==3&&a.tag!==27||a!==t&&a.alternate!==t)){a=!0;break a}a=a.return}a=!1}return a}return e&Node.DOCUMENT_POSITION_PRECEDING?((t=!!a)&&!(t=a===n)&&(t=T(n,a,w),t===null?t=!1:(m(t,!0,S,a,n),a=b,b=null,t=a!==null)),t):e&Node.DOCUMENT_POSITION_FOLLOWING?((t=!!a)&&!(t=a===r)&&(t=T(r,a,w),t===null?t=!1:(m(t,!0,C,a,r),a=b,x=b=null,t=a!==null)),t):!1}function Zp(e,t){var n=e.ownerDocument.createRange();n.selectNodeContents(e),e=n.getBoundingClientRect(),window.scrollTo(window.scrollX+e.left,t?window.scrollY+e.top:window.scrollY+e.bottom-window.innerHeight)}Fp.prototype.scrollIntoView=function(e){if(typeof e==`object`)throw Error(o(566));var t=[];m(this._fragmentFiber.child,!1,Hp,t,void 0,void 0);var n=!1!==e;if(t.length===0){var r=_(this._fragmentFiber);if(r=n?r[1]||r[0]||h(this._fragmentFiber):r[0]||r[1],r===null)return;if(r.tag===6){e=y(r),Zp(e,n);return}if(r=y(r),r.nodeType!==9){if(r.nodeType===11){n=`host`in r?r.host:null,n!==null&&n.scrollIntoView(e);return}r.scrollIntoView(e)}}for(r=n?t.length-1:0;r!==(n?-1:t.length);){var i=t[r];i.tag===6?(i=y(i),Zp(i,n)):y(i).scrollIntoView(e),r+=n?-1:1}};function Qp(e,t){return e=y(e),$p(e,t),!1}function $p(e,t){e.reactFragments??=new Set,e.reactFragments.add(t)}function em(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.addEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){for(var r=0,i=0;i<Kp.length;i++){var a=Kp[i];(a.fragmentInstance!==t||a.observer!==n||a.instance!==e)&&(Kp[r++]=a)}Kp.length=r,n.observe(e)}),$p(e,t))}function tm(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.removeEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){typeof n.rootMargin==`string`?Jp(t,n,e):n.unobserve(e)}),e.reactFragments!=null&&e.reactFragments.delete(t))}function nm(e){var t=e.firstChild;for(t&&t.nodeType===10&&(t=t.nextSibling);t;){var n=t;switch(t=t.nextSibling,n.nodeName){case`HTML`:case`HEAD`:case`BODY`:nm(n),At(n);continue;case`SCRIPT`:case`STYLE`:continue;case`LINK`:if(n.rel.toLowerCase()===`stylesheet`)continue}e.removeChild(n)}}function rm(e,t,n,r){for(;e.nodeType===1;){var i=n;if(e.nodeName.toLowerCase()!==t.toLowerCase()){if(!r&&(e.nodeName!==`INPUT`||e.type!==`hidden`))break}else if(!r){if(t===`input`&&e.type===`hidden`){var a=i.name==null?null:``+i.name;if(i.type===`hidden`&&e.getAttribute(`name`)===a)return e}else return e}else if(!e[Ot])switch(t){case`meta`:if(!e.hasAttribute(`itemprop`))break;return e;case`link`:if(a=e.getAttribute(`rel`),a===`stylesheet`&&e.hasAttribute(`data-precedence`)||a!==i.rel||e.getAttribute(`href`)!==(i.href==null||i.href===``?null:i.href)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin)||e.getAttribute(`title`)!==(i.title==null?null:i.title))break;return e;case`style`:if(e.hasAttribute(`data-precedence`))break;return e;case`script`:if(a=e.getAttribute(`src`),(a!==(i.src==null?null:i.src)||e.getAttribute(`type`)!==(i.type==null?null:i.type)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin))&&a&&e.hasAttribute(`async`)&&!e.hasAttribute(`itemprop`))break;return e;default:return e}if(e=lm(e.nextSibling),e===null)break}return null}function im(e,t,n){if(t===``)return null;for(;e.nodeType!==3;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!n||(e=lm(e.nextSibling),e===null))return null;return e}function am(e,t){for(;e.nodeType!==8;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!t||(e=lm(e.nextSibling),e===null))return null;return e}function om(e){return e.data===`$?`||e.data===`$~`}function sm(e){return e.data===`$!`||e.data===`$?`&&e.ownerDocument.readyState!==`loading`}function cm(e,t){var n=e.ownerDocument;if(e.data===`$~`)e._reactRetry=t;else if(e.data!==`$?`||n.readyState!==`loading`)t();else{var r=function(){t(),n.removeEventListener(`DOMContentLoaded`,r)};n.addEventListener(`DOMContentLoaded`,r),e._reactRetry=r}}function lm(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t===`$`||t===`$!`||t===`$?`||t===`$~`||t===`&`||t===`F!`||t===`F`)break;if(t===`/$`||t===`/&`)return null}}return e}var um=null;function dm(e){e=e.nextSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`/$`||n===`/&`){if(t===0)return lm(e.nextSibling);t--}else n!==`$`&&n!==`$!`&&n!==`$?`&&n!==`$~`&&n!==`&`||t++}e=e.nextSibling}return null}function fm(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`$`||n===`$!`||n===`$?`||n===`$~`||n===`&`){if(t===0)return e;t--}else n!==`/$`&&n!==`/&`||t++}e=e.previousSibling}return null}function pm(e,t){function n(){r=!0}if(e.ownerDocument.activeElement===e)return!0;var r=!1;try{e.ownerDocument.addEventListener(`focus`,n,!0),(e.focus||HTMLElement.prototype.focus).call(e,t)}finally{e.ownerDocument.removeEventListener(`focus`,n,!0)}return r}function mm(e){yp(function(){yp(function(t){return e(t)})})}function hm(e,t,n){switch(t=lp(n),e){case`html`:if(e=t.documentElement,!e)throw Error(o(452));return e;case`head`:if(e=t.head,!e)throw Error(o(453));return e;case`body`:if(e=t.body,!e)throw Error(o(454));return e;default:throw Error(o(451))}}function gm(e,t,n){for(var r in n){var i=n[r];n.hasOwnProperty(r)&&i!=null&&ep(e,t,r,null,rp,i)}n.dangerouslySetInnerHTML!=null&&(e.textContent=``),e.onclick===_n&&(e.onclick=null),At(e)}function _m(e){for(var t=e.attributes;t.length;)e.removeAttributeNode(t[0]);At(e)}var vm=new Map,ym=new Set;function bm(e){if(typeof e.getRootNode==`function`){var t=e.getRootNode();if(t.nodeType===9||t.nodeType===11)return t}return e.nodeType===9?e:e.ownerDocument}var xm=P.d;P.d={f:Sm,r:Cm,D:Em,C:Dm,L:Om,m:km,X:jm,S:Am,M:Mm};function Sm(){var e=xm.f(),t=Ld();return e||t}function Cm(e){var t=Mt(e);t!==null&&t.tag===5&&t.type===`form`?ec(t):xm.r(e)}var wm=typeof document>`u`?null:document;function Tm(e,t,n){var r=wm;if(r&&typeof t==`string`&&t){var i=tn(t);i=`link[rel="`+e+`"][href="`+i+`"]`,typeof n==`string`&&(i+=`[crossorigin="`+n+`"]`),ym.has(i)||(ym.add(i),e={rel:e,crossOrigin:n,href:t},r.querySelector(i)===null&&(t=r.createElement(`link`),np(t,`link`,e),Ft(t),r.head.appendChild(t)))}}function Em(e){xm.D(e),Tm(`dns-prefetch`,e,null)}function Dm(e,t){xm.C(e,t),Tm(`preconnect`,e,t)}function Om(e,t,n){xm.L(e,t,n);var r=wm;if(r&&e&&t){var i=`link[rel="preload"][as="`+tn(t)+`"]`;t===`image`&&n&&n.imageSrcSet?(i+=`[imagesrcset="`+tn(n.imageSrcSet)+`"]`,typeof n.imageSizes==`string`&&(i+=`[imagesizes="`+tn(n.imageSizes)+`"]`)):i+=`[href="`+tn(e)+`"]`;var a=i;switch(t){case`style`:a=Pm(e);break;case`script`:a=Rm(e)}if(!(vm.has(a)||(e=E({rel:`preload`,href:t===`image`&&n&&n.imageSrcSet?void 0:e,as:t},n),vm.set(a,e),r.querySelector(i)!==null||t===`style`&&r.querySelector(Fm(a))||t===`script`&&r.querySelector(zm(a))))){var o=r.createElement(`link`);np(o,`link`,e),t===`style`&&(o[kt]=!0,o.onload=o.onerror=function(){It(o)}),Ft(o),r.head.appendChild(o)}}}function km(e,t){xm.m(e,t);var n=wm;if(n&&e){var r=t&&typeof t.as==`string`?t.as:`script`,i=`link[rel="modulepreload"][as="`+tn(r)+`"][href="`+tn(e)+`"]`,a=i;switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:a=Rm(e)}if(!vm.has(a)&&(e=E({rel:`modulepreload`,href:e},t),vm.set(a,e),n.querySelector(i)===null)){switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:if(n.querySelector(zm(a)))return}r=n.createElement(`link`),np(r,`link`,e),Ft(r),n.head.appendChild(r)}}}function Am(e,t,n){xm.S(e,t,n);var r=wm;if(r&&e){var i=Pt(r).hoistableStyles,a=Pm(e);t||=`default`;var o=i.get(a);if(!o){var s={loading:0,preload:null};if(o=r.querySelector(Fm(a)))s.loading=5;else{e=E({rel:`stylesheet`,href:e,"data-precedence":t},n),(n=vm.get(a))&&Hm(e,n);var c=o=r.createElement(`link`);Ft(c),np(c,`link`,e),c._p=new Promise(function(e,t){c.onload=e,c.onerror=t}),c.addEventListener(`load`,function(){s.loading|=1}),c.addEventListener(`error`,function(){s.loading|=2}),s.loading|=4,Vm(o,t,r)}o={type:`stylesheet`,instance:o,count:1,state:s},i.set(a,o)}}}function jm(e,t){xm.X(e,t);var n=wm;if(n&&e){var r=Pt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=E({src:e,async:!0},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),Ft(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Mm(e,t){xm.M(e,t);var n=wm;if(n&&e){var r=Pt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=E({src:e,async:!0,type:`module`},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),Ft(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Nm(e,t,n,r){var i=(i=Ce.current)?bm(i):null;if(!i)throw Error(o(446));switch(e){case`meta`:case`title`:return null;case`style`:return typeof n.precedence==`string`&&typeof n.href==`string`?(n=Pm(n.href),t=Pt(i).hoistableStyles,r=t.get(n),r||(r={type:`style`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};case`link`:if(n.rel===`stylesheet`&&typeof n.href==`string`&&typeof n.precedence==`string`){e=Pm(n.href);var a=Pt(i).hoistableStyles,s=a.get(e);if(s||(i=i.ownerDocument||i,s={type:`stylesheet`,instance:null,count:0,state:{loading:0,preload:null}},a.set(e,s),(a=i.querySelector(Fm(e)))?a._p||(s.instance=a,s.state.loading=5):(a=vm.get(e),a||(a={rel:`preload`,as:`style`,href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},vm.set(e,a)),Lm(i,e,a,s.state))),t&&r===null)throw Error(o(528,``));return s}if(t&&r!==null)throw Error(o(529,``));return null;case`script`:return t=n.async,n=n.src,typeof n==`string`&&t&&typeof t!=`function`&&typeof t!=`symbol`?(n=Rm(n),t=Pt(i).hoistableScripts,r=t.get(n),r||(r={type:`script`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};default:throw Error(o(444,e))}}function Pm(e){return`href="`+tn(e)+`"`}function Fm(e){return`link[rel="stylesheet"][`+e+`]`}function Im(e){return E({},e,{"data-precedence":e.precedence,precedence:null})}function Lm(e,t,n,r){if(t=e.querySelector(`link[rel="preload"][as="style"][`+t+`]`)){if(!0!==t[kt]){r.loading=1;return}}else t=e.createElement(`link`),t[kt]=!0,t.onload=t.onerror=It.bind(null,t),np(t,`link`,n),Ft(t),e.head.appendChild(t);r.preload=t,t.addEventListener(`load`,function(){return r.loading|=1}),t.addEventListener(`error`,function(){return r.loading|=2})}function Rm(e){return`[src="`+tn(e)+`"]`}function zm(e){return`script[async]`+e}function Bm(e,t,n){if(t.count++,t.instance===null)switch(t.type){case`style`:var r=e.querySelector(`style[data-href~="`+tn(n.href)+`"]`);if(r)return t.instance=r,Ft(r),r;var i=E({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return r=(e.ownerDocument||e).createElement(`style`),Ft(r),np(r,`style`,i),Vm(r,n.precedence,e),t.instance=r;case`stylesheet`:i=Pm(n.href);var a=e.querySelector(Fm(i));if(a)return t.state.loading|=4,t.instance=a,Ft(a),a;r=Im(n),(i=vm.get(i))&&Hm(r,i),a=(e.ownerDocument||e).createElement(`link`),Ft(a);var s=a;return s._p=new Promise(function(e,t){s.onload=e,s.onerror=t}),np(a,`link`,r),t.state.loading|=4,Vm(a,n.precedence,e),t.instance=a;case`script`:return a=Rm(n.src),(i=e.querySelector(zm(a)))?(t.instance=i,Ft(i),i):(r=n,(i=vm.get(a))&&(r=E({},n),Um(r,i)),e=e.ownerDocument||e,i=e.createElement(`script`),Ft(i),np(i,`link`,r),e.head.appendChild(i),t.instance=i);case`void`:return null;default:throw Error(o(443,t.type))}else t.type===`stylesheet`&&!(t.state.loading&4)&&(r=t.instance,t.state.loading|=4,Vm(r,n.precedence,e));return t.instance}function Vm(e,t,n){for(var r=n.querySelectorAll(`link[rel="stylesheet"][data-precedence],style[data-precedence]`),i=r.length?r[r.length-1]:null,a=i,o=0;o<r.length;o++){var s=r[o];if(s.dataset.precedence===t)a=s;else if(a!==i)break}a?a.parentNode.insertBefore(e,a.nextSibling):(t=n.nodeType===9?n.head:n,t.insertBefore(e,t.firstChild))}function Hm(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.title??=t.title}function Um(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.integrity??=t.integrity}var Wm=null;function Gm(e,t,n){if(Wm===null){var r=new Map,i=Wm=new Map;i.set(n,r)}else i=Wm,r=i.get(n),r||(r=new Map,i.set(n,r));if(r.has(e))return r;for(r.set(e,null),n=n.getElementsByTagName(e),i=0;i<n.length;i++){var a=n[i];if(!(a[Ot]||a[xt]||e===`link`&&a.getAttribute(`rel`)===`stylesheet`)&&a.namespaceURI!==`http://www.w3.org/2000/svg`){var o=a.getAttribute(t)||``;o=e+o;var s=r.get(o);s?s.push(a):r.set(o,[a])}}return r}function Km(e,t,n){e=e.ownerDocument||e,e.head.insertBefore(n,t===`title`?e.querySelector(`head > title`):null)}function qm(e,t,n){if(n===1||t.itemProp!=null)return!1;switch(e){case`meta`:case`title`:return!0;case`style`:if(typeof t.precedence!=`string`||typeof t.href!=`string`||t.href===``)break;return!0;case`link`:if(typeof t.rel!=`string`||typeof t.href!=`string`||t.href===``||t.onLoad||t.onError)break;switch(t.rel){case`stylesheet`:return e=t.disabled,typeof t.precedence==`string`&&e==null;default:return!0}case`script`:if(t.async&&typeof t.async!=`function`&&typeof t.async!=`symbol`&&!t.onLoad&&!t.onError&&t.src&&typeof t.src==`string`)return!0}return!1}function Jm(e,t){return e===`img`&&t.src!=null&&t.src!==``&&t.onLoad==null&&t.loading!==`lazy`}function Ym(e){return!(e.type===`stylesheet`&&!(e.state.loading&3))}function Xm(e){return(e.width||100)*(e.height||100)*(typeof devicePixelRatio==`number`?devicePixelRatio:1)*.25}function Zm(e,t){typeof t.decode==`function`&&(e.imgCount++,t.complete||(e.imgBytes+=Xm(t),e.suspenseyImages.push(t)),e=rh.bind(e),t.decode().then(e,e))}function Qm(e,t,n,r){if(n.type===`stylesheet`&&(typeof r.media!=`string`||!1!==matchMedia(r.media).matches)&&!(n.state.loading&4)){if(n.instance===null){var i=Pm(r.href),a=t.querySelector(Fm(i));if(a){t=a._p,typeof t==`object`&&t&&typeof t.then==`function`&&(e.count++,e=nh.bind(e),t.then(e,e)),n.state.loading|=4,n.instance=a,Ft(a);return}a=t.ownerDocument||t,r=Im(r),(i=vm.get(i))&&Hm(r,i),a=a.createElement(`link`),Ft(a);var o=a;o._p=new Promise(function(e,t){o.onload=e,o.onerror=t}),np(a,`link`,r),n.instance=a}e.stylesheets===null&&(e.stylesheets=new Map),e.stylesheets.set(n,t),(t=n.state.preload)&&!(n.state.loading&3)&&(e.count++,n=nh.bind(e),t.addEventListener(`load`,n),t.addEventListener(`error`,n))}}var $m=0;function eh(e,t){return e.stylesheets&&e.count===0&&ah(e,e.stylesheets),0<e.count||0<e.imgCount?function(n){var r=setTimeout(function(){if(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}},6e4+t);0<e.imgBytes&&$m===0&&($m=62500*op());var i=setTimeout(function(){if(e.waitingForImages=!1,e.count===0&&(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend)){var t=e.unsuspend;e.unsuspend=null,t()}},(e.imgBytes>$m?50:800)+t);return e.unsuspend=n,function(){e.unsuspend=null,clearTimeout(r),clearTimeout(i)}}:null}function th(e){if(e.count===0&&(e.imgCount===0||!e.waitingForImages)){if(e.stylesheets)ah(e,e.stylesheets);else if(e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}}}function nh(){this.count--,th(this)}function rh(){this.imgCount--,th(this)}var ih=null;function ah(e,t){e.stylesheets=null,e.unsuspend!==null&&(e.count++,ih=new Map,t.forEach(oh,e),ih=null,nh.call(e))}function oh(e,t){if(!(t.state.loading&4)){var n=ih.get(e);if(n)var r=n.get(null);else{n=new Map,ih.set(e,n);for(var i=e.querySelectorAll(`link[data-precedence],style[data-precedence]`),a=0;a<i.length;a++){var o=i[a];(o.nodeName===`LINK`||o.getAttribute(`media`)!==`not all`)&&(n.set(o.dataset.precedence,o),r=o)}r&&n.set(null,r)}i=t.instance,o=i.getAttribute(`data-precedence`),a=n.get(o)||r,a===r&&n.set(null,i),n.set(o,i),this.count++,r=nh.bind(this),i.addEventListener(`load`,r),i.addEventListener(`error`,r),a?a.parentNode.insertBefore(i,a.nextSibling):(e=e.nodeType===9?e.head:e,e.insertBefore(i,e.firstChild)),t.state.loading|=4}}var sh={$$typeof:k,Provider:null,Consumer:null,_currentValue:_e,_currentValue2:_e,_threadCount:0};function ch(e,t,n,r,i,a,o,s,c){this.tag=1,this.containerInfo=e,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=ut(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=ut(0),this.hiddenUpdates=ut(null),this.identifierPrefix=r,this.onUncaughtError=i,this.onCaughtError=a,this.onRecoverableError=o,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=c,this.transitionTypes=null,this.incompleteTransitions=new Map}function lh(e,t,n,r,i,a,o,s,c,l,u,d){return e=new ch(e,t,n,o,c,l,u,d,s),t=1,!0===a&&(t|=24),a=ji(3,null,null,t),e.current=a,a.stateNode=e,t=ja(),t.refCount++,e.pooledCache=t,t.refCount++,a.memoizedState={element:r,isDehydrated:n,cache:t},po(a),e}function uh(e){return e?(e=ki,e):ki}function dh(e,t,n,r,i,a){i=uh(i),r.context===null?r.context=i:r.pendingContext=i,r=ho(t),r.payload={element:n},a=a===void 0?null:a,a!==null&&(r.callback=a),n=go(e,r,t),n!==null&&(Md(n,e,t),_o(n,e,t))}function fh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ph(e,t){fh(e,t),(e=e.alternate)&&fh(e,t)}function mh(e){if(e.tag===13||e.tag===31){var t=Ei(e,67108864);t!==null&&Md(t,e,67108864),ph(e,67108864)}}function hh(e){if(e.tag===13||e.tag===31){var t=kd();t=gt(t);var n=Ei(e,t);n!==null&&Md(n,e,t),ph(e,t)}}var gh=!0;function _h(e,t,n,r){var i=N.T;N.T=null;var a=P.p;try{P.p=2,yh(e,t,n,r)}finally{P.p=a,N.T=i}}function vh(e,t,n,r){var i=N.T;N.T=null;var a=P.p;try{P.p=8,yh(e,t,n,r)}finally{P.p=a,N.T=i}}function yh(e,t,n,r){if(gh){var i=bh(r);if(i===null)Gf(e,t,r,xh,n),Mh(e,r);else if(Ph(i,e,t,n,r))r.stopPropagation();else if(Mh(e,r),t&4&&-1<jh.indexOf(e)){for(;i!==null;){var a=Mt(i);if(a!==null)switch(a.tag){case 3:if(a=a.stateNode,a.current.memoizedState.isDehydrated){var o=it(a.pendingLanes);if(o!==0){var s=a;for(s.pendingLanes|=2,s.entangledLanes|=2;o;){var c=1<<31-Ze(o);s.entanglements[1]|=c,o&=~c}Tf(a),!(q&6)&&(Q=ze()+500,Ef(0,!1))}}break;case 31:case 13:s=Ei(a,2),s!==null&&Md(s,a,2),Ld(),ph(a,2)}if(a=bh(r),a===null&&Gf(e,t,r,xh,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Gf(e,t,r,null,n)}}function bh(e){return e=yn(e),Sh(e)}var xh=null;function Sh(e){if(xh=null,e=jt(e),e!==null){var t=c(e);if(t===null)e=null;else{var n=t.tag;if(n===13){if(e=l(t),e!==null)return e;e=null}else if(n===31){if(e=u(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null)}}return xh=e,null}function Ch(e){switch(e){case`beforetoggle`:case`cancel`:case`click`:case`close`:case`contextmenu`:case`copy`:case`cut`:case`auxclick`:case`dblclick`:case`dragend`:case`dragstart`:case`drop`:case`focusin`:case`focusout`:case`input`:case`invalid`:case`keydown`:case`keypress`:case`keyup`:case`mousedown`:case`mouseup`:case`paste`:case`pause`:case`play`:case`pointercancel`:case`pointerdown`:case`pointerup`:case`ratechange`:case`reset`:case`seeked`:case`submit`:case`toggle`:case`touchcancel`:case`touchend`:case`touchstart`:case`volumechange`:case`change`:case`selectionchange`:case`textInput`:case`compositionstart`:case`compositionend`:case`compositionupdate`:case`beforeblur`:case`afterblur`:case`beforeinput`:case`blur`:case`fullscreenchange`:case`fullscreenerror`:case`focus`:case`hashchange`:case`popstate`:case`select`:case`selectstart`:return 2;case`drag`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`mousemove`:case`mouseout`:case`mouseover`:case`pointermove`:case`pointerout`:case`pointerover`:case`resize`:case`scroll`:case`touchmove`:case`wheel`:case`mouseenter`:case`mouseleave`:case`pointerenter`:case`pointerleave`:return 8;case`message`:switch(Be()){case Ve:return 2;case He:return 8;case Ue:case We:return 32;case Ge:return 268435456;default:return 32}default:return 32}}var wh=!1,Th=null,Eh=null,Dh=null,Oh=new Map,kh=new Map,Ah=[],jh=`mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset`.split(` `);function Mh(e,t){switch(e){case`focusin`:case`focusout`:Th=null;break;case`dragenter`:case`dragleave`:Eh=null;break;case`mouseover`:case`mouseout`:Dh=null;break;case`pointerover`:case`pointerout`:Oh.delete(t.pointerId);break;case`gotpointercapture`:case`lostpointercapture`:kh.delete(t.pointerId)}}function Nh(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Mt(t),t!==null&&mh(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Ph(e,t,n,r,i){switch(t){case`focusin`:return Th=Nh(Th,e,t,n,r,i),!0;case`dragenter`:return Eh=Nh(Eh,e,t,n,r,i),!0;case`mouseover`:return Dh=Nh(Dh,e,t,n,r,i),!0;case`pointerover`:var a=i.pointerId;return Oh.set(a,Nh(Oh.get(a)||null,e,t,n,r,i)),!0;case`gotpointercapture`:return a=i.pointerId,kh.set(a,Nh(kh.get(a)||null,e,t,n,r,i)),!0}return!1}function Fh(e){var t=jt(e.target);if(t!==null){var n=c(t);if(n!==null){if(t=n.tag,t===13){if(t=l(n),t!==null){e.blockedOn=t,yt(e.priority,function(){hh(n)});return}}else if(t===31){if(t=u(n),t!==null){e.blockedOn=t,yt(e.priority,function(){hh(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ih(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=bh(e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);vn=r,n.target.dispatchEvent(r),vn=null}else return t=Mt(n),t!==null&&mh(t),e.blockedOn=n,!1;t.shift()}return!0}function Lh(e,t,n){Ih(e)&&n.delete(t)}function Rh(){wh=!1,Th!==null&&Ih(Th)&&(Th=null),Eh!==null&&Ih(Eh)&&(Eh=null),Dh!==null&&Ih(Dh)&&(Dh=null),Oh.forEach(Lh),kh.forEach(Lh)}function zh(e,t){e.blockedOn===t&&(e.blockedOn=null,wh||(wh=!0,n.unstable_scheduleCallback(n.unstable_NormalPriority,Rh)))}var Bh=null;function Vh(e){Bh!==e&&(Bh=e,n.unstable_scheduleCallback(n.unstable_NormalPriority,function(){Bh===e&&(Bh=null);for(var t=0;t<e.length;t+=3){var n=e[t],r=e[t+1],i=e[t+2];if(typeof r!=`function`){if(Sh(r||n)===null)continue;break}var a=Mt(n);a!==null&&(e.splice(t,3),t-=3,Qs(a,{pending:!0,data:i,method:n.method,action:r},r,i))}}))}function Hh(e){function t(t){return zh(t,e)}Th!==null&&zh(Th,e),Eh!==null&&zh(Eh,e),Dh!==null&&zh(Dh,e),Oh.forEach(t),kh.forEach(t);for(var n=0;n<Ah.length;n++){var r=Ah[n];r.blockedOn===e&&(r.blockedOn=null)}for(;0<Ah.length&&(n=Ah[0],n.blockedOn===null);)Fh(n),n.blockedOn===null&&Ah.shift();if(n=(e.ownerDocument||e).$$reactFormReplay,n!=null)for(r=0;r<n.length;r+=3){var i=n[r],a=n[r+1],o=i[St]||null;if(typeof a==`function`)o||Vh(n);else if(o){var s=null;if(a&&a.hasAttribute(`formAction`)){if(i=a,o=a[St]||null)s=o.formAction;else if(Sh(i)!==null)continue}else s=o.action;typeof s==`function`?n[r+1]=s:(n.splice(r,3),r-=3),Vh(n)}}}function Uh(){function e(e){e.canIntercept&&e.info===`react-transition`&&e.intercept({handler:function(){return new Promise(function(e){return i=e})},focusReset:`manual`,scroll:`manual`})}function t(){i!==null&&(i(),i=null),r||setTimeout(n,20)}function n(){if(!r&&!navigation.transition){var e=navigation.currentEntry;e&&e.url!=null&&navigation.navigate(e.url,{state:e.getState(),info:`react-transition`,history:`replace`})}}if(typeof navigation==`object`){var r=!1,i=null;return navigation.addEventListener(`navigate`,e),navigation.addEventListener(`navigatesuccess`,t),navigation.addEventListener(`navigateerror`,t),setTimeout(n,100),function(){r=!0,navigation.removeEventListener(`navigate`,e),navigation.removeEventListener(`navigatesuccess`,t),navigation.removeEventListener(`navigateerror`,t),i!==null&&(i(),i=null)}}}function Wh(e){this._internalRoot=e}Gh.prototype.render=Wh.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(o(409));var n=t.current;dh(n,kd(),e,t,null,null)},Gh.prototype.unmount=Wh.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;dh(e.current,2,null,e,null,null),Ld(),t[Ct]=null}};function Gh(e){this._internalRoot=e}Gh.prototype.unstable_scheduleHydration=function(e){if(e){var t=vt();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Ah.length&&t!==0&&t<Ah[n].priority;n++);Ah.splice(n,0,e),n===0&&Fh(e)}};var Kh=r.version;if(Kh!==`19.3.1`)throw Error(o(527,Kh,`19.3.1`));P.findDOMNode=function(e){var t=e._reactInternals;if(t===void 0)throw typeof e.render==`function`?Error(o(188)):(e=Object.keys(e).join(`,`),Error(o(268,e)));return e=f(t),e=e===null?null:p(e),e=e===null?null:e.stateNode,e};var qh={bundleType:0,version:`19.3.1`,rendererPackageName:`react-dom`,currentDispatcherRef:N,reconcilerVersion:`19.3.1`};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`){var Jh=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Jh.isDisabled&&Jh.supportsFiber)try{Je=Jh.inject(qh),Ye=Jh}catch{}}e.createRoot=function(e,t){if(!s(e))throw Error(o(299));var n=!1,r=``,i=xc,a=Sc,c=Cc;return t!=null&&(!0===t.unstable_strictMode&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onUncaughtError!==void 0&&(i=t.onUncaughtError),t.onCaughtError!==void 0&&(a=t.onCaughtError),t.onRecoverableError!==void 0&&(c=t.onRecoverableError)),t=lh(e,1,!1,null,null,n,r,null,i,a,c,Uh),e[Ct]=t.current,Uf(e),new Wh(t)}})),Dt=n(((e,t)=>{function n(){if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`&&typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE==`function`)try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=Et()})),z=e(t(),1),Ot=Dt();function kt(e){let t=``;for(let n of e){let e=n.codePointAt(0)??0;t+=n===`"`||n===`\\`||e<32||e===127?`\\${(e===0?65533:e).toString(16)} `:n}return t}function At(e){return e===`base`?``:e.split(`+`).map(e=>{let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?e:e.slice(t+1);return`[${We(n)}="${kt(r)}"]`}).join(``)}var jt={avatar:[{property:`borderRadius`,vars:[`--_avatar-radius`]}],banner:[{property:`borderRadius`,vars:[`--_banner-radius`]}],"bottom-sheet":[{property:`padding`,expand:`container`}],button:[{property:`borderRadius`,vars:[`--_button-radius`]}],card:[{property:`borderRadius`,vars:[`--_card-radius`]},{property:`padding`,expand:`container`}],"chat-composer":[{property:`borderRadius`,vars:[`--_chat-composer-radius`]},{property:`padding`,vars:[`--_chat-composer-padding`]}],dialog:[{property:`borderRadius`,vars:[`--_dialog-radius`]},{property:`padding`,expand:`container`}],drawer:[{property:`padding`,expand:`container`}],"context-menu":[{property:`borderRadius`,vars:[`--_dropdown-menu-radius`]},{property:`padding`,vars:[`--_dropdown-menu-padding`]}],"dropdown-menu":[{property:`borderRadius`,vars:[`--_dropdown-menu-radius`]},{property:`padding`,vars:[`--_dropdown-menu-padding`]}],field:[{property:`borderRadius`,vars:[`--_field-radius`]}],"hover-card":[{property:`borderRadius`,vars:[`--_hovercard-radius`]}],icon:[{property:`width`,vars:[`--_icon-width`],replaces:!0},{property:`height`,vars:[`--_icon-height`],replaces:!0}],item:[{property:`paddingInline`,vars:[`--_item-inset-inline`]}],"number-input":[{property:`padding`,expand:`container`},{property:`borderRadius`,vars:[`--_field-radius`]}],popover:[{property:`borderRadius`,vars:[`--_popover-radius`]}],"progress-bar-mark":[{property:`width`,vars:[`--_progressbar-mark-width`],replaces:!0},{property:`height`,vars:[`--_progressbar-mark-height`],replaces:!0}],section:[{property:`padding`,expand:`container`}],"segmented-control":[{property:`borderRadius`,vars:[`--_segmented-control-radius`]},{property:`padding`,vars:[`--_segmented-control-padding`]}],"text-area":[{property:`paddingInline`,vars:[`--_textarea-inline-padding`],replaces:!0}]};function Mt(e,t){let n=jt[e];return n?n.filter(e=>e.property===t):[]}var Nt=e=>{let t=e/255;return t<=.04045?t/12.92:((t+.055)/1.055)**2.4},Pt=e=>(e<=.0031308?12.92*e:1.055*e**(1/2.4)-.055)*255;function Ft(e,t){let n=e.a;return{r:e.r*n+t.r*(1-n),g:e.g*n+t.g*(1-n),b:e.b*n+t.b*(1-n),a:1}}function It({r:e,g:t,b:n}){return .2126*Nt(e)+.7152*Nt(t)+.0722*Nt(n)}function Lt(e,t){let n=t.a<1?Ft(t,{r:255,g:255,b:255,a:1}):t,r=It(e.a<1?Ft(e,n):e),i=It(n);return(Math.max(r,i)+.05)/(Math.min(r,i)+.05)}function Rt({r:e,g:t,b:n}){let r=Nt(e),i=Nt(t),a=Nt(n),o=Math.cbrt(.4122214708*r+.5363325363*i+.0514459929*a),s=Math.cbrt(.2119034982*r+.6806995451*i+.1073969566*a),c=Math.cbrt(.0883024619*r+.2817188376*i+.6299787005*a),l=.2104542553*o+.793617785*s-.0040720468*c,u=1.9779984951*o-2.428592205*s+.4505937099*c,d=.0259040371*o+.7827717662*s-.808675766*c;return{l,c:Math.hypot(u,d),h:Math.atan2(d,u)}}function zt({l:e,c:t,h:n}){let r=t*Math.cos(n),i=t*Math.sin(n),a=(e+.3963377774*r+.2158037573*i)**3,o=(e-.1055613458*r-.0638541728*i)**3,s=(e-.0894841775*r-1.291485548*i)**3;return[4.0767416621*a-3.3077115913*o+.2309699292*s,-1.2684380046*a+2.6097574011*o-.3413193965*s,-.0041960863*a-.7034186147*o+1.707614701*s]}function Bt(e){let t=e.c;for(let n=0;n<40;n++){let n=zt({...e,c:t});if(n.every(e=>e>=-1e-4&&e<=1.0001)){let[e,t,r]=n.map(e=>Pt(Math.min(1,Math.max(0,e))));return{r:e,g:t,b:r,a:1}}t*=.9}let[n,r,i]=zt({...e,c:0}).map(e=>Pt(Math.min(1,Math.max(0,e))));return{r:n,g:r,b:i,a:1}}var Vt=[.2,.95];function Ht(e){let{l:t,c:n}=Rt(e);return n>=.03&&t>=Vt[0]&&t<=Vt[1]}function Ut(e,t,n){let r=e=>t.every(t=>Lt(e,t)>=n),i={...e,a:1};if(r(i))return i;let a=t.reduce((e,t)=>e+It(t),0)/t.length,o=Rt(i),s=a>.18,c=s?0:o.l,l=s?o.l:1,u=e=>Bt({...o,l:e}),d=u(+!s);if(!r(d))return null;for(let e=0;e<32;e++){let e=(c+l)/2,t=r(u(e));s?t?c=e:l=e:t?l=e:c=e}let f=u(s?c:l);return r(f)?f:d}function Wt(e){return e===void 0?null:fe(e)}function B(e){let t=ue(e.r,e.g,e.b);return e.a>=1?t:`${t}${Math.round(e.a*255).toString(16).padStart(2,`0`).toUpperCase()}`}var Gt=[`info`,`success`,`warning`,`error`],Kt=4.5,qt=4.58,Jt=[`light`,`dark`];function Yt(e){return`--color-${e}`}function Xt(e){let t=[],n=0,r=``;for(let i of e)i===`(`&&n++,i===`)`&&n--,i===`,`&&n===0?(t.push(r.trim()),r=``):r+=i;return t.push(r.trim()),t}function Zt(e,t,n,r=new Set){if(!r.has(t))return Qt(e,e[t]??et[t],n,new Set([...r,t]))}function Qt(e,t,n,r){if(Array.isArray(t))return Qt(e,t[n===`light`?0:1],n,r);if(typeof t!=`string`)return;let i=t.trim();if(/^light-dark\(/i.test(i)&&i.endsWith(`)`)){let[t,a]=Xt(i.slice(i.indexOf(`(`)+1,-1));return Qt(e,n===`light`?t:a,n,r)}let a=/^var\(\s*(--[\w-]+)\s*(?:,(.*))?\)$/s.exec(i);return a?Zt(e,a[1],n,r)??(a[2]?Qt(e,a[2],n,r):void 0):i}var $t={};function en(e){let t=new Set([...Object.keys(et),...Object.keys(e)]),n=n=>{let r={};for(let i of t){let t=Zt(e,i,n);t!==void 0&&(r[i]=t)}return r};return{light:n(`light`),dark:n(`dark`)}}var tn;function nn(){return tn??=en($t),tn}function rn(e,t,n){let r=Wt(t[Yt(e)]);if(r!==null&&Ht(r))return{...r,a:1};let i=Wt(nn()[n][Yt(e)]);return i===null?null:{...i,a:1}}function an(e,t){return e.a<1?Ft(e,t):e}function on(e,t,n){let r=Wt(t[Yt(e)]),i=rn(e,t,n),a=Wt(t[`--color-background-surface`]),o=Wt(t[`--color-background-card`]),s=Wt(t[`--color-${e}-muted`]),c=Wt(t[`--color-on-dark`]),l=Wt(t[`--color-on-light`]);if(r===null||i===null||a===null||o===null||s===null||c===null||l===null)return null;let u=an(a,{r:255,g:255,b:255,a:1}),d=an(o,u),f=an(r,u),p={},m=an(c,f),h=an(l,f),g=Lt(m,f)>=Lt(h,f)?m:h,_={r:255,g:255,b:255,a:1},v={r:0,g:0,b:0,a:1};p.on=Lt(g,f)>=Kt?g:Lt(_,f)>=Lt(v,f)?_:v;let y=Ut(i,[u,d],qt);y!==null&&(p.text=y);let b=Ut(i,[Ft(s.a<1?s:{...s,a:1},u)],qt);return b!==null&&(p.onMuted=b),p}var sn={on:e=>`--color-on-${e}`,text:e=>`--color-text-${e}`,onMuted:e=>`--color-on-${e}-muted`};function cn(e,t={}){let n=new Set(Object.keys(e)),r=n.has(`--color-background-surface`)||n.has(`--color-background-card`),i=n.has(`--color-on-dark`)||n.has(`--color-on-light`),a=en({...t,...e}),o=[];for(let e of Gt){let t=n.has(Yt(e)),s=n.has(`--color-${e}-muted`),c={on:t||i,text:t||r,onMuted:t||s||r};if(!c.on&&!c.text&&!c.onMuted)continue;let l=Jt.map(t=>on(e,a[t],t));if(!l.some(e=>e===null))for(let t of Object.keys(sn)){let r=sn[t](e);if(!c[t]||n.has(r))continue;let[i,a]=l.map(e=>e?.[t]);i!==void 0&&a!==void 0&&o.push([r,`light-dark(${B(i)}, ${B(a)})`])}}return o}function ln(e){return e===`
`||e===`\r`||e===`\f`}function un(e){return e===` `||e===`	`||ln(e)}function dn(e){return/^[0-9a-fA-F]$/.test(e)}function fn(e){return/^[A-Za-z_]$/.test(e)||e.charCodeAt(0)>=128}function pn(e){return fn(e)||/^[0-9-]$/.test(e)}function mn(e){let t=e.replace(/\r\n?|\f/g,`
`).replace(/\0/g,`�`),n=``;for(let e=0;e<t.length;e++){let r=t.charCodeAt(e);if(r>=55296&&r<=56319){let r=t.charCodeAt(e+1);r>=56320&&r<=57343?(n+=t[e]+t[e+1],e++):n+=`�`}else n+=r>=56320&&r<=57343?`�`:t[e]}return n}var hn=/[\x01-\x08\x0b\x0e-\x1f\x7f]/;function gn(e,t){return e[t]===`\\`&&t+1<e.length&&!ln(e[t+1])}function _n(e,t){let n=e[t];if(n===`-`){let n=e[t+1];return n!==void 0&&(n===`-`||fn(n)||gn(e,t+1))}return n===`\\`?gn(e,t):n!==void 0&&fn(n)}function vn(e,t){if(dn(e[t])){let n=t;for(;n<e.length&&n-t<6&&dn(e[n]);)n++;return n<e.length&&un(e[n])&&n++,n}return t+((e.codePointAt(t)??0)>65535?2:1)}function yn(e,t){let n=``,r=t;for(;r<e.length;){let t=e[r];if(pn(t))n+=t,r++;else if(gn(e,r)){let t=vn(e,r+1);if(dn(e[r+1])){let i=e.slice(r+1,t).trim(),a=parseInt(i,16);n+=a===0||a>1114111||a>=55296&&a<=57343?`�`:String.fromCodePoint(a)}else n+=e.slice(r+1,t);r=t}else break}return{end:r,name:n}}function bn(e,t){let n=e[t],r=t+1;for(;r<e.length;){let t=e[r];if(t===n)return{end:r+1};if(ln(t))return{reason:`a newline inside a quoted string makes a bad string`};if(t===`\\`){if(r+1>=e.length)break;r=ln(e[r+1])?r+2:vn(e,r+1)}else r++}return{reason:`an unclosed ${n} string would swallow the rest of the rule`}}function xn(e,t){let n=t;for(;n<e.length;){let t=e[n];if(t===`)`)return{end:n+1};if(un(t)){for(;n<e.length&&un(e[n]);)n++;if(n>=e.length)break;return e[n]===`)`?{end:n+1}:{reason:`whitespace inside an unquoted url() makes a bad url`}}if(hn.test(t))return{reason:`a non-printable character inside url() makes a bad url`};if(t===`"`||t===`'`||t===`(`)return{reason:`"${t}" inside an unquoted url() makes a bad url; quote the URL instead`};if(t===`\\`){if(!gn(e,n))return{reason:`a stray backslash inside url() makes a bad url`};n=vn(e,n+1)}else n++}return{reason:`an unclosed url( would swallow the rest of the rule`}}var Sn={"(":`)`,"[":`]`,"{":`}`};function Cn(e){let t=mn(e),n=[],r=0;for(;r<t.length;){let e=t[r];if(e===`/`&&t[r+1]===`*`){let e=t.indexOf(`*/`,r+2);if(e===-1)return`an unclosed /* comment would swallow the rest of the rule`;r=e+2}else if(e===`"`||e===`'`){let e=bn(t,r);if(`reason`in e)return e.reason;r=e.end}else if(_n(t,r)){let e=yn(t,r);if(r=e.end,t[r]===`(`&&e.name.toLowerCase()===`url`){let e=r+1;for(;e<t.length&&un(t[e]);)e++;if(t[e]===`"`||t[e]===`'`){n.push(`)`),r+=1;continue}let i=xn(t,e);if(`reason`in i)return i.reason;r=i.end}}else if(e===`\\`){if(r+1>=t.length)return`a trailing backslash would escape the declaration terminator`;r=ln(t[r+1])?r+1:vn(t,r+1)}else if(e===`(`||e===`[`||e===`{`){if(e===`{`&&n.length===0)return`an unquoted "{" would open a nested block`;n.push(Sn[e]),r++}else if(e===`)`||e===`]`||e===`}`){if(n.length===0)return e===`}`?`an unquoted "}" would close the rule`:`an unbalanced "${e}" has nothing to close`;let t=n.pop();if(t!==e)return`"${e}" does not close the open "${t===`)`?`(`:t===`]`?`[`:`{`}"`;r++}else{if(e===`;`&&n.length===0)return`an unquoted ";" would end the declaration`;r++}}if(n.length>0){let e=n[n.length-1];return`an unclosed "${e===`)`?`(`:e===`]`?`[`:`{`}" would swallow the rest of the rule`}return null}function wn(e,t){let n=t;for(;;){for(;n<e.length&&un(e[n]);)n++;if(e[n]!==`/`||e[n+1]!==`*`)return n;let t=e.indexOf(`*/`,n+2);if(t===-1)return e.length;n=t+2}}function Tn(e,t){let n=wn(e,t+1);if(!_n(e,n))return null;let r=yn(e,n);if(!r.name.startsWith(`--`))return null;let i=wn(e,r.end);return e[i]===`,`||e[i]===`)`?r.name:null}function En(e,t){let n=e[t],r=e[t+1],i=e[t+2];return n===`+`||n===`-`?/^[0-9]$/.test(r??``)||r===`.`&&/^[0-9]$/.test(i??``):/^[0-9]$/.test(n??``)||n===`.`&&/^[0-9]$/.test(r??``)}function Dn(e,t){let n=t;for((e[n]===`+`||e[n]===`-`)&&n++;/^[0-9]$/.test(e[n]??``);)n++;if(e[n]===`.`&&/^[0-9]$/.test(e[n+1]??``))for(n+=2;/^[0-9]$/.test(e[n]??``);)n++;if((e[n]===`e`||e[n]===`E`)&&(/^[0-9]$/.test(e[n+1]??``)||(e[n+1]===`+`||e[n+1]===`-`)&&/^[0-9]$/.test(e[n+2]??``)))for(n++,(e[n]===`+`||e[n]===`-`)&&n++;/^[0-9]$/.test(e[n]??``);)n++;return n}var On=new Map,kn=4096;function An(e){let t=On.get(e);if(t!==void 0)return t;let n=Cn(e);if(n!==null){let t={reason:n,references:[],important:!1,singleVarReference:null};return On.size>=kn&&On.clear(),On.set(e,t),t}let r=mn(e),i=[],a=[],o=[],s=[],c=0,l=e=>{a.length===0&&(c++,s.length===0&&s.push(e),o.push(e),o.length>2&&o.shift())},u=0;for(;u<r.length;){let e=r[u];if(un(e))u++;else if(e===`/`&&r[u+1]===`*`)u=r.indexOf(`*/`,u+2)+2;else if(e===`"`||e===`'`){l({type:`other`,value:e});let t=bn(r,u);u=`end`in t?t.end:r.length}else{if(e===`#`){let e=u+1;if(pn(r[e])||gn(r,e)){l({type:`other`,value:`#`}),u=yn(r,e).end;continue}}if(e===`@`&&_n(r,u+1))l({type:`other`,value:`@`}),u=yn(r,u+1).end;else if(En(r,u))l({type:`other`,value:e}),u=Dn(r,u),_n(r,u)?u=yn(r,u).end:r[u]===`%`&&u++;else if(_n(r,u)){let e=yn(r,u);u=e.end;let t=r[u]===`(`,n=t&&e.name.toLowerCase()===`var`?Tn(r,u):null;if(l(n===null?{type:t?`other`:`ident`,value:e.name}:{type:`var-function`,value:e.name,reference:n}),t){if(n!==null&&i.push(n),e.name.toLowerCase()===`url`){let e=u+1;for(;e<r.length&&un(r[e]);)e++;if(r[e]!==`"`&&r[e]!==`'`){let t=xn(r,e);u=`end`in t?t.end:r.length;continue}}a.push(`)`),u++}}else e===`(`||e===`[`||e===`{`?(l({type:`other`,value:e}),a.push(Sn[e]),u++):e===`)`||e===`]`||e===`}`?(a.pop(),u++):(l(e===`!`?{type:`delim`,value:e}:{type:`other`,value:e}),u++)}}let d=o.at(-2),f=o.at(-1),p=d?.type===`delim`&&d.value===`!`&&f?.type===`ident`&&f.value.toLowerCase()===`important`,m=s[0],h={reason:null,references:i,important:p,singleVarReference:c-(p?2:0)==1&&m?.type===`var-function`?m.reference??null:null};return On.size>=kn&&On.clear(),On.set(e,h),h}function jn(e){let t=mn(e);if(t.length===0||!_n(t,0))return null;let n=yn(t,0);return n.end===t.length?n.name:null}function Mn(e){let t=mn(e);if(t.length===0||!_n(t,0))return`a property name must be a CSS identifier`;let{end:n}=yn(t,0);return n===t.length?null:`a property name must be one CSS identifier; "${t[n]}" cannot appear in one`}function Nn(e){return`[data-${$e}-theme="${e}"]`}var Pn=`[data-${$e}-theme]`;function Fn(e){return`[data-${$e}-media="${e}"]`}function In(e,t){return`.${Qe}-${e}${t}`}function Ln(e,t,n){let r=In(e,t);return n?`:is(${Fn(n)}) :is(${r})`:r}var Rn=`:where(:not(:disabled,[aria-disabled="true"]))`;function zn(e,t){let n=[],r=0,i=0;for(let t=0;t<e.length;t++){let a=e[t];a===`(`?r++:a===`)`?r=Math.max(0,r-1):a===`,`&&r===0&&(n.push(e.slice(i,t).trim()),i=t+1)}n.push(e.slice(i).trim());let a=Bn(t);return n.map(e=>`${e}${a}`).join(`, `)}function Bn(e){if(!/^:hover(?![-\w])/.test(e)||e.includes(`[aria-disabled`))return e;let t=e.indexOf(`::`);return t===-1?e+Rn:e.slice(0,t)+Rn+e.slice(t)}function Vn(e){return e.replace(/[A-Z]/g,e=>`-${e.toLowerCase()}`)}var Hn=new Set([`padding`,`paddingBlock`,`paddingInline`,`paddingBlockStart`,`paddingBlockEnd`,`paddingInlineStart`,`paddingInlineEnd`]),Un={paddingTop:`paddingBlockStart`,paddingBottom:`paddingBlockEnd`},Wn=new Set([...Hn,...Object.keys(Un)]);function Gn(e){let t={};for(let[n,r]of e)switch(Un[n]??n){case`padding`:{let e=r.trim().split(/\s+/);e.length===1?(t.blockStart=e[0],t.blockEnd=e[0],t.inline=e[0]):e.length===2?(t.blockStart=e[0],t.blockEnd=e[0],t.inline=e[1]):e.length>=3&&(t.blockStart=e[0],t.inline=e[1],t.blockEnd=e[2]);break}case`paddingBlock`:{let e=r.trim().split(/\s+/);t.blockStart=e[0],t.blockEnd=e[1]??e[0];break}case`paddingInline`:{let e=r.trim().split(/\s+/);e.length===1?t.inline=e[0]:(t.inlineStart=e[0],t.inlineEnd=e[1]);break}case`paddingBlockStart`:t.blockStart=r;break;case`paddingBlockEnd`:t.blockEnd=r;break;case`paddingInlineStart`:t.inlineStart=r;break;case`paddingInlineEnd`:t.inlineEnd=r}return t}function Kn(e,t,n=!1){let r=Ge(`${e}-padding`),i=[],a=t.inlineStart??t.inline,o=t.inlineEnd??t.inline;if(a!=null&&o!=null&&a===o&&t.blockStart!=null&&t.blockEnd!=null&&a===t.blockStart&&t.blockStart===t.blockEnd?i.push([r,a??``]):(t.inlineStart!=null||t.inlineEnd!=null?(a!=null&&i.push([`${r}-inline-start`,a]),o!=null&&i.push([`${r}-inline-end`,o])):t.inline!=null&&i.push([`${r}-inline`,t.inline]),t.blockStart!=null&&i.push([`${r}-block-start`,t.blockStart]),t.blockEnd!=null&&i.push([`${r}-block-end`,t.blockEnd])),n){let e=new Set(i.map(([e])=>e)),t=e.has(r)?[`${r}-inline`,`${r}-inline-start`,`${r}-inline-end`,`${r}-block-start`,`${r}-block-end`]:e.has(`${r}-inline`)?[`${r}-inline-start`,`${r}-inline-end`]:[];for(let n of t)e.has(n)||i.push([n,`initial`])}return i}var qn=e=>{console.warn(`[astryx theme] ${e}`)};function Jn(e){return e?t=>e.push(t):qn}function Yn(e){let t=JSON.stringify(e);return t.length>80?`${t.slice(0,77)}..."`:t}function Xn(e,t,n,r,i){e(`dropped "${t}" in ${r}: ${i} (value: ${Yn(n)})`)}function Zn(e,t,n,r){let i=Mn(e)??Cn(t);return i===null||(Xn(r,e,t,n,i),!1)}function Qn(e,t,n,r=e=>e){let i=[];for(let[a,o]of e){let e=rr(o),s=r(a);Zn(s,e,t,n)&&i.push(`    ${s}: ${e};`)}return i.join(`
`)}var $n=[[`--color-text-info`,`--color-text-blue`],[`--color-text-success`,`--color-text-green`],[`--color-text-warning`,`--color-text-yellow`],[`--color-text-error`,`--color-text-red`],[`--color-on-info-muted`,`--color-text-blue`],[`--color-on-success-muted`,`--color-text-green`],[`--color-on-warning-muted`,`--color-text-yellow`],[`--color-on-error-muted`,`--color-text-red`],[`--color-icon-info`,`--color-text-info`],[`--color-icon-success`,`--color-text-success`],[`--color-icon-warning`,`--color-text-warning`],[`--color-icon-error`,`--color-text-error`],[`--color-border-info`,`--color-text-info`],[`--color-border-success`,`--color-text-success`],[`--color-border-warning`,`--color-text-warning`],[`--color-border-error`,`--color-text-error`],[`--duration-exit`,`--duration-fast`]],er=new Map($n),tr=/^--color-(?:info|success|warning|error|on-|background-surface|background-card)/,nr=new Map($n.map(([e],t)=>[e,t]));function rr(e){return typeof e==`string`?e:String(e)}function ir(e,...t){for(let n of t)for(let[t,r]of Object.entries(n??{})){let n=jn(t),i=rr(r),a=An(i);n!==null&&a.reason===null&&(!e.important.has(n)||a.important)&&(e.values[n]=i,a.important?e.important.add(n):e.important.delete(n))}}function ar(...e){let t={values:{},important:new Set};return ir(t,...e),t}function or(e,t,n){let r=[...An(n).references],i=new Set;for(;r.length>0;){let n=r.pop();if(n!==void 0){if(n===t)return!0;if(!i.has(n)&&Object.prototype.hasOwnProperty.call(e,n)){i.add(n);for(let i of An(rr(e[n])).references)(i===t||Object.prototype.hasOwnProperty.call(e,i))&&r.push(i)}}}return!1}var sr={sm:1,md:2,lg:3,xl:4,"2xl":5};function cr(e,t,n,r,i){let a=e.width?.from;if(a!==void 0&&t<sr[a])return!1;let o=e.width?.below;return o!==void 0&&t>=sr[o]?!1:(e.pointer===void 0||e.pointer===n)&&(e.contrast===void 0||e.contrast===r)&&(e.motion===void 0||e.motion===i)}var lr=new WeakMap;function ur(e){let t=lr.get(e);if(t!==void 0)return t;let n=new Map;for(let t=0;t<=5;t++)for(let r of[`coarse`,`fine`,`none`])for(let i of[`more`,`less`,`no-preference`,`custom`])for(let a of[`reduce`,`no-preference`]){let o=e.flatMap((e,n)=>cr(e.when,t,r,i,a)?[n]:[]);n.set(o.join(`,`),o)}let r=[...n.values()];return lr.set(e,r),r}function dr(e,t,n,r,i,a=[]){let o=ar(e,Object.fromEntries(n),t);for(let e of i){let t=r[e];ir(o,t.tokens,Object.fromEntries(a[e]??[]),t.localTokens)}for(let e of i)ir(o,Object.fromEntries(mr(r[e].tokens)));return o.values}function fr(e){return!(`__adaptations`in e)&&!(`__adaptationRules`in e)?[]:Rr(e)}function pr(e,t={},n=[]){let r=cn(e,t),i=ar(e,Object.fromEntries(r),t).values,a=ur(n).map(i=>({indexes:i,values:dr(e,t,r,n,i)}));for(let[e,t]of $n){let n=`var(${t})`;if(!(i[e]!==void 0||i[t]===void 0&&!r.some(([e])=>e===t))&&a.every(({values:t})=>t[e]!==void 0||!or(t,e,n))){r.push([e,n]);for(let{values:t}of a)t[e]===void 0&&(t[e]=n)}}return r.sort(([e],[t])=>(nr.get(e)??0)-(nr.get(t)??0))}function mr(e){let t=ar(e).values;return $n.flatMap(([e])=>{let n=t[e];return n===void 0?[]:[[e,rr(n)]]})}function hr(e,t){return An(rr(e)).singleVarReference===t}function gr(e,t){let n=er.get(e);return n!==void 0&&hr(t,n)?t:`inherit`}function _r(e,t,n,r){let i=n.map(()=>[]),a=ar(e,Object.fromEntries(r),t).values,o=ur(n).map(a=>({indexes:a,values:dr(e,t,r,n,a,i)}));for(let[r,s]of n.entries()){let n=ar(s.tokens,s.localTokens).values;if(Object.keys(n).some(e=>tr.test(e))){let o=new Map(cn({...e,...s.tokens},{...t,...s.localTokens}));for(let[e,t]of o)n[e]===void 0&&a[e]!==t&&i[r].push([e,t])}let c=new Map(i[r]),l={...n,...Object.fromEntries(c)};for(let[e,t]of $n){let s=`var(${t})`,u=a[e];if(c.has(e)||n[e]!==void 0||l[t]===void 0&&!i[r].some(([e])=>e===t)||u!==void 0)continue;let d=o.filter(({indexes:e})=>e.includes(r));if(d.every(({values:t})=>t[e]!==void 0||!or(t,e,s))){i[r].push([e,s]);for(let{values:t}of d)t[e]===void 0&&(t[e]=s)}}}return i}function vr(e,t,n,r,i){let a=[],o=ar(t,Object.fromEntries(i),n).values,s=new Set(cn(t,n).map(([e])=>e)),c=ar(e).values,l=r.map(e=>new Map(mr(e.tokens))),u=ur(r).map(e=>({indexes:e,values:{...c}})),d=(e,t,n)=>{let r=ar({[e]:t});for(let t of n){let n=l[t].get(e);n!==void 0&&ir(r,{[e]:gr(e,n)})}return rr(r.values[e])};for(let[e,t]of $n){let n=`var(${t})`,r=o[e];if(c[e]!==void 0||c[t]===void 0&&!a.some(([e])=>e===t)||r!==void 0&&!hr(r,t)&&!s.has(e))continue;let i=u.map(({indexes:t,values:r})=>({values:r,candidate:d(e,n,t)}));if(i.every(({values:t,candidate:n})=>t[e]===n||!or(t,e,n))){a.push([e,n]);for(let{values:t,candidate:n}of i)t[e]=n}}let f=new Set(Gt.filter(e=>c[`--color-${e}`]!==void 0&&c[`--color-on-${e}`]===void 0).map(e=>`--color-on-${e}`));if(f.size>0)for(let[r,i]of cn({...t,...e},n))f.has(r)&&o[r]!==i&&a.push([r,i]);return a}function yr(e,t,n,r=Object.entries(e.tokens)){let i=[Qn(r,`${t}tokens`,n),Qn(Object.entries(e.localTokens??{}),`${t}localTokens`,n)].filter(e=>e.length>0).join(`
`);return r.length>0||Object.keys(e.localTokens??{}).length>0?`  :scope {\n${i}\n  }`:null}function br(e,t,n){return`${e}components.${t}[${JSON.stringify(n)}]`}var xr=new WeakMap,Sr;function Cr(e){return Object.entries(e??{}).map(([e,t])=>[e,typeof t,rr(t)])}function wr(e,t){return JSON.stringify([Cr(e.tokens),Cr(e.localTokens),t.map(e=>[e.query,e.when,Cr(e.tokens),Cr(e.localTokens),e.components])])}function Tr(e){let t=Rr(e),n=wr(e,t),r=xr.get(e);if(r?.signature===n)return r.context;let i={theme:e,rules:t,rootAliases:pr(e.tokens,e.localTokens,t)};return xr.set(e,{signature:n,context:i}),i}function Er(e){return Sr?.theme===e?Sr:Tr(e)}function Dr(e){return e.adaptationAliases??=_r(e.theme.tokens,e.theme.localTokens,e.rules,e.rootAliases),e.adaptationAliases}function Or(e,t){let n=[],r=Jn(t),i=e=>`var(${e})`,a=Sr?.theme===e?Sr:void 0,o=a?.rules??fr(e),s=a?.rootAliases??pr(e.tokens,e.localTokens,o),c=yr(e,``,r,[...Object.entries(e.tokens),...s]);return c!==null&&n.push(c),e.tokens[`--font-family-body`]&&n.push(`  :scope {
    font-family: var(--font-family-body);
  }`),e.components&&Mr(e.components,n,{sink:r}),Nr(i,n),Pr(e.components||{},n),Ir(e.components||{},n),Ar(e.components||{},n),n}var kr={normal:`var(--font-weight-normal)`,medium:`var(--font-weight-medium)`,semibold:`var(--font-weight-semibold)`,bold:`var(--font-weight-bold)`};function Ar(e,t,{surface:n,inheritedComponents:r,forceAll:i=!1,authoredOnly:a=!1}={}){let o=e.heading,s=o&&typeof o==`object`&&!Array.isArray(o)?o:void 0;if(!s&&!i)return;let c=r?.heading,l=c&&typeof c==`object`?c:void 0;for(let[e,r]of Object.entries(kr)){let i=`weight:${e}`,o=s?jr(s,i):void 0;if(a&&o===void 0)continue;let c=l?jr(l,i):void 0,u=o??c??r,d=Ln(`heading`,At(i),n);t.push(`  ${d} { font-weight: ${u}; }`)}}function jr(e,t){let n=e[t];if(typeof n!=`object`||!n||Array.isArray(n))return;let r=n.fontWeight;return typeof r==`string`&&Cn(r)===null?r:void 0}function Mr(e,t,{resetInheritedPaddingSpecificity:n=!1,surface:r,sink:i=qn,location:a=``}={}){for(let[o,s]of Object.entries(e))for(let[e,c]of Object.entries(s)){let s=Object.entries(c);if(s.length===0)continue;let l=Ln(o,At(e),r),u=br(a,o,e),d=[],f=[];for(let[e,t]of s)e.startsWith(`:`)&&typeof t==`object`?f.push([e,t]):Zn(Vn(e),`${t}`,u,i)&&d.push([e,t]);let p=d,m=[],h=!1,g=new Set;for(let[e,t]of d){let n=Mt(o,e),r=Hn.has(e)&&e!==`padding`?Mt(o,`padding`):[];for(let i of[...n,...r])if(i.expand===`container`&&Hn.has(e)&&(h=!0),i.replaces&&g.add(e),i.vars)for(let e of i.vars)m.push([e,t]);e in Un&&Mt(o,`padding`).some(e=>e.expand===`container`)&&(h=!0)}if(h){let e=d.filter(([e])=>Wn.has(e)),t=d.filter(([e])=>!Wn.has(e)),r=Kn(o,Gn(e),n);p=[...t,...r]}if(g.size>0&&(p=p.filter(([e])=>!g.has(e))),m.length>0&&(p=[...p,...m]),p.length>0){let e=Qn(p,u,i,Vn);t.push(`  ${l} {\n${e}\n  }`)}for(let[e,n]of f){let r=Object.entries(n);if(r.length>0){let n=Qn(r,`${u}[${JSON.stringify(e)}]`,i,Vn);t.push(`  ${zn(l,e)} {\n${n}\n  }`)}}}}function Nr(e,t){t.push(`  :where(h1, h2, h3, h4, h5, h6) {
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
  }`)}function Pr(e,t,n){let r={primary:`var(--color-text-primary)`,secondary:`var(--color-text-secondary)`,disabled:`var(--color-text-disabled)`,placeholder:`var(--color-text-secondary)`,accent:`var(--color-text-accent)`,inherit:`inherit`},i=`text`in e,a=`heading`in e,o=`link`in e;if(i||a||o)for(let[e,s]of Object.entries(r))i&&t.push(`  ${Ln(`text`,At(`color:${e}`),n)} { color: ${s}; }`),a&&t.push(`  ${Ln(`heading`,At(`color:${e}`),n)} { color: ${s}; }`),o&&t.push(`  ${Ln(`link`,At(`color:${e}`),n)} { color: ${s}; }`)}var Fr={"4xs":`var(--font-size-4xs)`,"3xs":`var(--font-size-3xs)`,"2xs":`var(--font-size-2xs)`,xsm:`var(--font-size-xs)`,sm:`var(--font-size-sm)`,base:`var(--font-size-base)`,lg:`var(--font-size-lg)`,xl:`var(--font-size-xl)`,"2xl":`var(--font-size-2xl)`,"3xl":`var(--font-size-3xl)`,"4xl":`var(--font-size-4xl)`};function Ir(e,t,n){if(`text`in e)for(let[e,r]of Object.entries(Fr)){let i=At(`size:${e}`);t.push(`  ${Ln(`text`,i,n)} { font-size: ${r}; }`)}}function Lr(e,t){let n=Or(e,t),r=[],i=[];for(let e of n)e.trimStart().startsWith(`:where(`)?r.push(e):i.push(e);return{component:i,prose:r}}function Rr(e){let t=e.__adaptations===void 0?void 0:re(e.name,e.__adaptations,void 0);return Array.isArray(e.__adaptationRules)?e.__adaptationRules:t===void 0?[]:Ae(e.name,t,e.__axes??{},e.tokens,e.localTokens,e.components)??[]}function zr(e,t){let n=[],r=Nn(e.name),i=Jn(t),{rules:a,rootAliases:o}=Er(e),s=new Map;for(let t of[`dark`,`light`]){let r=t===`dark`?e.__onDark:e.__onLight;if(!r)continue;let c=t===`dark`?`onDark.`:`onLight.`,l=vr(r.tokens,e.tokens,e.localTokens,a,o);s.set(t,l);let u=[...Object.entries(r.tokens),...l];if(u.length>0){let e=Qn(u,`${c}tokens`,i);n.push(`  ${Fn(t)} {\n${e}\n  }`)}r.components&&(Mr(r.components,n,{resetInheritedPaddingSpecificity:!0,surface:t,sink:i,location:c}),Pr(r.components,n,t),Ir(r.components,n,t),Ar(r.components,n,{surface:t,inheritedComponents:e.components}))}let c=[];for(let[t,n]of a.entries()){let a=mr(n.tokens);if(a.length===0)continue;let o=[];for(let n of[`dark`,`light`]){if(!(n===`dark`?e.__onDark:e.__onLight))continue;let r=new Set((s.get(n)??[]).map(([e])=>e)),c=a.filter(([e])=>r.has(e)).map(([e,t])=>[e,gr(e,t)]);if(c.length===0)continue;let l=Qn(c,`adaptations[${t}].on${n}.statusRoles.`,i);o.push(`  ${Fn(n)} {\n${l}\n  }`)}o.length>0&&c.push(Hr(n.query,o,r))}if(n.length===0&&c.length===0)return``;let l=[];if(n.length>0){let e=n.join(`

`);l.push(`@scope (${r}) to (${Pn}) {\n${e}\n}`)}return l.push(...c),l.join(`

`)}function Br(e,t,n,r){let i=[],a=yr(e,t,n,[...Object.entries(e.tokens),...r]);return a!==null&&i.push(a),e.components&&(Mr(e.components,i,{resetInheritedPaddingSpecificity:!0,sink:n,location:t}),Pr(e.components,i),Ir(e.components,i)),i}function Vr(e){let t=e?.heading;return typeof t==`object`&&!!t&&!Array.isArray(t)}function Hr(e,t,n){return`@media ${e} {\n  @scope (${n}) to (${Pn}) {\n${t.map(Wr).join(`

`)}\n  }\n}`}function Ur(e,t){let n=Jn(t),r=Er(e),{rules:i}=r;if(i.length===0)return{prose:``,component:``};let a=Nn(e.name),o=[],s=i.some(e=>Vr(e.components)),c=Dr(r);for(let[e,t]of i.entries()){let r=Br(t,`adaptations[${e}].`,n,c[e]);r.length!==0&&o.push(Hr(t.query,r,a))}for(let[e,t]of i.entries()){let r=mr(t.tokens);if(r.length===0)continue;let i=yr({tokens:{}},`adaptations[${e}].statusRoles.`,n,r);i!==null&&o.push(Hr(t.query,[i],a))}if(s){let t=[];Ar(e.components??{},t,{forceAll:!0}),o.push(`@scope (${a}) to (${Pn}) {\n${t.join(`

`)}\n}`);for(let e of i){let t=[];Ar(e.components??{},t,{authoredOnly:!0}),t.length>0&&o.push(Hr(e.query,t,a))}}return{prose:``,component:o.join(`

`)}}function Wr(e){return`  ${e}`}function Gr(e,t){if(Sr?.theme===e)return Kr(e,t);let n=Sr;Sr=Tr(e);try{return Kr(e,t)}finally{Sr=n}}function Kr(e,t){let{component:n,prose:r}=Lr(e,t),i=Nn(e.name),a=Pn,o=``;r.length>0&&(o=`@scope (${i}) to (${a}) {\n${r.join(`

`)}\n}`);let s=``;n.length>0&&(s=`@scope (${i}) to (${a}) {\n${n.join(`

`)}\n}`);let c=Ur(e,t);c.component&&(s=s?`${s}\n\n${c.component}`:c.component);let l=zr(e,t);return l&&(s=s?`${s}\n\n${l}`:l),{prose:o,component:s}}var qr={name:`arrow-down`,size:24,node:[[`path`,{d:`M12 5v14`,key:`s699le`}],[`path`,{d:`m19 12-7 7-7-7`,key:`1idqje`}]]};qr.node;var Jr=j(qr),Yr={name:`arrow-up-down`,size:24,node:[[`path`,{d:`m21 16-4 4-4-4`,key:`f6ql7i`}],[`path`,{d:`M17 20V4`,key:`1ejh1v`}],[`path`,{d:`m3 8 4-4 4 4`,key:`11wl7u`}],[`path`,{d:`M7 4v16`,key:`1glfcx`}]]};Yr.node;var Xr=j(Yr),Zr={name:`arrow-up`,size:24,node:[[`path`,{d:`m5 12 7-7 7 7`,key:`hav0vg`}],[`path`,{d:`M12 19V5`,key:`x0mq9r`}]]};Zr.node;var Qr=j(Zr),$r={name:`calendar`,size:24,node:[[`path`,{d:`M8 2v3`,key:`1ioesn`}],[`path`,{d:`M16 2v3`,key:`otl347`}],[`rect`,{x:`3`,y:`3`,width:`18`,height:`18`,rx:`2`,key:`h1oib`}],[`path`,{d:`M3 9h18`,key:`1pudct`}]]};$r.node;var ei=j($r),ti={name:`check-check`,size:24,node:[[`path`,{d:`M18 6 7 17l-5-5`,key:`116fxf`}],[`path`,{d:`m22 10-7.5 7.5L13 16`,key:`ke71qq`}]]};ti.node;var ni=j(ti),ri={name:`check`,size:24,node:[[`path`,{d:`M20 6 9 17l-5-5`,key:`1gmf2c`}]]};ri.node;var ii=j(ri),ai={name:`chevron-down`,size:24,node:[[`path`,{d:`m6 9 6 6 6-6`,key:`qrunsl`}]]};ai.node;var oi=j(ai),si={name:`chevron-left`,size:24,node:[[`path`,{d:`m15 18-6-6 6-6`,key:`1wnfg3`}]]};si.node;var ci=j(si),li={name:`chevron-right`,size:24,node:[[`path`,{d:`m9 18 6-6-6-6`,key:`mthhwq`}]]};li.node;var ui=j(li),di={name:`chevrons-left`,size:24,node:[[`path`,{d:`m11 17-5-5 5-5`,key:`13zhaf`}],[`path`,{d:`m18 17-5-5 5-5`,key:`h8a8et`}]]};di.node;var fi=j(di),pi={name:`chevrons-right`,size:24,node:[[`path`,{d:`m6 17 5-5-5-5`,key:`xnjwq`}],[`path`,{d:`m13 17 5-5-5-5`,key:`17xmmf`}]]};pi.node;var mi=j(pi),hi={name:`circle-check-big`,size:24,node:[[`path`,{d:`M21.801 10A10 10 0 1 1 17 3.335`,key:`yps3ct`}],[`path`,{d:`m9 11 3 3L22 4`,key:`1pflzl`}]],aliases:[`check-circle`]};hi.node;var gi=j(hi),_i={name:`circle-x`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`10`,key:`1mglay`}],[`path`,{d:`m15 9-6 6`,key:`1uzhvr`}],[`path`,{d:`m9 9 6 6`,key:`z0biqf`}]],aliases:[`x-circle`]};_i.node;var vi=j(_i),yi={name:`clock`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`10`,key:`1mglay`}],[`path`,{d:`M12 6v6l4 2`,key:`mmk7yg`}]]};yi.node;var bi=j(yi),xi={name:`columns-2`,size:24,node:[[`rect`,{width:`18`,height:`18`,x:`3`,y:`3`,rx:`2`,key:`afitv7`}],[`path`,{d:`M12 3v18`,key:`108xh3`}]],aliases:[`columns`]};xi.node;var Si=j(xi),Ci={name:`ellipsis`,size:24,node:[[`circle`,{cx:`12`,cy:`12`,r:`1`,key:`41hilf`}],[`circle`,{cx:`19`,cy:`12`,r:`1`,key:`1wjl8i`}],[`circle`,{cx:`5`,cy:`12`,r:`1`,key:`1pcz8c`}]],aliases:[`more-horizontal`]};Ci.node;var wi=j(Ci),Ti={name:`external-link`,size:24,node:[[`path`,{d:`M15 3h6v6`,key:`1q9fwt`}],[`path`,{d:`M10 14 21 3`,key:`gplh6r`}],[`path`,{d:`M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6`,key:`a6xqqp`}]]};Ti.node;var Ei=j(Ti),Di={name:`eye-off`,size:24,node:[[`path`,{d:`M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49`,key:`ct8e1f`}],[`path`,{d:`M14.084 14.158a3 3 0 0 1-4.242-4.242`,key:`151rxh`}],[`path`,{d:`M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143`,key:`13bj9a`}],[`path`,{d:`m2 2 20 20`,key:`1ooewy`}]]};Di.node;var Oi=j(Di),ki={name:`funnel`,size:24,node:[[`path`,{d:`M10 20a1 1 0 0 0 .553.895l2 1A1 1 0 0 0 14 21v-7a2 2 0 0 1 .517-1.341L21.74 4.67A1 1 0 0 0 21 3H3a1 1 0 0 0-.742 1.67l7.225 7.989A2 2 0 0 1 10 14z`,key:`sc7q7i`}]],aliases:[`filter`]};ki.node;var Ai=j(ki),ji={name:`mic`,size:24,node:[[`path`,{d:`M12 19v3`,key:`npa21l`}],[`path`,{d:`M19 10v2a7 7 0 0 1-14 0v-2`,key:`1vc78b`}],[`rect`,{x:`9`,y:`2`,width:`6`,height:`13`,rx:`3`,key:`s6n7sd`}]]};ji.node;var Mi=j(ji),Ni={name:`square`,size:24,node:[[`rect`,{width:`18`,height:`18`,x:`3`,y:`3`,rx:`2`,key:`afitv7`}]]};Ni.node;var Pi=j(Ni),Fi={name:`upload`,size:24,node:[[`path`,{d:`M12 3v12`,key:`1x0j5s`}],[`path`,{d:`m17 8-5-5-5 5`,key:`7q97r8`}],[`path`,{d:`M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4`,key:`ih7n3h`}]]};Fi.node;var Ii=j(Fi),Li={name:`wrench`,size:24,node:[[`path`,{d:`M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.106-3.105c.32-.322.863-.22.983.218a6 6 0 0 1-8.259 7.057l-7.91 7.91a1 1 0 0 1-2.999-3l7.91-7.91a6 6 0 0 1 7.057-8.259c.438.12.54.662.219.984z`,key:`1ngwbx`}]]};Li.node;var Ri=j(Li),zi=new Map,Bi=new Set,Vi=0,Hi=!1;function Ui(){if(Vi>0)Hi=!0;else for(let e of[...Bi])try{e()}catch(e){queueMicrotask(()=>{throw e})}}function Wi(e){Vi+=1;try{e()}finally{--Vi,Vi===0&&Hi&&(Hi=!1,Ui())}}function Gi(e,t){return zi.set(e,t),Ui(),()=>{zi.get(e)===t&&(zi.delete(e),Ui())}}var Ki=`--astryx-presented`;function qi(e){let t=Number(e.style.getPropertyValue(Ki))||0;e.style.setProperty(Ki,String(t+1))}function Ji(e){e.style.removeProperty(Ki)}var Yi=`data-astryx-reduced-motion`;function Xi(e){return e?{[Yi]:``}:{}}function Zi(e,t){t?e.setAttribute(Yi,``):e.removeAttribute(Yi)}function Qi(e){return e?.closest?.(`[${Yi}]`)!=null}var $i={"--duration-fast-min":`130ms`,"--duration-fast":`175ms`,"--duration-fast-max":`230ms`,"--duration-medium-min":`310ms`,"--duration-medium":`410ms`,"--duration-medium-max":`550ms`,"--duration-slow-min":`730ms`,"--duration-slow":`975ms`,"--duration-slow-max":`1300ms`,"--duration-exit":`var(--duration-fast)`,"--ease-standard":`cubic-bezier(0.24, 1, 0.4, 1)`,"--ease-entrance":`cubic-bezier(0.24, 1, 0.4, 1)`,"--ease-move":`cubic-bezier(0.24, 1, 0.4, 1)`,"--ease-exit":`cubic-bezier(0.3, 0, 0.6, 0.6)`};function ea(e,t,{jumps:n=!1}={}){let r=n=>e<=0||n>=e?1:t(Math.max(0,n)/e),i=Math.min(.001,e/100);return{duration:e,at:r,velocity:t=>{if(e<=0||t>=e)return 0;let a=Math.max(0,t-i),o=Math.min(e,t+i);if(!n)return(r(o)-r(a))/(o-a);let s=r(t),c=[];return t>a&&c.push((s-r(a))/(t-a)),o>t&&c.push((r(o)-s)/(o-t)),c.reduce((e,t)=>Math.abs(t)<Math.abs(e)?t:e)}}}function ta(e){return a(`duration`,e.transitionDuration.split(`,`)[0]??``)}function na(e){return e.transitionTimingFunction.split(/,(?![^(]*\))/)[0]??``}function ra(e){return c(na(e))}function V(e,t){let n=e.defaultView;if(typeof n?.Animation!=`function`||typeof n.KeyframeEffect!=`function`)return null;let r;try{r=new n.Animation(new n.KeyframeEffect(null,null,{duration:1e3,easing:t,fill:`both`}),null)}catch{return null}return e=>(r.currentTime=Math.min(1,Math.max(0,e))*1e3,r.effect?.getComputedTiming().progress??1)}function ia(e,{ms:t,points:n},r){if(t==null||t<=0)return null;if(n!=null)return ea(t/1e3,o(n));let i=V(e,r);return i==null?null:ea(t/1e3,i,{jumps:!0})}function aa(e,{duration:t,easing:n}){let r=t==null?null:a(`duration`,t),i=n==null?null:c(n),o=e.ownerDocument,s=o.defaultView,l=t!=null&&r==null,u=n!=null&&i==null;if(s==null||!l&&!u)return{ms:r,points:i};let d=o.createElement(`span`);d.style.setProperty(`display`,`none`,`important`);let f=(e,t)=>(d.style.setProperty(e,t,`important`),d.style.getPropertyValue(e)!==``),p=l&&f(`transition-duration`,t),m=u&&f(`transition-timing-function`,n);(o.body??o.documentElement).append(d);try{let e=s.getComputedStyle(d);return{ms:p?ta(e):r,points:m?ra(e):i}}finally{d.remove()}}function oa(e,t){let n=e.getPropertyValue(t).trim();if(n!==``)return n;let r={};for(let[t,n]of Object.entries($i))r[t]=e.getPropertyValue(t).trim()||n;return b(r)[t]??``}function sa(e,t){let n=e.ownerDocument.defaultView?.getComputedStyle(e);return n==null?null:oa(n,t)}function ca(e,t){let n=e.ownerDocument,r=n.defaultView?.getComputedStyle(e);if(r==null)return null;if(t!=null){let i=oa(r,t.easing);return ia(n,aa(e,{duration:oa(r,t.duration),easing:i}),i)}let i=na(r);return ia(n,{ms:ta(r),points:c(i)},i)}var la=1500;function ua(e){return e.getElementsByTagName(`*`).length<=la}function da(e,t){if(e.width===0||e.height===0)return!0;let n=e.getContext(`2d`,{willReadFrequently:!0});if(n==null)return!0;try{n.drawImage(t,0,0);let{data:r}=n.getImageData(0,0,e.width,e.height);for(let e=3;e<r.length;e+=4)if(r[e]!==0)return!0;return!1}catch{return!1}}var fa=`color.color-scheme.cursor.direction.font-family.font-feature-settings.font-size.font-stretch.font-style.font-variant.font-variation-settings.font-weight.hyphens.letter-spacing.line-height.list-style-type.overflow-wrap.tab-size.text-align.text-indent.text-shadow.text-transform.visibility.white-space.word-break.word-spacing.writing-mode`.split(`.`),pa=new WeakMap;function ma(e){let t=parseFloat(e.getPropertyValue(`line-height`)),n=parseFloat(e.getPropertyValue(`font-size`));return Number.isFinite(t)&&n>0?Math.round(t/n*1e3)/1e3:null}function ha(e){let t=new Map,n=getComputedStyle(e);for(let e of fa)t.set(e,n.getPropertyValue(e));if(typeof e.computedStyleMap==`function`){let n=e.computedStyleMap().get(`line-height`);n!=null&&`unit`in n&&n.unit===`number`&&`value`in n&&t.set(`line-height`,String(n.value))}else{let e=ma(n);e!=null&&t.set(`line-height`,String(e))}for(let e=0;e<n.length;e++){let r=n[e];r.startsWith(`--`)&&t.set(r,n.getPropertyValue(r))}return t}var ga=`button, fieldset, input, object, output, select, textarea`;function _a(e,t){return e.localName.includes(`-`)?t?.get(e.localName)?.formAssociated===!0:!1}var va=new WeakMap,ya=new WeakMap;function ba(e){let t=e.cloneNode(!0),n=[e,...e.querySelectorAll(`*`)],r=[t,...t.querySelectorAll(`*`)],i=[];n.forEach((e,t)=>{(e.scrollTop!==0||e.scrollLeft!==0)&&i.push([r[t],e.scrollTop,e.scrollLeft])}),va.set(t,i),pa.set(t,ha(e)),ya.set(t,{saved:xt(e),copyOf:new Map(n.map((e,t)=>[e,r[t]]))});for(let e of r)for(let t of[`id`,`data-testid`,`name`,`for`,`autofocus`])e.removeAttribute(t);let a=e.ownerDocument.defaultView;n.forEach((e,t)=>{let n=a?.getComputedStyle(e).getPropertyValue(`anchor-name`);n!=null&&n!==``&&n!==`none`&&r[t].style?.setProperty(`anchor-name`,`none`)});let o=e.ownerDocument.defaultView?.customElements;for(let e of r)(e.matches(ga)||_a(e,o))&&e.setAttribute(`form`,``);for(let e of t.querySelectorAll(`iframe, video, audio, source, embed, object`))for(let t of[`src`,`srcdoc`,`data`,`autoplay`])e.removeAttribute(t);let s=e.querySelectorAll(`canvas`);return Array.from(t.querySelectorAll(`canvas`)).every((e,t)=>da(e,s[t]))?(t.setAttribute(`aria-hidden`,`true`),t.setAttribute(`inert`,``),t.style.pointerEvents=`none`,t):null}function xa(e){for(let[t,n,r]of va.get(e)??[])typeof t.scrollTo==`function`?t.scrollTo({top:n,left:r,behavior:`instant`}):(t.scrollTop=n,t.scrollLeft=r)}function Sa(e){for(let t=e;t;t=t.parentElement){let e=getComputedStyle(t).backgroundColor;if(e!==`transparent`&&!/^rgba\(.*,\s*0\)$/.test(e))return e}return`Canvas`}function Ca(e,t){let n=getComputedStyle(t);for(let[t,r]of pa.get(e)??[])r!==``&&(t===`line-height`||n.getPropertyValue(t)!==r)&&e.style.setProperty(t,r)}function wa(e,t,n,r=null){Object.assign(e.style,{position:`absolute`,boxSizing:`border-box`,margin:`0px`,left:`0px`,top:`0px`,width:`${n.width}px`,height:`${n.height}px`}),t.insertBefore(e,r),Ca(e,t);let i=e.getBoundingClientRect(),a=Math.abs(e.offsetWidth-n.width)<1?n.width:e.offsetWidth,o=Math.abs(e.offsetHeight-n.height)<1?n.height:e.offsetHeight,s=i.width>0&&a>0?i.width/a:1,c=i.height>0&&o>0?i.height/o:1;e.style.width=`${n.width/s}px`,e.style.height=`${n.height/c}px`,e.style.left=`${(n.left-i.left)/s}px`,e.style.top=`${(n.top-i.top)/c}px`,xa(e);let l=ya.get(e);l!=null&&(ya.delete(e),Ct(l.saved,e,e=>l.copyOf.get(e)??null))}function Ta(e){let t=ba(e);if(t==null)return null;t.removeAttribute(`role`),t.removeAttribute(`tabindex`),Object.assign(t.style,{position:`absolute`,top:`${-e.scrollTop}px`,left:`${-e.scrollLeft}px`,boxSizing:`border-box`,width:`${e.clientWidth}px`,height:`auto`,padding:getComputedStyle(e).padding,borderWidth:`0px`,margin:`0px`,overflow:`visible`,translate:`none`});let n=e.ownerDocument.createElement(`div`);n.setAttribute(`aria-hidden`,`true`),n.setAttribute(`inert`,``),Object.assign(n.style,{overflow:`hidden`,pointerEvents:`none`}),n.appendChild(t),va.set(n,(va.get(t)??[]).filter(([e])=>e!==t));let r=pa.get(t);r!=null&&pa.set(n,r);let i=ya.get(t);i!=null&&(ya.set(n,i),ya.delete(t));let a=e.getBoundingClientRect();return{copy:n,rect:DOMRect.fromRect({x:a.left+e.clientLeft,y:a.top+e.clientTop,width:e.clientWidth,height:e.clientHeight})}}var Ea=`http://www.w3.org/2000/svg`;function Da(e){let t=e.ownerDocument.createTreeWalker(e,4),n=[];for(;t.nextNode();)t.currentNode.textContent?.trim()&&n.push(t.currentNode);let r=new Set;for(let t of n){let n=t.parentElement;if(n?.closest(`style, script, textarea, title, noscript`)!=null)continue;if(n?.namespaceURI===Ea){let e=n.closest(`text`);e!=null&&r.add(e);continue}let i=e.ownerDocument.createElement(`span`);t.replaceWith(i),i.appendChild(t),r.add(i)}return[...r]}function Oa(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Malformed settings response.`);return e}function ka(e){if(typeof e!=`string`||!/^[0-9a-f]{64}$/.test(e))throw Error(`Malformed settings fingerprint.`);return e}function Aa(e){let t=Oa(e);if(!Array.isArray(t.schema)||t.schema.length>256)throw Error(`Malformed settings schema.`);let n=new Set;return{schema:t.schema.map(e=>{let t=Oa(e);if(typeof t.name!=`string`||n.has(t.name)||typeof t.type!=`string`||![`bool`,`int`,`int_or_null`,`float`,`str`,`str_or_null`,`array`,`object`,`object_or_null`].includes(t.type)||typeof t.mutable!=`boolean`||typeof t.help!=`string`||!(t.allowed===null||Array.isArray(t.allowed)&&t.allowed.every(e=>typeof e==`string`))||t.reason!==void 0&&typeof t.reason!=`string`||!Object.hasOwn(t,`default`))throw Error(`Malformed setting specification.`);return n.add(t.name),t}),current:Oa(t.current),fingerprint:ka(t.fingerprint)}}function ja(e){let t=Oa(e);if(!Array.isArray(t.rejected)||!t.rejected.every(e=>{let t=Oa(e);return typeof t.name==`string`&&typeof t.reason==`string`}))throw Error(`Malformed settings rejection list.`);return{applied:Oa(t.applied),rejected:t.rejected,current:Oa(t.current),fingerprint:ka(t.fingerprint)}}function Ma(e){let t=Oa(e),n=Oa(t.default_generation_settings),r=e=>typeof e==`number`&&Number.isSafeInteger(e)&&e>0?e:null;return{nCtx:r(n.n_ctx),kvCacheMode:typeof t.kv_cache_mode==`string`&&t.kv_cache_mode.length<=64&&/^[a-z0-9+_-]+$/.test(t.kv_cache_mode)?t.kv_cache_mode:null,totalSlots:r(t.total_slots),geometry:t.geometry===void 0?null:Oa(t.geometry)}}function Na(e,t){if(!e.mutable)throw Error(e.reason??`Restart required.`);let n=t;if(![`str`,`str_or_null`].includes(e.type)||e.type===`str_or_null`&&t===`null`)try{n=JSON.parse(t)}catch{throw Error(`Enter a valid ${e.type} value.`)}let r=e.type.endsWith(`_or_null`),i=e.type.replace(`_or_null`,``);if(!(r&&n===null)&&(i===`bool`?typeof n!=`boolean`:i===`int`?typeof n!=`number`||!Number.isSafeInteger(n):i===`float`?typeof n!=`number`||!Number.isFinite(n):i===`str`?typeof n!=`string`:i===`array`?!Array.isArray(n):typeof n!=`object`||!n||Array.isArray(n)))throw Error(`Expected ${e.type}.`);if(e.allowed!==null&&(typeof n!=`string`||!e.allowed.includes(n)))throw Error(`Allowed values: ${e.allowed.join(`, `)}.`);return n}function Pa(e,t){return typeof t==`string`&&[`str`,`str_or_null`].includes(e.type)?t:JSON.stringify(t)??``}var Fa=7;function Ia(e){if(!Number.isFinite(e))return String(e);let t=Math.fround(e);for(let e=1;e<Fa;e+=1){let n=Number(t.toPrecision(e));if(Math.fround(n)===t)return String(n)}return String(Number(t.toPrecision(Fa)))}function La(e,t){if(t==null)return``;let n=e.type.replace(`_or_null`,``);return n===`float`&&typeof t==`number`?Ia(t):n===`int`&&typeof t==`number`?String(t):Pa(e,t)}function Ra(e){let t=Oa(e);if(!Array.isArray(t.tokens)||!t.tokens.every(e=>typeof e==`number`&&Number.isInteger(e)))throw Error(`Malformed tokenization response.`);return t.tokens.length}var za=1048576,Ba=class{decoder=new TextDecoder(`utf-8`);maxFrameBytes;onMessage;onDone;buffered=``;eventName=``;dataLines=[];id=null;retry=null;frameBytes=0;ended=!1;get done(){return this.ended}constructor(e){this.maxFrameBytes=e.maxFrameBytes??za,this.onMessage=e.onMessage,this.onDone=e.onDone}push(e){if(!this.ended){if(this.frameBytes+=e.byteLength,this.frameBytes>this.maxFrameBytes)throw Error(`SSE frame exceeded the configured byte limit.`);this.buffered+=this.decoder.decode(e,{stream:!0}),this.drainLines(!1)}}close(){if(this.ended)return;let e=this.decoder.decode();e.length>0&&(this.buffered+=e),this.drainLines(!0),this.buffered.length>0&&(this.consumeLine(this.buffered),this.buffered=``),this.dispatch()}drainLines(e){for(;this.buffered.length>0;){let t=this.buffered.indexOf(`
`),n=this.buffered.indexOf(`\r`),r;if(r=t===-1?n:n===-1?t:Math.min(t,n),r===-1||this.buffered[r]===`\r`&&this.buffered[r+1]===void 0&&!e)break;let i=this.buffered.slice(0,r),a=this.buffered[r]===`\r`&&this.buffered[r+1]===`
`?r+2:r+1;this.buffered=this.buffered.slice(a),this.consumeLine(i)}e&&this.buffered.length===0&&this.dispatch()}consumeLine(e){if(e.length===0){this.dispatch();return}if(e.startsWith(`:`))return;let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?``:e.slice(t+1),i=r.startsWith(` `)?r.slice(1):r;n===`event`?this.eventName=i:n===`data`?this.dataLines.push(i):n===`id`?this.id=i:n===`retry`&&/^\d+$/.test(i)&&(this.retry=Number(i))}dispatch(){if(this.dataLines.length===0){this.eventName=``,this.retry=null,this.frameBytes=0;return}let e=this.dataLines.join(`
`),t=this.eventName.length===0?`message`:this.eventName;this.eventName=``,this.dataLines=[],this.frameBytes=0,e===`[DONE]`?(this.ended=!0,this.onDone?.()):(this.onMessage({event:t,data:e,id:this.id,retry:this.retry}),this.retry=null)}};function Va(e){return Array.from(e).some(e=>{let t=e.charCodeAt(0);return t<=32||t>=127&&t<=159})}function Ha(e){if(/[\\?#]/.test(e)||Va(e)||e.includes(`//`))throw Error(`WebUI API base must not contain a query, hash, backslash, whitespace, controls or empty path segments.`);for(let t of e.split(`/`)){let e;try{e=decodeURIComponent(t)}catch{throw Error(`WebUI API base contains an invalid path escape.`)}if(e===`.`||e===`..`)throw Error(`WebUI API base must not contain dot path segments.`);if(/[/\\%?#]/.test(e)||Va(e))throw Error(`WebUI API base contains an ambiguous encoded path segment.`)}}function Ua(e){let t=e??Ga();if(t===``)return``;if(!t.startsWith(`/`)||t.startsWith(`//`))throw Error(`WebUI API base must be a same-origin absolute path.`);return Ha(t),new URL(t,`http://webui.invalid`).pathname.replace(/\/$/,``)}function Wa(e){return/^(.*)\/webui(?:\/(?:index\.html)?)?$/.exec(e)?.[1]??null}function Ga(){if(typeof document>`u`)return``;let e=document.querySelector(`meta[name="mlxcel-ui-api-base"]`)?.content;if(e!==void 0&&e.length>0)return Ua(e);let t=document.querySelector(`base`)?.getAttribute(`href`);if(t!=null&&t.length>0){let e=t.replace(/^[a-z][a-z\d+.-]*:\/\/[^/]*/i,``);Ha(e===`.`?``:e.replace(/^\.\//,``));let n=new URL(t,document.URL);if(n.origin!==window.location.origin||n.username||n.password||n.search||n.hash)throw Error(`WebUI document base must stay on the current origin without credentials, query or hash.`);return Ua(Wa(n.pathname)??n.pathname)}let n=window.location.pathname,r=Wa(n);return r===null?``:(Ha(n),Ua(r))}function Ka(e,t,n){if(!t.startsWith(`/ui-api/v1/`)&&t!==`/v1/chat/completions`&&t!==`/v1/responses`&&t!==`/settings`&&t!==`/props`&&t!==`/tokenize`)throw Error(`WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.`);let r=new URLSearchParams;for(let[e,t]of Object.entries(n??{}))t!=null&&r.set(e,String(t));return`${e}${t}${r.size===0?``:`?${r.toString()}`}`}function qa(e){if(e.length===0)throw Error(`Opaque path segment must not be empty.`);return encodeURIComponent(e)}var Ja=`{
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
`,H=class extends Error{path;constructor(e,t){super(`${e}: ${t}`),this.path=e,this.name=`ValidationError`}},Ya=Qa(Ja);function Xa(e,t,n=`$`){let r=uo(Ya,[`components`,`schemas`],`#/components/schemas`)[e];if(r===void 0)throw new H(n,`unknown schema ${e}`);$a(r,t,n)}function Za(e){return JSON.parse(e)}function Qa(e){return lo(Za(e),`$contract`)}function $a(e,t,n){let r=lo(e,`${n}#schema`);if(typeof r.$ref==`string`){$a(so(r.$ref),t,n);return}if(r.anyOf!==void 0){if(!Array.isArray(r.anyOf)||!r.anyOf.some(e=>co(e,t,n)))throw new H(n,`did not match any allowed schema`);return}if(r.oneOf!==void 0){if(!Array.isArray(r.oneOf))throw new H(n,`schema oneOf must be an array`);let e=r.oneOf.filter(e=>co(e,t,n)).length;if(e!==1)throw new H(n,`matched ${e} oneOf schemas`);return}if(r.const!==void 0&&t!==r.const)throw new H(n,`expected constant ${String(r.const)}`);if(Array.isArray(r.enum)&&!r.enum.some(e=>e===t))throw new H(n,`unexpected enum value ${String(t)}`);let i=r.type;if(Array.isArray(i)){if(!i.some(e=>oo(e,t)))throw new H(n,`expected one of ${i.join(`, `)}`)}else if(typeof i==`string`&&!oo(i,t))throw new H(n,`expected ${i}`);typeof t==`string`&&ro(r,t,n),typeof t==`number`&&ao(r,t,n),Array.isArray(t)&&to(r,t,n),fo(t)&&eo(r,t,n)}function eo(e,t,n){let r=po(e.required,`${n}#schema.required`,!1),i=e.properties===void 0?{}:lo(e.properties,`${n}#schema.properties`);for(let e of r)if(!(e in t))throw new H(`${n}.${e}`,`missing required property`);if(e.maxProperties!==void 0&&Object.keys(t).length>mo(e.maxProperties,`${n}#schema.maxProperties`))throw new H(n,`too many object properties`);for(let[r,a]of Object.entries(t)){let t=i[r];if(t!==void 0)$a(t,a,`${n}.${r}`);else if(e.additionalProperties===!1)throw new H(`${n}.${r}`,`unexpected property`);else fo(e.additionalProperties)&&$a(e.additionalProperties,a,`${n}.${r}`)}}function to(e,t,n){if(e.minItems!==void 0&&t.length<mo(e.minItems,`${n}#schema.minItems`))throw new H(n,`array shorter than minimum`);if(e.maxItems!==void 0&&t.length>mo(e.maxItems,`${n}#schema.maxItems`))throw new H(n,`array longer than maximum`);e.items!==void 0&&t.forEach((t,r)=>$a(e.items,t,`${n}[${r}]`))}function no(e){let t=0;for(let n=0;n<e.length;t++)n+=(e.codePointAt(n)??0)>65535?2:1;return t}function ro(e,t,n){let r=no(t);if(e.minLength!==void 0&&r<mo(e.minLength,`${n}#schema.minLength`))throw new H(n,`string shorter than minimum`);if(e.maxLength!==void 0&&r>mo(e.maxLength,`${n}#schema.maxLength`))throw new H(n,`string longer than maximum`);if(typeof e.pattern==`string`&&!new RegExp(e.pattern,`u`).test(t))throw new H(n,`string does not match pattern`);if(e.format===`date-time`&&!io(t))throw new H(n,`invalid RFC3339 date-time`)}function io(e){return/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(e)?!Number.isNaN(Date.parse(e)):!1}function ao(e,t,n){if(!Number.isFinite(t))throw new H(n,`expected finite number`);if(e.type===`integer`&&!Number.isInteger(t))throw new H(n,`expected integer`);if(e.minimum!==void 0&&t<mo(e.minimum,`${n}#schema.minimum`))throw new H(n,`number below minimum`);if(e.maximum!==void 0&&t>mo(e.maximum,`${n}#schema.maximum`))throw new H(n,`number above maximum`)}function oo(e,t){return e===`null`?t===null:e===`array`?Array.isArray(t):e===`object`?fo(t):e===`integer`?typeof t==`number`&&Number.isInteger(t):e===`number`?typeof t==`number`&&Number.isFinite(t):typeof t===e}function so(e){if(!e.startsWith(`#/`))throw new H(`$ref`,`unsupported external ref ${e}`);return e.slice(2).split(`/`).map(e=>e.replace(/~1/g,`/`).replace(/~0/g,`~`)).reduce((t,n)=>lo(t,e)[n],Ya)}function co(e,t,n){try{return $a(e,t,n),!0}catch(e){if(e instanceof H)return!1;throw e}}function lo(e,t){if(!fo(e))throw new H(t,`expected object`);return e}function uo(e,t,n){return t.reduce((e,t)=>lo(e,n)[t],e)}function fo(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function po(e,t,n){if(e===void 0&&!n)return[];if(!Array.isArray(e)||!e.every(e=>typeof e==`string`))throw new H(t,`expected string array`);return e}function mo(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw new H(t,`expected numeric schema bound`);return e}function ho(e,t,n){return Xa(e,t,n),t}function go(e,t=`$`){return ho(`ErrorEnvelope`,e,t)}function _o(e,t=`$`){return ho(`BootstrapResponse`,e,t)}function vo(e,t=`$`){return ho(`CatalogEntry`,e,t)}function yo(e,t=`$`){return ho(`CatalogListResponse`,e,t)}function bo(e,t=`$`){return ho(`OperationsListResponse`,e,t)}function xo(e,t=`$`){return ho(`Operation`,e,t)}function So(e,t=`$`){return ho(`OperationAccepted`,e,t)}function Co(e,t=`$`){return ho(`RuntimeSnapshot`,e,t)}function wo(e,t=`$`){return ho(`UiEvent`,e,t)}var To=class extends Error{status;envelope;constructor(e,t,n){super(n??t?.error.message??`WebUI request failed with HTTP ${e}`),this.status=e,this.envelope=t,this.name=`WebUiHttpError`}},Eo=2097152,Do=class{apiBase;fetchImpl;bearerToken=null;onUnauthorized;controllers=new Set;constructor(e={}){this.apiBase=Ua(e.apiBase),this.fetchImpl=e.fetchImpl??fetch.bind(globalThis),this.onUnauthorized=e.onUnauthorized}setBearerToken(e){this.bearerToken=e}abortAll(){for(let e of this.controllers)e.abort();this.controllers.clear()}async bootstrap(e){return this.request(`/ui-api/v1/bootstrap`,_o,{method:`GET`,signal:e})}async catalog(e={},t){return this.request(`/ui-api/v1/catalog`,yo,{method:`GET`,query:{...e},signal:t})}async catalogEntry(e,t){return this.request(`/ui-api/v1/catalog/${qa(e)}`,vo,{method:`GET`,signal:t})}async refreshCatalog(e,t){return this.request(`/ui-api/v1/catalog/refresh`,So,{method:`POST`,body:{idempotency_key:e},signal:t})}async modelAction(e,t){return this.request(`/ui-api/v1/model-actions`,So,{method:`POST`,body:e,signal:t})}async download(e,t){return this.request(`/ui-api/v1/downloads`,So,{method:`POST`,body:e,signal:t})}async removeModel(e,t){return this.request(`/ui-api/v1/model-removals`,So,{method:`POST`,body:e,signal:t})}async operationsPage(e={},t){return this.request(`/ui-api/v1/operations`,bo,{method:`GET`,query:e,signal:t})}async operations(e){let t=[],n,r=null;for(;;){let i=await this.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null)r=i.server_instance_id;else if(i.server_instance_id!==r)throw Error(`Operations pagination crossed a server restart; refresh the authoritative snapshot.`);if(t.push(...i.items),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operation(e,t){return this.request(`/ui-api/v1/operations/${qa(e)}`,xo,{method:`GET`,signal:t})}async cancelOperation(e,t){return this.request(`/ui-api/v1/operations/${qa(e)}/cancel`,So,{method:`POST`,body:{},signal:t})}async runtime(e,t){let n=await this.request(`/ui-api/v1/runtime`,Co,{method:`GET`,query:{model_id:e,autoload:!1},signal:t});if(n.model_id!==e)throw Error(`Runtime response model_id did not match the requested model.`);return n}async tokenCount(e,t,n){return this.request(`/tokenize`,Ra,{method:`POST`,query:{model:e,autoload:!1},body:{content:t,add_special:!1,parse_special:!0,with_pieces:!1},signal:n})}async settings(e,t){return this.request(`/settings`,Aa,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async patchSettings(e,t,n){return this.request(`/settings`,ja,{method:`PATCH`,query:{model:e,autoload:!1},body:{op:`merge`,values:t},signal:n})}async modelProps(e,t){return this.request(`/props`,Ma,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async events(e,t,n){let r=n?.serverInstanceId!==void 0&&n.serverInstanceId!==null&&n.afterSequence!==null&&n.afterSequence!==void 0,i=r?{server_instance_id:n.serverInstanceId,after_sequence:n.afterSequence}:void 0,a=new Headers({Accept:`text/event-stream`});!r&&n?.lastEventId!==void 0&&n.lastEventId!==null&&a.set(`Last-Event-ID`,n.lastEventId),await this.sse(Ka(this.apiBase,`/ui-api/v1/events`,i),{method:`GET`,signal:t,headers:a},{onDone:e.onDone,onFrame:t=>{t.retry!==null&&e.onRetryAfter?.(t.retry),t.data.length>0&&e.onEvent(wo(Za(t.data)))}})}async chatCompletions(e,t,n,r){await this.sse(Ka(this.apiBase,`/v1/chat/completions`,{autoload:!1}),{method:`POST`,body:Ao(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async responses(e,t,n,r){await this.sse(Ka(this.apiBase,`/v1/responses`,{autoload:!1}),{method:`POST`,body:Ao(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async request(e,t,n){let r=await this.fetchWithAuth(Ka(this.apiBase,e,n.query),n);try{return r.response.ok||await this.throwHttp(r.response,r.signal),t(Za(await this.readBody(r.response,r.signal)))}finally{this.controllers.delete(r.controller),r.cleanup()}}async sse(e,t,n){let r=await this.fetchWithAuth(e,t);try{let e=r.response;if(e.ok||await this.throwHttp(e,r.signal),e.body===null)throw Error(`WebUI event stream response has no body.`);let t=new Ba({onDone:n.onDone,onMessage:n.onFrame}),i=e.body.getReader();try{for(;;){let e=await jo(i,r.signal);if(e.done)break;t.push(e.value)}if(t.close(),!t.done&&r.signal.aborted!==!0)throw Error(`WebUI event stream ended before a DONE frame.`)}catch(e){throw await i.cancel().catch(()=>void 0),e}finally{i.releaseLock()}}finally{this.controllers.delete(r.controller),r.cleanup()}}async fetchWithAuth(e,t){let n=new AbortController,{signal:r,cleanup:i}=Mo(n,t.signal);this.controllers.add(n);let a=new Headers(t.headers);a.set(`Accept`,a.get(`Accept`)??`application/json`),this.bearerToken!==null&&a.set(`Authorization`,`Bearer ${this.bearerToken}`),t.body!==void 0&&a.set(`Content-Type`,`application/json`);try{return{response:await this.fetchImpl(e,{method:t.method,headers:a,body:t.body===void 0?void 0:JSON.stringify(t.body),signal:r,credentials:`same-origin`,cache:`no-store`,redirect:`error`}),controller:n,signal:r,cleanup:i}}catch(e){throw this.controllers.delete(n),i(),e}}async throwHttp(e,t){let n=this.bearerToken;e.status===401&&(this.bearerToken=null,this.abortAll(),this.onUnauthorized?.());let r=null;try{let n=await this.readBody(e,t);n.length>0&&(r=go(Za(n)))}catch{r=null}throw new To(e.status,Oo(r,n),ko(r?.error.message??`WebUI request failed with HTTP ${e.status}`,n))}async readBody(e,t){if(e.body===null)return``;let n=e.body.getReader(),r=[],i=0;try{for(;;){let e=await jo(n,t);if(e.done)break;if(i+=e.value.byteLength,i>Eo)throw Error(`WebUI JSON response exceeded the configured byte limit.`);r.push(e.value)}}catch(e){throw await n.cancel().catch(()=>void 0),e}finally{n.releaseLock()}let a=new Uint8Array(i),o=0;for(let e of r)a.set(e,o),o+=e.byteLength;return new TextDecoder().decode(a)}};function Oo(e,t){return e===null?null:{...e,error:{...e.error,message:ko(e.error.message,t),field_errors:e.error.field_errors?.map(e=>({...e,message:ko(e.message,t)}))}}}function ko(e,t){let n=e.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`).replace(/token[=:]\s*\S+/gi,`token=[redacted]`);return t!==null&&t.length>0&&(n=n.split(t).join(`[redacted]`)),n}function Ao(e,t){if(t.length===0)throw Error(`Inference model id is required for streaming inference.`);if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Streaming inference bodies must be JSON objects.`);return{...e,model:t,stream:!0}}async function jo(e,t){if(t.aborted)throw await e.cancel().catch(()=>void 0),No();let n=null;try{let r=await Promise.race([e.read(),new Promise((r,i)=>{n=()=>{e.cancel().catch(()=>void 0),i(No())},t.addEventListener(`abort`,n,{once:!0})})]);if(t.aborted)throw No();return r}finally{n!==null&&t.removeEventListener(`abort`,n)}}function Mo(e,t){if(t===void 0)return{signal:e.signal,cleanup:()=>void 0};let n=()=>e.abort(t.reason);return t.aborted?n():t.addEventListener(`abort`,n,{once:!0}),{signal:e.signal,cleanup:()=>t.removeEventListener(`abort`,n)}}function No(){return typeof DOMException==`function`?new DOMException(`The WebUI request was aborted.`,`AbortError`):Error(`The WebUI request was aborted.`)}var U=r();function Po(e){return e instanceof To?e.status===401?`wrong-key`:e.status===403?`forbidden`:`generic`:e instanceof H||e instanceof Error&&e.name===`ValidationError`?`schema`:e instanceof TypeError||e instanceof Error&&/fetch|network|offline|connection/i.test(e.message)?`offline`:`generic`}var Fo=new Set([`ready`,`loading`,`draining`,`unloading`]);function Io(e){return Fo.has(e)}function Lo(e,t){let n=e=>e.lifecycle.state===`ready`?0:Io(e.lifecycle.state)?1:2;return n(e)-n(t)||e.identity.display_name.localeCompare(t.identity.display_name)||e.identity.id.localeCompare(t.identity.id)}function Ro(e){return e.catalog.filter(e=>Io(e.lifecycle.state)).sort(Lo)}function zo(e,t){return e.catalog.some(e=>e.identity.id===t)?t:null}var W={model_free:`connection.mode.model_free`,single_model:`connection.mode.single_model`,router_pool:`connection.mode.router_pool`};function G(e,t){return L(e,Object.hasOwn(W,t)?W[t]:`connection.mode.unknown`)}function Bo(e,t){return t.bootstrap===null?L(e,`connection.ready`):L(e,`connection.footer.connected`,{mode:G(e,t.bootstrap.server.mode),version:t.bootstrap.server.build.version,status:Ho(e,t.connection)})}function Vo(e,t){if(t.bootstrap===null)return null;let n=L(e,`format.unknown`),r=t.serverInstanceId??t.bootstrap.server.server_instance_id,i=t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot;return{instance:r||n,sequence:i===null?n:String(i)}}function Ho(e,t){return L(e,{idle:`connection.status.idle`,bootstrapping:`connection.status.bootstrapping`,ready:`connection.status.ready`,streaming:`connection.status.streaming`,polling:`connection.status.polling`,offline:`connection.status.offline`,stale:`connection.status.stale`,unauthorized:`connection.status.unauthorized`,forbidden:`connection.status.forbidden`,"schema-mismatch":`connection.status.schema_mismatch`,error:`connection.status.error`}[t])}function Uo(e){return e.snapshot.auth.status===`authenticated`?e.snapshot.connection===`schema-mismatch`?(0,U.jsx)(Ko,{...e}):(0,U.jsx)(Go,{...e}):(0,U.jsx)(Wo,{...e})}function Wo(e){return e.authFailure===`schema`?(0,U.jsx)(Ko,{...e}):(0,U.jsxs)(`div`,{className:`screen-stack`,children:[(0,U.jsx)(Ce,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:L(e.locale,`connection.prompt.body`),descriptionTestId:R(`connection.prompt.body`)}),(0,U.jsx)(we,{title:L(e.locale,`state.unauthorized.title`),body:L(e.locale,`state.unauthorized.body`),tokenLabel:L(e.locale,`login.token.label`),tokenHelp:L(e.locale,`login.token.help`),submitLabel:L(e.locale,`login.submit`),logoutLabel:L(e.locale,`login.logout`),error:e.authFailure===null?void 0:Yo(e.locale,e.authFailure),busy:e.snapshot.auth.status===`authenticating`,onSubmit:e.onLogin,onLogout:e.onLogout,testId:R(`auth.login`)})]})}function Go(e){let t=e.snapshot.connection===`offline`||e.snapshot.connection===`stale`||e.snapshot.connection===`error`||e.snapshot.connection===`unauthorized`||e.snapshot.connection===`forbidden`;return(0,U.jsxs)(`div`,{className:`screen-stack`,children:[(0,U.jsx)(Ce,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:L(e.locale,`connection.authenticated.body`),descriptionTestId:R(`connection.authenticated.body`)}),(0,U.jsx)(je,{tone:`info`,title:L(e.locale,`connection.authenticated.title`),body:qo(e.locale,e.snapshot)}),t?(0,U.jsx)(je,{title:L(e.locale,`connection.error.title`),body:Jo(e.locale,e.snapshot.connection),action:(0,U.jsx)(de,{onClick:e.onRetry,children:L(e.locale,`common.retry`)}),testId:R(`connection.error.title`)}):null]})}function Ko(e){return(0,U.jsxs)(`div`,{className:`screen-stack`,children:[(0,U.jsx)(Ce,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId}),(0,U.jsx)(ze,{title:L(e.locale,`state.schema_mismatch.title`),body:L(e.locale,`state.schema_mismatch.body`),actionLabel:L(e.locale,`common.reload`),onRecover:e.onRecoverSchema})]})}function qo(e,t){let n=t.bootstrap;return n===null?L(e,`connection.prompt.detail`):L(e,`connection.authenticated.detail`,{mode:G(e,n.server.mode),version:n.server.build.version,status:Ho(e,t.connection),count:t.catalogSequence===null?L(e,`connection.snapshot.pending`):String(t.catalog.length),operations:t.resourceFences.operationsSnapshot===null?L(e,`connection.snapshot.pending`):String(t.operations.size),sequence:Zo(e,t)})}function Jo(e,t){return t===`offline`?L(e,`state.offline.body`):t===`stale`?L(e,`connection.error.stale`):t===`forbidden`?L(e,`connection.error.forbidden`):t===`unauthorized`?L(e,`connection.error.unauthorized`):L(e,`connection.error.generic`)}function Yo(e,t){return t===`wrong-key`?L(e,`login.error.wrong_key`):t===`offline`?L(e,`login.error.offline`):t===`forbidden`?L(e,`login.error.forbidden`):t===`schema`?L(e,`login.error.schema`):L(e,`login.error.generic`)}function Xo(e,t){return L(e,{unloaded:`models.status.unloaded`,loading:`models.status.loading`,ready:`models.status.ready`,draining:`models.status.draining`,unloading:`models.status.unloading`,failed:`models.status.failed`}[t])}function Zo(e,t){return String(t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot??L(e,`connection.snapshot.pending`))}var Qo=[{id:`models`,key:`nav.models`,testId:`command-route-models`},{id:`chat`,key:`nav.chat`,testId:`command-route-chat`},{id:`activity`,key:`nav.activity`,testId:`command-route-activity`},{id:`settings`,key:`nav.settings`,testId:`command-route-settings`},{id:`new-chat`,key:`chat.new_conversation`,testId:`command-new-chat`}];function $o(e,t,n=20){let r=t.trim().toLowerCase();return r===``?[]:e.filter(e=>e.identity.display_name.toLowerCase().includes(r)||e.identity.id.toLowerCase().includes(r)).sort(Lo).slice(0,Math.max(0,n))}function es(e){let[t,n]=(0,z.useState)(``),[r,i]=(0,z.useState)(e.open),a=(0,z.useId)(),o=(0,z.useId)();e.open!==r&&(i(e.open),n(``));let s=t.trim().toLowerCase(),c=Qo.filter(t=>s===``||t.id.includes(s)||L(e.locale,t.key).toLowerCase().includes(s)),l=s===``?e.loaded:$o(e.catalog,s),u=t=>{t.id===`new-chat`?e.onNewChat():e.onNavigate(t.id)};return(0,U.jsxs)(Se,{open:e.open,title:L(e.locale,`command.title`),onClose:e.onClose,testId:`command-dialog`,closeLabel:L(e.locale,`common.close`),children:[(0,U.jsx)(Ie,{label:L(e.locale,`command.search`),value:t,onChange:n,testId:R(`command.search`)}),c.length===0&&l.length===0?(0,U.jsx)(`p`,{"data-testid":R(`command.no_results`),children:L(e.locale,`command.no_results`)}):null,c.length>0?(0,U.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":a,children:[(0,U.jsx)(`h3`,{id:a,children:L(e.locale,`command.section.commands`)}),(0,U.jsx)(`div`,{className:`command-list`,children:c.map(t=>(0,U.jsx)(de,{"data-testid":t.testId,onClick:()=>u(t),children:L(e.locale,t.key)},t.id))})]}):null,l.length>0?(0,U.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":o,children:[(0,U.jsx)(`h3`,{id:o,children:L(e.locale,`command.section.models`)}),(0,U.jsx)(`div`,{className:`command-list`,children:l.map(t=>(0,U.jsx)(de,{className:`command-model`,title:t.identity.display_name,"data-testid":`command-model`,onClick:()=>e.onOpenModel(t.identity.id),children:`${t.identity.display_name} · ${Xo(e.locale,t.lifecycle.state)}`},t.identity.id))})]}):null]})}var ts=(0,z.createContext)(null);ts.displayName=`LayoutAreaContext`;var ns={hasHeader:!1,hasFooter:!1,hasStart:!1,hasEnd:!1},rs=d(ns);rs.displayName=`LayoutSlotsContext`;var is=d(null);is.displayName=`LayoutDividerContext`;var as={center:{kGNEyG:`x6s0dn4`,$$css:!0},end:{kGNEyG:`xuk3077`,$$css:!0},start:{kGNEyG:`x1cy8zhl`,$$css:!0},stretch:{kGNEyG:`x1qjc9v5`,$$css:!0}},os={start:{kjj79g:`x1nhvcw1`,$$css:!0},center:{kjj79g:`xl56j7k`,$$css:!0},end:{kjj79g:`x13a6bvl`,$$css:!0},between:{kjj79g:`x1qughib`,$$css:!0},around:{kjj79g:`x1l1ennw`,$$css:!0},evenly:{kjj79g:`xaw8158`,$$css:!0}},ss={horizontal:{kXwgrk:`x1q0g3np`,$$css:!0},vertical:{kXwgrk:`xdt5ytf`,$$css:!0}},cs={nowrap:{kwnvtZ:`xozqiw3`,$$css:!0},wrap:{kwnvtZ:`x1a02dak`,$$css:!0},"wrap-reverse":{kwnvtZ:`x8hhl5t`,$$css:!0}},ls={stack:{k1xSpc:`x78zum5`,$$css:!0}},us={0:{k1C7PZ:`x1o57wo1`,khm7nJ:`x6yxi7o`,$$css:!0},1:{k1C7PZ:`x1lfs0n9`,khm7nJ:`x1ngg2t4`,$$css:!0},2:{k1C7PZ:`xak3so`,khm7nJ:`x1x7z4sm`,$$css:!0},3:{k1C7PZ:`xewh9hi`,khm7nJ:`x4olc9o`,$$css:!0},4:{k1C7PZ:`xty4p9g`,khm7nJ:`xtx9w7w`,$$css:!0},5:{k1C7PZ:`x1eqhezk`,khm7nJ:`x1iu6piu`,$$css:!0},6:{k1C7PZ:`x3qlgwd`,khm7nJ:`xczp1bk`,$$css:!0},8:{k1C7PZ:`xicv188`,khm7nJ:`xgx0vcf`,$$css:!0},10:{k1C7PZ:`x1p37tyl`,khm7nJ:`x1xpicb7`,$$css:!0},"0.5":{k1C7PZ:`x1kihgfc`,khm7nJ:`x1tw44j4`,$$css:!0},"1.5":{k1C7PZ:`x1thn6ci`,khm7nJ:`xhq53yo`,$$css:!0}};function ds({crossAlign:e,direction:t,gap:n,mainAlign:r,wrap:i}){return[ls.stack,ss[t],n!=null&&us[n],e!=null&&as[e],r!=null&&os[r],i!=null&&cs[i]]}var fs={reset:{kAzted:`x2lwn1j`,k7Eaqz:`xeuugli`,$$css:!0}},ps={center:{kSGwAc:`xamitd3`,$$css:!0},end:{kSGwAc:`xpvyfi4`,$$css:!0},start:{kSGwAc:`xqcrz7y`,$$css:!0},stretch:{kSGwAc:`xkh2ocl`,$$css:!0}},ms={fill:{kzQI83:`x1iyjqo2`,$$css:!0},static:{kzQI83:`x1c4vz4f`,kmuXW:`x2lah0s`,$$css:!0}};function hs({crossAlignSelf:e,size:t}={}){return[fs.reset,ms[t??`static`],e!=null&&ps[e]]}var gs={xo0ftvq:`xo0ftvq`,$$css:!0};function _s(e){if(e==null||typeof e==`number`)return!0;let t=e.trim().toLowerCase();return t.includes(`%`)?!1:t===`0`||/^-?(?:\d+(?:\.\d+)?|\.\d+)[a-z]+$/.test(t)||/^(?:calc|min|max|clamp)\(/.test(t)}var vs={layoutOuter:{keTefX:`xojxgvx`,k71WvV:`x1fcf3bl`,keoZOQ:`x1sa9bsh`,k1K539:`x6h7pi7`,$$css:!0},layoutInner:{"--container-padding-inline-start":`xrhngw9`,"--container-padding-inline-end":`xjsfl84`,"--container-padding-block-start":`x1047aw6`,"--container-padding-block-end":`xax9j7h`,"--layout-content-width":`x15lplax`,"--layout-alignment-width":`x19kr0ht`,"--layout-padding-own-outer-x":`x1vii845`,"--layout-padding-own-outer-y":`xojybgt`,$$css:!0},fill:{kZKoxP:`x12qplqi`,kskxy:`xenllk4`,$$css:!0},auto:{kAzted:`x1us19tq`,$$css:!0},middle:{kUk6DE:`x98rzlu`,kAzted:`x2lwn1j`,"--_layout-flush-block-start":`x1akq5vd`,"--_layout-flush-block-end":`x51rtne`,$$css:!0},middleQuery:{k9g6sI:`xsw3flo`,$$css:!0},singleColumnContent:{kzqmXN:`xh8yej3`,ks0D6T:`xjl2t3p xpgkkid`,kUOVxO:`xvueqy4`,$$css:!0},singlePanelMiddle:{kB7OPa:`x9f619`,kzqmXN:`xh8yej3`,ks0D6T:`xjl2t3p xnpzo02`,kUOVxO:`xvueqy4 xb0m1lw`,$$css:!0},singleStartPanel:{kZCmMZ:`x1vvd0s7`,$$css:!0},singleEndPanel:{kwRFfy:`xxdn8bs`,$$css:!0},fullBleed:{"--layout-padding-own-outer-x":`xbjcssu`,"--layout-padding-own-outer-y":`x1bz2q72`,$$css:!0}},ys={kzqmXN:`xh8yej3`,kUOVxO:`xvueqy4`,$$css:!0},bs={contentWidthVar:e=>[{"--layout-content-width":(typeof e==`number`?`${e}px`:e)==null?typeof e==`number`?`${e}px`:e:`x4906uf`,$$css:!0},{"--x---layout-content-width":(typeof e==`number`?`${e}px`:e)==null?void 0:typeof e==`number`?`${e}px`:e}],contentAlignmentWidthVar:e=>[{"--layout-alignment-width":(typeof e==`number`?`${e}px`:e)==null?typeof e==`number`?`${e}px`:e:`x1b1nz06`,$$css:!0},{"--x---layout-alignment-width":(typeof e==`number`?`${e}px`:e)==null?void 0:typeof e==`number`?`${e}px`:e}],contentWidth:e=>[ys,{ks0D6T:(typeof e==`number`?`${e}px`:e)==null?typeof e==`number`?`${e}px`:e:`xf68679`,$$css:!0},{"--x-maxWidth":(e=>typeof e==`number`?e+`px`:e??void 0)(typeof e==`number`?`${e}px`:e)}]};function xs({area:e,children:t}){return t==null?null:(0,U.jsx)(ts,{value:e,children:t})}var Ss={"x-default-marker":`x-default-marker`,$$css:!0};function Cs({children:e,content:t,contentWidth:n,defaultHasDividers:r,end:i,footer:a,header:o,height:s=`fill`,padding:c,ref:l,start:u,xstyle:d,className:f,style:p,...m}){let h=s===`fill`,g=t??e,_=(0,z.useMemo)(()=>r==null?null:{defaultHasDividers:r},[r]),v=o!=null,y=a!=null,b=u!=null,x=i!=null,S=b&&x,C=b!==x,w=n!=null&&_s(n),T=(0,z.useMemo)(()=>({hasHeader:v,hasFooter:y,hasStart:b,hasEnd:x}),[v,y,b,x]),E=(0,U.jsx)(rs,{value:T,children:(0,U.jsx)(`div`,{...m,ref:l,...tt(Ye(`layout`,{height:s}),He(vs.layoutOuter,h?vs.fill:vs.auto,d),f,p),children:(0,U.jsxs)(`div`,{...He(Ss,vs.layoutInner,...ds({direction:`vertical`}),h?vs.fill:vs.auto,c===0&&vs.fullBleed,c!=null&&gt[c],c!=null&&pt[c],n!=null&&bs.contentWidthVar(n),w&&bs.contentAlignmentWidthVar(n)),children:[(0,U.jsx)(xs,{area:`header`,children:o}),(0,U.jsxs)(`div`,{...He(...ds({direction:`horizontal`}),vs.middle,n!=null&&(!w||S)&&bs.contentWidth(n),w&&!S&&vs.middleQuery,w&&C&&vs.singlePanelMiddle,w&&b&&!x&&vs.singleStartPanel,w&&!b&&x&&vs.singleEndPanel),children:[(0,U.jsx)(xs,{area:`start`,children:u}),(0,U.jsx)(`div`,{...He(...hs({size:`fill`}),w&&!b&&!x&&vs.singleColumnContent),children:(0,U.jsx)(xs,{area:`content`,children:g})}),(0,U.jsx)(xs,{area:`end`,children:i})]}),(0,U.jsx)(xs,{area:`footer`,children:a})]})})});return _==null?E:(0,U.jsx)(is,{value:_,children:E})}Cs.displayName=`Layout`;var ws={header:{kmuXW:`x2lah0s`,$$css:!0},inner:{kB7OPa:`x9f619`,ks0D6T:`xjl2t3p`,kUOVxO:`xvueqy4`,kZCmMZ:`xaku5ci`,kwRFfy:`x19eyx77`,kLKAdn:`xuqi0w`,kGO01o:`xg476vw`,"--container-padding-inline-start":`xt2a87w`,"--container-padding-inline-end":`xhq06x6`,"--container-padding-block-start":`x186o6qk`,"--container-padding-block-end":`xi9ns85`,$$css:!0},fullBleed:{kZCmMZ:`x1c1uobl`,kwRFfy:`xyri2b`,kLKAdn:`xexx8yu`,kGO01o:`x18d9i69`,"--container-padding-inline-start":`xrhngw9`,"--container-padding-inline-end":`xjsfl84`,"--container-padding-block-start":`x1047aw6`,"--container-padding-block-end":`xax9j7h`,$$css:!0},divider:{kt9PQ7:`xso031l`,kfdmCh:`x1q0q8m5`,kL6WhQ:`xw8gpjh`,$$css:!0}},Ts={sizing:e=>[{kZKoxP:e==null?e:`x16ye13r`,$$css:!0},{"--x-height":(e=>typeof e==`number`?e+`px`:e??void 0)(e)}]};function Es({children:e,hasDivider:t,height:n,label:r,padding:i,paddingBlockEnd:a,role:o,xstyle:s,className:c,style:l,ref:u,...d}){let f=(0,z.use)(is),p=t??f?.defaultHasDividers??!1,m=i===0;return(0,U.jsx)(`div`,{ref:u,role:o,"aria-label":r,"data-divider":p||void 0,...tt(Ye(`layout-header`),He(gs,ws.header,Ts.sizing(n??null),p&&ws.divider,s),c,l),...d,children:(0,U.jsx)(`div`,{...He(ws.inner,m&&ws.fullBleed,i!=null&&ut[i],i!=null&&_t[i],i!=null&&bt[i],i!=null&&ht[i],a!=null&&ft[a],a!=null&&ht[a]),children:e})})}Es.displayName=`LayoutHeader`;var Ds={"@astryx.scrollRegion.label":`Content`},Os={attributes:!0,attributeFilter:[`class`,`dir`,`hidden`,`style`],characterData:!0,childList:!0,subtree:!0};function ks(e,t){let n=!0,r=null,i=null,a=()=>{if(r=null,!n)return;let a=getComputedStyle(e),o=dt(a.writingMode,a.direction),s=vt(e,`inline`,o),c=vt(e,`block`,o);if(s==null||c==null)return;let l=s.isScrollable||c.isScrollable;l!==i&&(i=l,t(l))},o=()=>{n&&r==null&&(r=window.requestAnimationFrame(a))},s=new Map,c=()=>{let t=new Set(Array.from(e.children));for(let[e,n]of s)t.has(e)||(n(),s.delete(e));for(let e of t)s.has(e)||s.set(e,ot(e,o))},l=()=>{c(),o()};a(),c();let u=ot(e,o),d=at(e,Os,l);window.addEventListener(`resize`,o),window.visualViewport?.addEventListener(`resize`,o),e.addEventListener(`load`,o,!0),e.addEventListener(`transitionend`,o),e.addEventListener(`animationend`,o);let f=document.fonts;return f?.addEventListener?.(`loadingdone`,o),f?.ready.then(o),()=>{n=!1,r!=null&&(window.cancelAnimationFrame(r),r=null),u(),d();for(let e of s.values())e();s.clear(),window.removeEventListener(`resize`,o),window.visualViewport?.removeEventListener(`resize`,o),e.removeEventListener(`load`,o,!0),e.removeEventListener(`transitionend`,o),e.removeEventListener(`animationend`,o),f?.removeEventListener?.(`loadingdone`,o)}}Te(Ds);function As({isScrollable:e,role:t,label:n}){let r=Be(),[i,a]=(0,z.useState)(!1),o=t!=null,s=(0,z.useCallback)(t=>{if(t==null||!e||o)a(!1);else return ks(t,a)},[o,e]),c=!o&&e&&i;return{ref:s,role:t??(c?`group`:void 0),"aria-label":n??(c?r(`@astryx.scrollRegion.label`):void 0)}}var js={"@astryx.resizable.collapsed":`Collapsed`,"@astryx.resizable.handle.label":`Resize handle`},Ms=`astryx-resizable:`;function Ns(e,t,n,r){let i=Math.min(n,Math.max(t,e));if(r.length>0){let e=r[0],a=Math.abs(i-e);for(let t=1;t<r.length;t++){let n=Math.abs(i-r[t]);n<a&&(e=r[t],a=n)}return Math.min(n,Math.max(t,e))}return i}function Ps(e){if(typeof window>`u`)return null;try{let t=localStorage.getItem(Ms+e);if(t==null)return null;let n=JSON.parse(t);if(typeof n==`number`)return Number.isFinite(n)?n===0?{size:null,isCollapsed:!0}:{size:n,isCollapsed:null}:null;if(typeof n==`object`&&n){let{size:e,isCollapsed:t}=n,r=typeof e==`number`&&Number.isFinite(e)&&e>0;if(r||t===!0)return{size:r?e:null,isCollapsed:t===!0}}}catch{}return null}function Fs(e,t){if(typeof window<`u`)try{localStorage.setItem(Ms+e,JSON.stringify(t))}catch{}}var Is=1200,Ls=`Use a non-negative number of pixels, an exact "Npx" string, an exact "N%" string from 0% to 100%, pixel(value), or percent(value, {min: pixel(value)}) / percent(value, {max: pixel(value)}).`,Rs=/^(\d+(?:\.\d+)?)px$/,zs=/^(\d+(?:\.\d+)?)%$/;function Bs(e){let t=Rs.exec(e);if(t){let e=Number(t[1]);return Number.isFinite(e)?{kind:`px`,value:e}:null}let n=zs.exec(e);if(n){let e=Number(n[1]);return Number.isFinite(e)&&e<=100?{kind:`percent`,value:e}:null}return null}function Vs(e){if(typeof e!=`object`||!e||Array.isArray(e))return null;let t=e;return t.type===`pixel`&&typeof t.value==`number`&&Number.isFinite(t.value)&&t.value>=0?t.value:null}function Hs(e){if(Array.isArray(e))return null;let t=e;if(t.type!==`percent`||typeof t.value!=`number`||!Number.isFinite(t.value)||t.value<0||t.value>100)return null;let n=t.min!==void 0;if(n===(t.max!==void 0))return null;if(n){let e=Vs(t.min);return e==null?null:{kind:`percent`,value:t.value,min:e}}let r=Vs(t.max);return r==null?null:{kind:`percent`,value:t.value,max:r}}function Us(e){if(typeof e==`number`)return Number.isFinite(e)&&e>=0?{kind:`px`,value:e}:null;if(typeof e==`string`)return Bs(e);if(typeof e==`object`&&e){let t=Vs(e);return t==null?Hs(e):{kind:`px`,value:t}}return null}function Ws(e){return e?.kind===`percent`}function Gs(e,t){if(e.kind===`px`)return e.value;if(t==null)return null;let n=Math.round(e.value/100*t);return e.min===void 0?e.max===void 0?n:Math.min(n,e.max):Math.max(n,e.min)}function Ks(e){if(typeof e==`number`&&!Number.isFinite(e))return String(e);try{return JSON.stringify(e)??String(e)}catch{return String(e)}}function qs(e){if(typeof e!=`object`||!e||Array.isArray(e))return Ks(e);let t=e;return t.type===`pixel`?`{"type":"pixel","value":${Ks(t.value)}}`:Ks(e)}function Js(e){if(typeof e==`object`&&e&&!Array.isArray(e)){let t=e;if(t.type===`pixel`)return qs(e);if(t.type===`percent`){let e=[`"type":"percent"`,`"value":${Ks(t.value)}`];return t.min!==void 0&&e.push(`"min":${qs(t.min)}`),t.max!==void 0&&e.push(`"max":${qs(t.max)}`),`{${e.join(`,`)}}`}}return Ks(e)}function Ys(e,t,n,r,i,a,o,s=`useResizable`){return e===void 0?r:e===1/0&&r===1/0?1/0:t==null?(o.current||(o.current=!0,`${i}${Js(e)}${a}${r}`),r):Gs(t,n)??r}function Xs(e,t,n){let r=n?.contentBoxSize?.[0];if(r)return t===`vertical`?r.blockSize:r.inlineSize;let i=typeof window>`u`?null:window.getComputedStyle(e);if(i==null)return null;let a=t===`vertical`?e.clientHeight-parseFloat(i.paddingTop)-parseFloat(i.paddingBottom):e.clientWidth-parseFloat(i.paddingLeft)-parseFloat(i.paddingRight);return Number.isFinite(a)?a:null}function Zs(e){document.body.style.cursor=e?`col-resize`:`row-resize`,document.body.style.userSelect=`none`}function Qs(){document.body.style.cursor=``,document.body.style.userSelect=``}function $s(e,t,n){let r=[...new Set(e.snaps.map(t=>Math.min(e.maxPx,Math.max(e.minPx,t))))].sort((e,t)=>e-t),{size:i}=e,a=t>0?r.filter(e=>e>i):r.filter(e=>e<i).reverse();return a.length===0?i:a[Math.min(n,a.length)-1]}function ec(e,t){e.start(),e.move(t),e.end()}function tc(e){let t=e.state();e.collapse!=null&&e.expand!=null?t.isCollapsed?e.expand():e.collapse():ec(e,t.isCollapsed?t.minPx:-t.size)}function nc(e,t,n,r){let i=null,a=()=>{let t=e();return t==null?1:t.ownerDocument.defaultView?.getComputedStyle(t).direction===`rtl`?-1:1},o=()=>{let e=i;i=null;let t=e?.target;e!=null&&t!=null&&t.hasPointerCapture?.(e.pointerId)===!0&&t.releasePointerCapture(e.pointerId),e!=null&&Qs(),r?.(!1)};return{get isDragging(){return i!=null},pointerDown(e){let o=n();if(o.isDisabled||i!=null)return;e.preventDefault(),e.stopPropagation();let s=e.currentTarget,c=typeof s?.setPointerCapture==`function`?s:null;c?.setPointerCapture?.(e.pointerId);let l=o.direction===`horizontal`;i={pointerId:e.pointerId,startPos:l?e.clientX:e.clientY,rtl:l?a():1,target:c},r?.(!0),t.start(),Zs(l)},pointerMove(e){if(i==null||i.pointerId!==e.pointerId)return;let r=n(),a=r.direction===`horizontal`?e.clientX:e.clientY,o=r.isReversed?-1:1;t.move((a-i.startPos)*i.rtl*o)},pointerUp(e){i?.pointerId===e.pointerId&&(o(),t.end())},pointerCancel(e){i?.pointerId===e.pointerId&&(o(),t.cancel())},interrupt(){i!=null&&(o(),t.cancel())},keyDown(e){let r=n();if(r.isDisabled)return;let i=t.state(),o=r.direction===`horizontal`,s=r.isReversed?-1:1,c=e.shiftKey?50:10,l=o?a():1,u=e=>(e===`ArrowRight`||e===`ArrowDown`?1:e===`ArrowLeft`||e===`ArrowUp`?-1:0)*(o?l:1)*s>0;switch(e.key){case`ArrowRight`:case`ArrowDown`:case`ArrowLeft`:case`ArrowUp`:if(e.preventDefault(),i.isCollapsed){u(e.key)&&ec(t,Math.max(i.minPx+c,i.collapsedSize));break}if(i.snaps.length>0){ec(t,$s(i,u(e.key)?1:-1,e.shiftKey?2:1)-i.size);break}ec(t,(e.key===`ArrowRight`||e.key===`ArrowDown`?1:-1)*c*(o?l:1)*s);break;case`Home`:e.preventDefault(),ec(t,i.minPx-i.size);break;case`End`:e.preventDefault(),i.maxPx!==1/0&&ec(t,i.maxPx-i.size);break;case`Enter`:e.preventDefault(),i.collapsible&&tc(t)}},doubleClick(){!n().isDisabled&&t.state().collapsible&&tc(t)}}}function rc(e){let{value:t}=e;return{"aria-orientation":e.direction===`horizontal`?`vertical`:`horizontal`,"aria-valuenow":t==null?void 0:t.isCollapsed?Math.max(t.size,t.minPx):t.size,"aria-valuemin":t?.minPx,"aria-valuemax":t?.maxPx!=null&&Number.isFinite(t.maxPx)?t.maxPx:void 0,"aria-valuetext":t?.isCollapsed?t.collapsedText:void 0,"aria-disabled":e.isDisabled||void 0,tabIndex:0}}Te(js);var ic=`@astryx.resizable.handle.label`,ac=`@astryx.resizable.collapsed`;function oc(){return typeof window>`u`?Is:window.innerWidth}function sc(e){return typeof window>`u`?null:e===`vertical`?window.innerHeight:window.innerWidth}var cc={current:!0};function lc(e={},t=`client`){let{defaultSize:n,minSize:r,maxSize:i,containerRef:a,collapsible:o=!1,snaps:s=[],autoSaveId:c,defaultIsCollapsed:l,isCollapsed:u}=e,d=t===`server`,f=c&&!d?Ps(c):null,p=Us(n),m=Us(r),h=Us(i),g=a!=null,_=g&&(Ws(m)||Ws(h)||f?.size==null&&Ws(p)),v=g||d?Is:oc(),y=Ys(r,m,v,50,`minSize`,Ls,cc),b=Ys(i,h,v,1/0,`maxSize`,Ls,cc),x=Ys(n,p,v,250,`defaultSize`,Ls,cc),S=Ns(f?.size??(_?null:Ns(x,y,b,s))??x,y,b,s),C=o&&(u===void 0?f?.isCollapsed??l??!1:u),w=g||d?null:sc(e.direction??`horizontal`);return{size:C?0:S,isCollapsed:C,expandedSize:S,minPx:y,maxPx:b,effectiveMaxPx:b===1/0?w==null?void 0:Math.max(w,C?y:S):b}}var uc=new WeakMap;function dc(e){return uc.get(e)}function fc(e){return e==null?null:String(e)}var pc=[`role`,`tabindex`,`aria-label`,`aria-orientation`,`aria-valuenow`,`aria-valuemin`,`aria-valuemax`,`aria-valuetext`,`aria-disabled`,`data-resizing`];function mc(e,t,n={}){let r=n,i=!1,a=!1,o=new Set,s=()=>r.isReversed??!1,c=()=>r.isDisabled??!1,l={min:{current:!1},max:{current:!1},defaultSize:{current:!1},inverted:!1},u=r.autoSaveId,d=u?Ps(u):null,f=null,p=null,m=null,h=null,g=!1,_=!1,v=null,y=!1,b=0,x=null,S,C=null,w=d?.isCollapsed??r.defaultIsCollapsed??!1,T={size:0,isCollapsed:!1,min:50,max:1/0,effectiveMax:void 0,isBasisMeasured:!0},E=null,ee=null,te=!1,D=null,ne=new Set,O=()=>r.direction??`horizontal`,re=()=>r.collapsible??!1,ie=()=>r.isCollapsed!==void 0,k=()=>r.snaps??[],ae=()=>{A()};function oe(){let e=Us(r.defaultSize),t=Ws(Us(r.minSize))||Ws(Us(r.maxSize)),n=d?.size==null&&Ws(e)&&x?.isFinal!==!0,i=r.containerRef!=null;return{hasContainer:i,container:i&&(t||n),viewport:!i&&t}}function se(){let e=oe(),t=e.container?r.containerRef?.current??null:null;if((t!==f||p!==O())&&(h?.(),h=null,f=t,m=null,p=O(),t!=null)){g=!0;let e=t;h=ot(e,t=>{m=Xs(e,O(),t),g||A()}),g=!1}let n=e.viewport&&typeof window<`u`;n&&!_?(window.addEventListener(`resize`,ae),_=!0):!n&&_&&(window.removeEventListener(`resize`,ae),_=!1)}function ce(){let t=oe(),n=m!=null&&m>0,i=t.hasContainer?n?m:Is:oc(),a=!t.container||n,o=v??i,s=Ys(r.minSize,Us(r.minSize),o,50,`minSize`,Ls,l.min,e),c=Ys(r.maxSize,Us(r.maxSize),o,1/0,`maxSize`,Ls,l.max,e);x!=null&&x.isFinal||(x={px:Ys(r.defaultSize,Us(r.defaultSize),o,250,`defaultSize`,Ls,l.defaultSize,e),isFinal:a});let u=x.px;S===void 0&&(S=d?.size??(a?Ns(u,s,c,k()):null)),a&&S==null&&(S=Ns(u,s,c,k())),a&&v==null&&(C==null||C.min!==s||C.max!==c)&&(C={min:s,max:c},S!=null&&(S=Ns(S,s,c,k())));let f=Ns(S??u,s,c,k()),p=re()&&(ie()?!!r.isCollapsed:w);s>c&&!l.inverted&&(l.inverted=!0,`${s}${c}`);let h=t.hasContainer?n&&t.container?m:null:sc(O());T={size:f,isCollapsed:p,min:s,max:c,effectiveMax:c===1/0?h==null?void 0:Math.max(h,p?s:f):c,isBasisMeasured:a}}function A(){if(i)return;if(a)ce();else for(let e=0;e<3;e++){let e=oe();se(),ce();let t=oe();if(e.container===t.container&&e.viewport===t.viewport)break}t!=null&&!a&&de(),j();let e={size:T.isCollapsed?0:T.size,isCollapsed:T.isCollapsed},n={size:e.size,isCollapsed:e.isCollapsed,expandedSize:T.size,minPx:T.min,maxPx:T.max,effectiveMaxPx:T.effectiveMax},r=D==null||Object.keys(n).some(e=>D?.[e]!==n[e]);if(r&&(D=n),E==null||E.size!==e.size||E.isCollapsed!==e.isCollapsed){let t=E==null;if(E=e,!t)for(let e of[...o])e()}if(r)for(let e of[...ne])e()}function j(){let e=r.autoSaveId;if(e==null||e===``||!T.isBasisMeasured||y)return;let t={key:e,size:T.size,isCollapsed:T.isCollapsed};(te||ee==null||ee.key!==t.key||ee.size!==t.size||ee.isCollapsed!==t.isCollapsed)&&(te=!1,ee=t,Fs(e,{size:t.size,isCollapsed:t.isCollapsed}))}let M=new Map;if(t!=null)for(let e of pc)M.set(e,t.getAttribute(e));let le=t!=null&&(t.hasAttribute(`aria-label`)||t.hasAttribute(`aria-labelledby`)),ue=(e,n)=>{t!=null&&(n==null?t.removeAttribute(e):t.getAttribute(e)!==n&&t.setAttribute(e,n))};function de(){M.get(`role`)??ue(`role`,`separator`),le||ue(`aria-label`,r.label??js[ic]);let{size:e,isCollapsed:t,min:n,effectiveMax:i}=T,a=rc({direction:O(),isDisabled:c(),value:{size:t?0:e,minPx:n,maxPx:i,isCollapsed:t,collapsedText:r.collapsedLabel??js[ac]}});M.get(`tabindex`)??ue(`tabindex`,String(a.tabIndex)),ue(`aria-orientation`,a[`aria-orientation`]),ue(`aria-valuenow`,fc(a[`aria-valuenow`])),ue(`aria-valuemin`,fc(a[`aria-valuemin`])),ue(`aria-valuemax`,fc(a[`aria-valuemax`])),ue(`aria-valuetext`,fc(a[`aria-valuetext`])),ue(`aria-disabled`,fc(a[`aria-disabled`]))}let fe=e=>{ie()||(w=e)};function pe(){i||!re()||T.isCollapsed||(fe(!0),A(),r.onCollapseChange?.(!0),r.onSizeChange?.(0))}function me(){if(i)return;let e=T.isCollapsed;fe(!1),A(),e&&r.onCollapseChange?.(!1),r.onSizeChange?.(T.size)}function he(e){if(i)return;if(!Number.isFinite(e)||e<0){`${String(e)}`;return}let t=Ns(e,T.min,T.max,k()),n=T.isCollapsed;S=t,fe(!1),A(),n&&r.onCollapseChange?.(!1),r.onSizeChange?.(t)}let ge={state(){return{size:T.isCollapsed?0:T.size,isCollapsed:T.isCollapsed,minPx:T.min,maxPx:T.max,snaps:k(),collapsedSize:r.collapsedSize??40,collapsible:re()}},start(){v=N(),y=!0,b=T.isCollapsed?0:T.size},move(e){let t=b+e,n=r.collapsedSize??40;if(re()&&t<n){T.isCollapsed||(fe(!0),A(),r.onCollapseChange?.(!0),r.onSizeChange?.(0));return}let i=T.isCollapsed&&t>=n;i&&fe(!1);let a=Ns(t,T.min,T.max,k());S=a,A(),i&&r.onCollapseChange?.(!1),r.onSizeChange?.(a)},end(){P()},cancel(){P()},collapse:pe,expand:me};function N(){return oe().hasContainer?m!=null&&m>0?m:Is:oc()}function P(){y&&(v=null,y=!1,te=!0,A())}let _e=nc(()=>t,ge,()=>({direction:O(),isReversed:s(),isDisabled:c()}),e=>{ue(`data-resizing`,e?``:null)}),ve=e=>{_e.pointerDown(e)},ye=e=>{_e.pointerMove(e)},be=e=>{_e.pointerUp(e)},F=e=>{_e.pointerCancel(e)},I=e=>{_e.keyDown(e)},xe=()=>{_e.doubleClick()};function Se(){t?.addEventListener(`pointerdown`,ve),t?.addEventListener(`pointermove`,ye),t?.addEventListener(`pointerup`,be),t?.addEventListener(`pointercancel`,F),t?.addEventListener(`lostpointercapture`,F),t?.addEventListener(`keydown`,I),t?.addEventListener(`dblclick`,xe)}function Ce(){t?.removeEventListener(`pointerdown`,ve),t?.removeEventListener(`pointermove`,ye),t?.removeEventListener(`pointerup`,be),t?.removeEventListener(`pointercancel`,F),t?.removeEventListener(`lostpointercapture`,F),t?.removeEventListener(`keydown`,I),t?.removeEventListener(`dblclick`,xe),h?.(),h=null,f=null,_&&=(window.removeEventListener(`resize`,ae),!1);for(let[e,t]of M)ue(e,t)}Se(),A();let we={get size(){return T.isCollapsed?0:T.size},get isCollapsed(){return T.isCollapsed},collapse:pe,expand:me,resize:he,update(e){i||(r=e,r.autoSaveId!==u&&(u=r.autoSaveId,d=u?Ps(u):null),A())},subscribe(e){return o.add(e),()=>{o.delete(e)}},destroy(){i||(_e.interrupt(),P(),i=!0,a||Ce(),o.clear(),ne.clear())}};return uc.set(we,{snapshot:()=>D,subscribeSnapshot(e){return ne.add(e),()=>{ne.delete(e)}},model:ge,suspend(){i||a||(_e.interrupt(),P(),a=!0,Ce())},resume(){!i&&a&&(a=!1,Se(),A())},syncContainer(){i||a||(oe().container?r.containerRef?.current??null:null)!==f&&A()}}),we}var hc=[];function gc(e){let{getResizable:t,size:n,isCollapsed:r,snapshot:i,collapse:a,expand:o,resize:s}=Ec(e),c=(0,z.useCallback)(()=>{let e=t();return e==null?void 0:dc(e)?.model},[t]),l={_size:n,_isCollapsed:r,_onResizeStart:(0,z.useCallback)(()=>{c()?.start()},[c]),_onResizeMove:(0,z.useCallback)(e=>{c()?.move(e)},[c]),_onResizeEnd:(0,z.useCallback)(()=>{c()?.end()},[c]),_onResizeCancel:(0,z.useCallback)(()=>{c()?.cancel()},[c]),_minSizePx:i.minPx,_maxSizePx:i.maxPx,_snaps:e.snaps??hc,_collapsedSize:e.collapsedSize??40,_collapsible:e.collapsible??!1,_direction:e.direction??`horizontal`,_isResizableProps:!0};return vc.set(l,{collapse:a,expand:o}),i.effectiveMaxPx!=null&&yc.set(l,i.effectiveMaxPx),{size:n,isCollapsed:r,collapse:a,expand:o,resize:s,props:l}}function _c(e){let{regions:t,autoSaveId:n,containerRef:r,direction:i}=e,a=Object.entries(t),o=a.map(([e,t])=>gc({...t,containerRef:r,direction:i,autoSaveId:n?`${n}:${e}`:void 0})),s={};return a.forEach(([e],t)=>{s[e]=o[t]}),s}var vc=new WeakMap,yc=new WeakMap;function bc(e){return`props`in e?e.props:e}function xc(e){return typeof e==`object`&&!!e&&`props`in e}function Sc(e){return yc.get(e)}function Cc(e){return vc.get(e)}function wc(e,t){return e.size===t.size&&e.isCollapsed===t.isCollapsed&&e.expandedSize===t.expandedSize&&e.minPx===t.minPx&&e.maxPx===t.maxPx&&e.effectiveMaxPx===t.effectiveMaxPx}function Tc(e){let t=null,n=e,r=new Set,i=null,a=()=>{for(let e of[...r])e()},o=()=>{if(i?.(),i=null,t!=null&&r.size>0){let{resizable:e}=t,n=dc(e);i=n==null?e.subscribe(a):n.subscribeSnapshot(a)}};return{get:()=>t,set(e){t=e,o(),a()},subscribe(e){return r.add(e),r.size===1&&o(),()=>{r.delete(e),r.size===0&&(i?.(),i=null)}},read(){if(t==null)return n;let{resizable:r}=t,i=dc(r)?.snapshot()??{...e,size:r.size,isCollapsed:r.isCollapsed,expandedSize:r.size};return wc(n,i)||(n=i),n}}}function Ec(e){let t=(0,z.useRef)(e);t.current=e;let[n]=(0,z.useState)(()=>Tc(lc(e))),[r]=(0,z.useState)(()=>lc(e,`server`)),i=(0,z.useCallback)(()=>{let e=n.get();if(e!=null){dc(e.resizable)?.resume();return}let r=t.current,i=mc(`useResizable`,null,r);n.set({resizable:i,passed:r})},[n]);O(()=>{i()}),O(()=>()=>{let e=n.get();if(e==null)return;let t=dc(e.resizable);t==null?(n.set(null),e.resizable.destroy()):t.suspend()},[n]),O(()=>{let t=n.get();t!=null&&e!==t.passed&&(t.passed=e,t.resizable.update(e))},[n,e]),O(()=>{let e=n.get();e!=null&&dc(e.resizable)?.syncContainer()});let a=(0,z.useSyncExternalStore)(n.subscribe,n.read,()=>r),[,o]=(0,z.useReducer)(e=>e+1,0);O(()=>{n.read()!==a&&o()});let s=e.isCollapsed!==void 0,c=s?(e.collapsible??!1)&&!!e.isCollapsed:a.isCollapsed,l=s?c?0:a.expandedSize:a.size,u=(0,z.useCallback)(()=>n.get()?.resizable??null,[n]),d=(0,z.useCallback)(()=>{let e=n.get();if(e!=null)return e.resizable;let r=t.current,i=mc(`useResizable`,null,r);return dc(i)?.suspend(),n.set({resizable:i,passed:r}),i},[n]);return{snapshot:a,getResizable:u,size:l,isCollapsed:c,collapse:(0,z.useCallback)(()=>{d().collapse()},[d]),expand:(0,z.useCallback)(()=>{d().expand()},[d]),resize:(0,z.useCallback)(e=>{d().resize(e)},[d])}}function Dc(e){return`regions`in e?_c(e):gc(e)}var Oc=new WeakMap,kc=new Set,Ac=new WeakSet;function jc(e){return Ac.add(e),()=>{Ac.delete(e)}}function Mc(){for(let e of[...kc])e()}function Nc(e){return kc.add(e),()=>{kc.delete(e)}}function Pc(e,t){let n=t?e.nextElementSibling:e.previousElementSibling;return n==null?null:Oc.get(n)??null}function Fc(e,t,n,r){return t!==e&&e?.setCollapsed(!1,null),t?.setCollapsed(n,r),t}function Ic(e,t){let n=t===`start`?[e.nextElementSibling]:t===`end`?[e.previousElementSibling]:[e.nextElementSibling,e.previousElementSibling];for(let e of n)if(e instanceof HTMLElement&&Ac.has(e))return e;return null}function Lc(){let e=!1,t=null,n=null,r=new Set,i=t=>{if(t!==e){e=t;for(let e of[...r])e()}};return{subscribe(e){return r.add(e),()=>{r.delete(e)}},isCollapsed:()=>e,focusTarget:()=>t,attach(e){if(n?.element===e)return;let r=n!=null;if(n!=null&&Oc.get(n.element)===n&&Oc.delete(n.element),n=null,e==null)t=null,i(!1);else{let r={element:e,setCollapsed(e,a){n===r&&(t=a,i(e))}};n=r,Oc.set(e,r)}(r||e!=null)&&Mc()}}}var Rc=class extends z.Component{getSnapshotBeforeUpdate(e){return Object.is(e.value,this.props.value)||this.props.onBeforeChange(e.value),null}componentDidUpdate(){}render(){return null}};function zc(e,t,n){let r=(0,z.useRef)(t),i=(0,z.useRef)(n),a=(0,z.useRef)(null);return(0,z.useLayoutEffect)(()=>{r.current=t,i.current=n}),(0,z.useLayoutEffect)(()=>{let t=a.current;a.current=null,t!=null&&i.current(t,e)},[e]),(0,z.useCallback)(e=>{a.current=r.current(e)},[])}var Bc={panel:{kB7OPa:`x9f619`,kmuXW:`x2lah0s`,kVQacm:`x7giv3`,kZCmMZ:`xwjyata`,kwRFfy:`x1peupej`,kLKAdn:`xqty4a`,kGO01o:`xg476vw`,"--container-padding-inline-start":`x408pgh`,"--container-padding-inline-end":`xikqloz`,"--container-padding-block-start":`xjmgx01`,"--container-padding-block-end":`xi9ns85`,$$css:!0},startPanel:{kZCmMZ:`xaku5ci`,$$css:!0},endPanel:{kwRFfy:`x19eyx77`,$$css:!0},noHeader:{kLKAdn:`xuqi0w`,$$css:!0},noFooter:{kGO01o:`x1363qu6`,$$css:!0},fullBleed:{kZCmMZ:`x1c1uobl`,kwRFfy:`xyri2b`,kLKAdn:`xexx8yu`,kGO01o:`x18d9i69`,"--container-padding-inline-start":`xrhngw9`,"--container-padding-inline-end":`xjsfl84`,"--container-padding-block-start":`x1047aw6`,"--container-padding-block-end":`xax9j7h`,$$css:!0},scrollable:{kVQacm:`xysyzu8`,$$css:!0},zeroWidth:{kZCmMZ:`x1c1uobl`,kwRFfy:`xyri2b`,kVQacm:`x7giv3`,$$css:!0},dividerEnd:{ke9TFa:`x1lun4ml`,k8ry5P:`x18b5jzi`,kBCPoo:`x1gejf6u`,$$css:!0},dividerStart:{k2ei4v:`xpilrb4`,kVhnKS:`x1t7ytsu`,kGJrpR:`x1j92z86`,$$css:!0},collapseStart:{keTefX:`x1wim8z0`,$$css:!0},collapseEnd:{k71WvV:`x1kpg4um`,$$css:!0}},Vc=()=>!1,Hc={sizing:e=>[{kzqmXN:e==null?e:`x5lhr3w`,$$css:!0},{"--x-width":(e=>typeof e==`number`?e+`px`:e??void 0)(e)}]};function Uc({children:e,hasDivider:t=!1,isScrollable:n=!0,label:r,padding:i,role:a,width:o,resizable:s,xstyle:c,className:l,style:u,ref:d,...f}){let p=s==null?void 0:bc(s),m=(0,z.use)(ts),{hasHeader:h,hasFooter:g}=(0,z.use)(rs),v=p?p._size:o,[y]=(0,z.useState)(Lc),b=(0,z.useSyncExternalStore)(y.subscribe,y.isCollapsed,Vc),x=p==null,S=(0,z.useCallback)(e=>{y.attach(x?e:null)},[y,x]),C=(0,z.useRef)(null),w=p==null?b:p._isCollapsed,T=zc(w,e=>e?null:lt(C.current),(e,t)=>{let n=C.current;t&&n!=null&&st(e,y.focusTarget()??Ic(n,m),n)}),[E,ee]=(0,z.useState)(w);!w&&E&&ee(!1),O(()=>{w&&!E&&ee(!0)},[w,E]);let te=w&&E,D=m===`start`,ne=m===`end`,re=i===0,ie=v===0,k=!t&&!re&&i==null&&!ie,ae=As({isScrollable:n&&!ie,role:a,label:r}),oe=_(d,ae.ref,C,S),se=D?Bc.dividerEnd:ne?Bc.dividerStart:null,ce=D?Bc.collapseEnd:ne?Bc.collapseStart:null;return(0,U.jsxs)(`div`,{ref:oe,role:ae.role,"aria-label":ae[`aria-label`],...tt(Ye(`layout-panel`),He(Bc.panel,Hc.sizing(v??null),D&&!re&&i==null&&Bc.startPanel,ne&&!re&&i==null&&Bc.endPanel,!h&&!re&&i==null&&Bc.noHeader,!g&&!re&&i==null&&Bc.noFooter,n&&Bc.scrollable,re&&Bc.fullBleed,i!=null&&ut[i],i!=null&&_t[i],i!=null&&bt[i],i!=null&&ht[i],t&&se,k&&ce,ie&&Bc.zeroWidth,c),l,u),...f,...te?{inert:!0,"aria-hidden":!0}:{},children:[(0,U.jsx)(Rc,{value:w,onBeforeChange:T}),e]})}Uc.displayName=`LayoutPanel`;var Wc={content:{kB7OPa:`x9f619`,kZKoxP:`x5yr21d`,kUk6DE:`x98rzlu`,kAzted:`x2lwn1j`,kVQacm:`x7giv3`,kZCmMZ:`xwjyata`,kwRFfy:`x1peupej`,kLKAdn:`x1q01h0h`,kGO01o:`x12g7ifj`,"--container-padding-inline-start":`x408pgh`,"--container-padding-inline-end":`xikqloz`,"--container-padding-block-start":`xjmgx01`,"--container-padding-block-end":`xi9ns85`,$$css:!0},noStart:{kZCmMZ:`xaku5ci`,"--container-padding-inline-start":`xt2a87w`,"--container-padding-inline-end":`xhq06x6`,$$css:!0},noEnd:{kwRFfy:`x19eyx77`,$$css:!0},noHeader:{kLKAdn:`xuqi0w`,"--container-padding-block-start":`x186o6qk`,$$css:!0},noFooter:{kGO01o:`x1363qu6`,"--container-padding-block-end":`xfsxxt8`,$$css:!0},scrollable:{kVQacm:`xysyzu8`,$$css:!0},constrainedNoPanelsStart:{kZCmMZ:`xhlv5e7`,$$css:!0},constrainedNoPanelsEnd:{kwRFfy:`x1ahicqp`,$$css:!0},constrainedSingleStartPanel:{kwRFfy:`xy07wb4`,$$css:!0},constrainedSingleEndPanel:{kZCmMZ:`xxjme2g`,$$css:!0},fullBleed:{kZCmMZ:`x1c1uobl`,kwRFfy:`xyri2b`,kLKAdn:`xexx8yu`,kGO01o:`x18d9i69`,"--container-padding-inline-start":`xrhngw9`,"--container-padding-inline-end":`xjsfl84`,"--container-padding-block-start":`x1047aw6`,"--container-padding-block-end":`xax9j7h`,$$css:!0}};function Gc({children:e,isScrollable:t=!0,padding:n,label:r,role:i,xstyle:a,className:o,style:s,ref:c,...l}){let{hasHeader:u,hasFooter:d,hasStart:f,hasEnd:p}=(0,z.use)(rs),m=n===0,h=As({isScrollable:t,role:i,label:r}),g=_(c,h.ref);return(0,U.jsx)(`div`,{ref:g,role:h.role,"aria-label":h[`aria-label`],...tt(Ye(`layout-content`),He(Wc.content,!f&&!m&&n==null&&Wc.noStart,!p&&!m&&n==null&&Wc.noEnd,!u&&!m&&n==null&&Wc.noHeader,!d&&!m&&n==null&&Wc.noFooter,t&&Wc.scrollable,m&&Wc.fullBleed,n!=null&&ut[n],n!=null&&_t[n],n!=null&&bt[n],n!=null&&ht[n],!f&&!p&&!m&&Wc.constrainedNoPanelsStart,!f&&!p&&!m&&Wc.constrainedNoPanelsEnd,f&&!p&&!m&&Wc.constrainedSingleStartPanel,!f&&p&&!m&&Wc.constrainedSingleEndPanel,a),o,s),...l,children:e})}Gc.displayName=`LayoutContent`;var Kc=[s.reset,{root:{"--layout-content-width":`x1vpwbdg`,"--layout-alignment-width":`x479rhq`,"--_layout-flush-block-start":`xu9n9za`,"--_layout-flush-block-end":`xt8hc1d`,"--container-max-height":`x52i7cl`,$$css:!0}}.root],qc=(0,z.createContext)({isMobile:!1,isMobileNavOpen:!1,toggleMobileNav:()=>{},openMobileNav:()=>{},closeMobileNav:()=>{},isMobileNavEnabled:!1,hasAutoToggle:!0});qc.displayName=`AppShellMobileContext`;function Jc(){return(0,z.use)(qc)}Te({"@astryx.mobileNav.toggle.open":`Open navigation`});function Yc({ref:e,children:t,label:n,"data-testid":r,xstyle:i,className:a,style:o,onClick:s,...c}){let l=Be(),u=n??l(`@astryx.mobileNav.toggle.open`),{isMobile:d,isMobileNavEnabled:f,isMobileNavOpen:p,mobileNavId:m,toggleMobileNav:h}=Jc();return!d||!f?null:(0,U.jsx)(F,{...c,ref:e,variant:`ghost`,label:u,icon:t??(0,U.jsx)(ke,{icon:`menu`,color:`inherit`}),onClick:I(h,s),"aria-expanded":p,"aria-controls":m||void 0,"data-testid":r??`mobile-nav-toggle`,xstyle:i,className:a,style:o,isIconOnly:!0})}Yc.displayName=`MobileNavToggle`;function Xc(e,{isModal:t=!0,entrance:n=e,capMs:r}={}){if(n==null||typeof n.getAnimations!=`function`)return null;let i=n.getAnimations().filter(Qc);if(i.length===0||!Ke(e.ownerDocument.defaultView))return null;let a=Promise.all(i.map(async e=>e.finished));return k(e,{until:r==null?a:Promise.race([a,Zc(r)]),scope:t?`viewport`:`surface`})}async function Zc(e){await new Promise(t=>{setTimeout(t,e)})}function Qc(e){if(e.playState!==`running`)return!1;let t=e.effect?.getComputedTiming().endTime;return typeof t==`number`&&Number.isFinite(t)}function $c(e){return h(e)&&!e.inert&&e.getAttribute(`aria-hidden`)!==`true`}function el(e){for(let t=e;t!=null;t=t.parent){let e=t.dialogRef.current;if(h(e))return e}return null}var tl=[];function nl(e){let t=tl.indexOf(e);t!==-1&&tl.splice(t,1),e.open&&tl.push(e)}function rl(e){let t=tl.indexOf(e);t!==-1&&tl.splice(t,1)}function il(){for(let e=tl.length-1;e>=0;--e)if($c(tl[e]))return tl[e];return null}function al(e){for(let t=tl.length-1;t>=0;--t){let n=tl[t];if(n!==e&&$c(n))return n;if(h(n))return`covered`}let t=null;for(let n of document.querySelectorAll(`dialog`))n!==e&&!tl.includes(n)&&$c(n)&&(t=n);return t}function ol(){for(let e of document.querySelectorAll(`dialog`))if(h(e))return!0;return!1}var sl=new Map;function cl(e,t,n,r){let i=sl.get(e);i??(i={polite:[],assertive:[]},sl.set(e,i));let a=i[n].find(e=>e.text===t);a??(a={text:t,receipts:[]},i[n].push(a)),a.receipts.push(r);let o=a;r.withdraw=()=>ll(e,n,o,r)}function ll(e,t,n,r){let i=sl.get(e),a=n.receipts.indexOf(r);i==null||!i[t].includes(n)||a<0||(n.receipts.splice(a,1),n.receipts.length===0&&(i[t]=i[t].filter(e=>e!==n),i.polite.length===0&&i.assertive.length===0&&sl.delete(e)))}function ul(e,t){let n=sl.get(e);n!=null&&(n[t]=[],n.polite.length===0&&n.assertive.length===0&&sl.delete(e))}function dl(e){let t=sl.get(e);if(t!=null){sl.delete(e);for(let n of[`polite`,`assertive`])S(t[n],t=>$c(e)?_l(e,t,n):pl(e,t,n))}}function fl(){for(let e of[...sl.keys()])dl(e)}function pl(e,t,n){let r=al(e);if(r===`covered`||r==null&&ol()){let r={withdraw:null};return cl(e,t,n,r),x(r)}return r==null?y(t,n):_l(r,t,n)}function ml(e){for(let t of[`polite`,`assertive`])S(gl(e,t),n=>pl(e,n,t))}function hl(e,t){E(e,t),ul(e,t)}function gl(e,t){let n=f(e,t);if(n==null)return[];let r=ne(n);return E(e,t),r}function _l(e,t,n=`polite`){if(!$c(e))return pl(e,t,n);let r=il();return r!=null&&r!==e&&tl.includes(e)?(E(e,n),null):l(e,t,n,{isExposed:()=>$c(e),redirect:t=>S(t,t=>pl(e,t,n))})}var vl=new WeakMap;function yl(e,t,n){let r=vl.get(e);if(r==null){let t={frame:requestAnimationFrame(()=>{vl.delete(e);let n=el(e);for(let e of[`polite`,`assertive`])t.clears[e]&&(n==null?m(e):hl(n,e)),S(t.words[e],t=>n==null?y(t,e):_l(n,t,e))}),words:{polite:[],assertive:[]},clears:{polite:!1,assertive:!1}};r=t,vl.set(e,r)}let i=r;if(!t){for(let e of i.words[n])for(let t of e.receipts)t.withdraw=null;return i.words[n]=[],i.clears[n]=!0,null}let a=i.words[n].find(e=>e.text===t);a??(a={text:t,receipts:[]},i.words[n].push(a));let o=a,s={withdraw:null};return s.withdraw=()=>{let e=o.receipts.indexOf(s);e<0||(o.receipts.splice(e,1),o.receipts.length===0&&(i.words[n]=i.words[n].filter(e=>e!==o)))},o.receipts.push(s),x(s)}function bl(e){for(let t=e;t!=null;t=t.parent){let e=t.dialogRef.current;if(h(e))return!1;if(t.isOpen&&(e==null||!e.open))return!0}return!1}function xl(e,t,n){if(bl(e))return yl(e,t,n);let r=el(e);return t?r==null?y(t,n):_l(r,t,n):(r==null?m(n):hl(r,n),null)}g({announce:xl,announceInModal:_l});function Sl({dialogRef:e,isOpen:t,children:n}){let r=(0,z.use)(w),i=(0,z.useMemo)(()=>({dialogRef:e,isOpen:!1,parent:r}),[e,r]);O(()=>{i.isOpen=t},[i,t]);let a=(0,z.useRef)(null);return(0,z.useEffect)(()=>{let n=e.current;if(n==null||a.current!=null||!t&&!n.open)return;nl(n);let r=n.open,i=()=>{o(),rl(n),a.current=null},o=()=>s(),s=at(n,{attributes:!0,attributeFilter:[`open`,`inert`,`aria-hidden`]},()=>{n.open!==r&&(r=n.open,nl(n)),$c(n)||ml(n),(!n.open||$c(n))&&fl(),n.open||i()});a.current={dialog:n,stop:o}},[e,t]),(0,z.useEffect)(()=>()=>{let e=a.current;e!=null&&(a.current=null,e.stop(),ml(e.dialog),rl(e.dialog),fl())},[]),(0,U.jsx)(w,{value:i,children:n})}Te({"@astryx.mobileNav.closeNavigation":`Close navigation`,"@astryx.mobileNav.navigation":`Navigation`});var Cl={dialog:{kVAEAm:`xixxii4`,kogj98:`x1ghz6dp`,kmVPX3:`x1717udv`,kMzoRj:`xc342km`,ksu8eU:`xng3xce`,ks0D6T:`x1x1rfll`,kskxy:`x7ab17h`,kpwlN0:`x10a8y8t`,kzqmXN:`xn9wirt`,kZKoxP:`xtdtrs8`,kWkggS:`xjbqb8w`,kVQacm:`x7giv3`,kZeWKH:`xish69e`,kFalU9:`x1j4r261`,kI3sdo:`x1a2a7pz`,k1xSpc:`x1s85apg`,k1ekBW:`xrgi2yo`,kIyJzY:`x80gvsz`,kzIqYQ:`xd00j3c`,$$css:!0},open:{k1xSpc:`x78zum5`,$$css:!0},backdrop:{kGyWv1:`xnixb3f`,k5sjJv:`xph5o2a`,kND0Po:`x167zut7`,k9an0g:`xft5bk6`,kb4ib:`x15h3t91`,kA5Tbj:`x1viac0w`,$$css:!0},backdropOpen:{k5sjJv:`xb3n6bw xxiuuzi`,$$css:!0},drawer:{kVAEAm:`x10l6tqk`,k87sOh:`x13vifvy`,krVfgx:`x1ey2m1c`,k1xSpc:`x78zum5`,kXwgrk:`xdt5ytf`,kWkggS:`x10xzikg`,kB7OPa:`x9f619`,kVQacm:`xb3r6kr`,k1ekBW:`x11xpdln`,kIyJzY:`x80gvsz`,kAMwcw:`xlr8y92`,kI3sdo:`x1a2a7pz`,k6CgDc:`xzg1mie`,$$css:!0},drawerStart:{kLqNvP:`x1o0tod`,ke9TFa:`xw8tdv1`,k8ry5P:`x18b5jzi`,kBCPoo:`x1gejf6u`,k3aq6I:`x5i6ehr xttggg`,$$css:!0},drawerStartOpen:{k3aq6I:`xbryuvx x6mt36l x14gflnl`,$$css:!0},drawerEnd:{kt4wiu:`xtijo5x`,k2ei4v:`xgbv0en`,kVhnKS:`x1t7ytsu`,kGJrpR:`x1j92z86`,k3aq6I:`xumwmo6 x1df3fe5`,$$css:!0},drawerEndOpen:{k3aq6I:`xbryuvx x1yqmsfc x1lymnkk`,$$css:!0},headerText:{keTefX:`x11g1kdw`,$$css:!0}},wl={kzqmXN:`x5lhr3w`,ks0D6T:`x1jrhgas`,$$css:!0},Tl={width:e=>[wl,{"--x-width":(e=>typeof e==`number`?e+`px`:e??void 0)(`${e}px`)}]},El=.6;function Dl(e){let t=e.split(`,`).map(e=>{let t=e.trim(),n=Number.parseFloat(t);return Number.isFinite(n)?t.endsWith(`ms`)?n:t.endsWith(`s`)?n*1e3:null:null}).filter(e=>e!==null);return t.length?Math.min(...t):null}function Ol(e){let t=Xe(e.ownerDocument.defaultView,`(prefers-reduced-motion: reduce)`)?0:250,n=Dl(window.getComputedStyle(e).transitionDuration);return n===null?t:n<=0?0:Math.min(t,n*El)}function kl(e,t){e.close();let n=t.current;t.current=null,n&&n!==document.body&&n.isConnected&&n.focus()}function Al({isOpen:e,onOpenChange:t,children:n,header:r,width:i=320,side:a=`auto`,label:o,"data-testid":c,xstyle:l,className:d,style:f,onClick:m,ref:h,...g}){let v=Be(),y=Jc(),b=e??y.isMobileNavOpen,x=(0,z.useId)(),S=y.mobileNavId||x,w=(0,z.useMemo)(()=>t??(e=>{e?y.openMobileNav():y.closeMobileNav()}),[t,y]),E=(0,z.useRef)(null);ct(E,b);let ee=(0,z.useRef)(null),ne=(0,z.useRef)(null),O=(0,z.useRef)(null),re=(0,z.useRef)(null),[ie,k]=(0,z.useState)(!1),[ae,oe]=(0,z.useState)(b);ae!==b&&(oe(b),k(!b)),mt(b);let[se,ce]=(0,z.useState)(a===`auto`?`end`:a);(0,z.useEffect)(()=>{if(b){if(a===`auto`){let e=document.activeElement;if(e&&e!==document.body){let t=e.getBoundingClientRect(),n=t.left+t.width/2;ce(n<window.innerWidth/2?`start`:`end`)}}else ce(a)}},[b,a]),(0,z.useEffect)(()=>{let e=E.current;if(e)return b?(O.current?.(),O.current=null,e.open||(re.current=document.activeElement,e.showModal()),O.current=Xc(e,{entrance:ee.current,capMs:550})):e.open?(O.current?.(),O.current=null,ne.current=setTimeout(()=>{kl(e,re),k(!1)},Ol(e))):k(!1),()=>{ne.current&&=(clearTimeout(ne.current),null),O.current?.(),O.current=null}},[b]),(0,z.useEffect)(()=>{let e=E.current;return()=>{O.current?.(),O.current=null,e?.open&&kl(e,re)}},[]);let{shouldDismissOnCloseRequest:A}=T({isActive:b,onDismiss:()=>w(!1),getContainer:()=>E.current}),j=(0,z.useCallback)(e=>{e.target===e.currentTarget&&(e.preventDefault(),A()&&w(!1))},[w,A]),M=(0,z.useCallback)(e=>{e.target===e.currentTarget&&w(!1)},[w]),ue=se===`start`,de=o??(typeof r==`string`?r:v(`@astryx.mobileNav.navigation`));return(0,U.jsx)(`dialog`,{ref:_(h,E),id:S,...tt(Ye(`mobile-nav`,{side:se}),He(D.reset,p.reset,Cl.dialog,s.reset,(b||ie)&&Cl.open,Cl.backdrop,b&&Cl.backdropOpen,l),d,f),...g,"data-testid":c,"aria-label":de,...u({...g,onClick:I(m,M)}),onCancel:j,children:(0,U.jsx)(Sl,{dialogRef:E,isOpen:b,children:(0,U.jsx)(C,{children:(0,U.jsx)(te,{children:(0,U.jsxs)(`div`,{ref:ee,tabIndex:-1,...He(Cl.drawer,Tl.width(i),ue&&Cl.drawerStart,ue&&b&&Cl.drawerStartOpen,!ue&&Cl.drawerEnd,!ue&&b&&Cl.drawerEndOpen),children:[(0,U.jsxs)(`div`,{...{0:{className:`x78zum5 x6s0dn4 x1qughib x3unnv6 xf314gf x2lah0s x92x3c3 x1q0q8m5 xw8gpjh`},1:{className:`x78zum5 x6s0dn4 x3unnv6 xf314gf x2lah0s x92x3c3 x1q0q8m5 xw8gpjh x13a6bvl`}}[!r|0],children:[typeof r==`string`?(0,U.jsx)(le,{level:2,xstyle:Cl.headerText,children:r}):r??null,(0,U.jsx)(F,{variant:`ghost`,label:v(`@astryx.mobileNav.closeNavigation`),icon:(0,U.jsx)(ke,{icon:`close`,color:`inherit`}),onClick:()=>w(!1),isIconOnly:!0})]}),(0,U.jsx)(`nav`,{"aria-labelledby":g[`aria-labelledby`],"aria-label":de,className:`x98rzlu x1odjw0f x6ikm8r xish69e xge0x1e xf314gf xce4md1`,children:n})]})})})})})}Al.displayName=`MobileNav`;var jl=(0,z.createContext)(`default`);jl.displayName=`SideNavRenderContext`;function Ml(){return(0,z.use)(jl)}var Nl=.5,Pl=200,Fl=60;function Il(e,t){let n=e.ownerDocument.defaultView;return typeof n?.matchMedia==`function`&&n.matchMedia(t).matches}function Ll(e){let t=ca(e,{duration:`--duration-medium`,easing:`--ease-move`}),n=sa(e,`--ease-move`);if(t==null||n==null)return null;let r=t.duration*1e3;return{durationMs:r,easing:n,remaining:e=>e>=r?0:1-t.at(e/1e3)}}function Rl(e){if(Il(e,`(prefers-reduced-motion: reduce)`)||Qi(e)){if(!Il(e,`(pointer: coarse)`))return{kind:`instant`};let[t,n,r,i]=v.curve;return{kind:`fade`,durationMs:Math.min(Pl,v.durationMs),easing:`cubic-bezier(${t}, ${n}, ${r}, ${i})`}}let t=Ll(e);return t==null?{kind:`instant`}:{kind:`glide`,progress:t}}function zl(e){return Math.max(2,Math.ceil(e/1e3*Fl)+1)}function Bl(e,t){let n=zl(e),r=[];for(let i=0;i<n;i++){let a=i/(n-1);r.push({...t(a*e),offset:a})}return r}function Vl(e,t){return Bl(e.durationMs,n=>t(e.remaining(n)))}var Hl=new WeakMap,Ul=new Set,Wl=[];function Gl(){Wl.length!==0&&Wi(()=>{for(let e of Wl.splice(0))e.remove();for(let e of[...Ul]){let{element:t}=e;t.isConnected&&t.ownerDocument.defaultView!=null||e.remove()}})}function Kl(e){let t=[];for(let[n,r]of e.entries()){let i=/^(-?[\d.e+-]+)px (-?[\d.e+-]+)px$/.exec(String(r.translate??``));if(i==null)return null;t.push({at:r.offset??n/Math.max(1,e.length-1),x:Number(i[1]),y:Number(i[2])})}return t.length>0?t:null}function ql(e,t,n,r,i){let a=e.ownerDocument.defaultView,o=Number(r.duration),s=(r.easing??`linear`)===`linear`&&r.delay==null?Kl(n):null;if(a==null||s==null||!(o>0)||!(`style`in e)||e.closest(`[inert]`)!=null)return;let c=e,l=()=>a.performance.now(),u=e=>{let n=t.startTime,r=n==null?0:Math.min(1,Math.max(0,(e-Number(n))/o)),a=1;for(;a<s.length-1&&s[a].at<r;)a+=1;let c=s[a-1]??s[0],l=s[a]??c,u=l.at-c.at,d=u>0?Math.min(1,Math.max(0,(r-c.at)/u)):1;return{x:(c.x+(l.x-c.x)*d)*i.x,y:(c.y+(l.y-c.y)*d)*i.y}},d=u(l()),f=!1,p=!1,m={};Hl.set(e,m);let h=Gi(e,{offset:(e=l())=>{if(f)return{x:0,y:0};let t=u(e);return{x:t.x-d.x,y:t.y-d.y}},resolve:()=>{f||(qi(c),d=u(l()))}}),g={element:e,remove:()=>{p||(p=!0,Ul.delete(g),h(),Hl.get(e)===m&&(Hl.delete(e),Ji(c)))}};Ul.add(g);let _=()=>{f||(f=!0,Wl.push(g),setTimeout(Gl,0))};t.addEventListener?.(`finish`,_),t.addEventListener?.(`cancel`,_)}function Jl(e,t,n,r,i={x:1,y:1}){if(typeof e.animate!=`function`)return null;let a;try{a=e.animate(t,n)}catch{return null}tu.add(a);let o=e.ownerDocument,s=o.timeline?.currentTime;if(s!=null){let e=r!=null&&r.frame===Number(s)?r:Zl(o,Number(s));if(e.last=Xl(o),a.startTime=e.at,e.at!==Number(s)){let e=a.effect?.getTiming().fill;a.effect?.updateTiming({fill:e===`forwards`||e===`both`?`both`:`backwards`})}}return n.pseudoElement??ql(e,a,t,n,i),a}var Yl=new WeakMap;function Xl(e){return(e.defaultView?.performance??performance).now()}function Zl(e,t){let n=Xl(e),r=Yl.get(e);if(r!=null&&r.frame===t)return r.last=n,r;let i={frame:t,at:n-t<=34?t:n,last:n};return Yl.set(e,i),(e.defaultView?.setTimeout??setTimeout)(()=>{Yl.get(e)===i&&Yl.delete(e)},0),i}function Ql(e){let t=Yl.get(e);t!=null&&Xl(e)-t.last>34&&Yl.delete(e)}function $l(e){let t=e.timeline?.currentTime;return t==null?0:Zl(e,Number(t)).at}function eu(e,t){let n=e.getBoundingClientRect(),r=su.get(e),i=r?.animation.startTime;if(r==null||i==null)return n;let a=e.ownerDocument.defaultView?.getComputedStyle(e).translate??``;if(a.includes(`%`))return n;let[o=0,s=0]=ru(a),c=r.offsetAt(Math.max(0,t-Number(i)));return new DOMRect(n.x+(c.x-o)*r.scale.x,n.y+(c.y-s)*r.scale.y,n.width,n.height)}var tu=new WeakSet,nu=[`transform`,`translate`,`rotate`,`scale`];function ru(e){return(e.match(/-?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?/gi)??[]).map(Number)}function K(e,t){if(t===`none`||e===`scale`)return!0;if(t.includes(`var(`))return!1;if(e===`translate`)return ru(t).every(e=>e===0);if(e===`rotate`){let e=ru(t);return e.length>0&&e[e.length-1]===0}let n=[...t.matchAll(/([a-z0-9]+)\(([^)]*)\)/gi)];return n.length>0&&n.every(([,e,t])=>{let n=ru(t),r=e.toLowerCase();return r===`matrix`?n.length===6&&[1,2,4,5].every(e=>n[e]===0):r===`matrix3d`?n.length===16&&n.every((e,t)=>t%5==0||e===0):r===`rotate3d`?n[3]===0:r.startsWith(`scale`)?!0:r.startsWith(`translate`)||r.startsWith(`rotate`)||r.startsWith(`skew`)?n.every(e=>e===0):!1})}function iu(e){let t=e.ownerDocument.defaultView?.getComputedStyle(e);return t!=null&&nu.some(e=>{let n=t.getPropertyValue(e);return n!==``&&n!==`none`&&!K(e,n)})?!0:typeof e.getAnimations==`function`&&e.getAnimations().some(e=>{if(tu.has(e))return!1;let t=e.effect;return t instanceof KeyframeEffect&&t.getKeyframes().some(e=>nu.some(t=>t in e&&!K(t,String(e[t]))))})}var au=()=>({x:0,y:0});function ou(e,t){let n=zl(e),r=Array.from({length:n},(r,i)=>t(i/(n-1)*e));return t=>{let i=Math.min(Math.max(t/e,0),1)*(n-1),a=Math.min(Math.floor(i),n-2),o=i-a,s=r[a],c=r[a+1];return{x:s.x+(c.x-s.x)*o,y:s.y+(c.y-s.y)*o}}}var su=new WeakMap,cu=Symbol(`astryx.animatedLayout.area`);function lu(e){return Object.freeze(Object.defineProperty({},cu,{value:e}))}function uu(e,t){let n=e[cu]();return n!=null&&n!==t&&n.contains(t)?n:null}function du(e,t){if(t.display===`none`||t.visibility===`hidden`||t.position===`absolute`||t.position===`fixed`)return!1;try{if(e.matches(`:popover-open, :modal`))return!1}catch{}return!0}function fu(e,t){return{start:e.top-t,end:e.bottom+t}}function pu(e,t){let n=t.parts.get(e);if(n===void 0){let r=e.ownerDocument.defaultView?.getComputedStyle(e);n=r==null?!1:r.display===`contents`?`contents`:du(e,r),t.parts.set(e,n)}return n}function mu(e){if(e==null)return!1;let t=e.ownerDocument.defaultView?.getComputedStyle(e);if(t==null)return!1;let n=t.getPropertyValue(`column-count`),r=t.getPropertyValue(`column-width`);return n!==``&&n!==`auto`||r!==``&&r!==`auto`||t.display.includes(`flex`)&&t.flexDirection.startsWith(`column`)&&t.flexWrap.startsWith(`wrap`)||t.display.includes(`grid`)&&t.getPropertyValue(`grid-auto-flow`).startsWith(`column`)}function hu(e){let t=e.ownerDocument.defaultView;for(let n=e.parentElement;n!=null&&n!==e.ownerDocument.documentElement;n=n.parentElement){let r=t?.getComputedStyle(n).overflowY??`visible`;if((n!==e.ownerDocument.body||t?.getComputedStyle(e.ownerDocument.documentElement).overflowY!==`visible`)&&r!==`visible`&&r!==`clip`){let e=n.getBoundingClientRect();return{start:e.top+n.clientTop,end:e.top+n.clientTop+n.clientHeight}}}return{start:0,end:t?.innerHeight??0}}function gu(e,t){let n=e.ownerDocument.defaultView?.getComputedStyle(e);if(n?.position!==`sticky`)return!1;let r=yu(e,t),i=hu(e),a=Number.parseFloat(n.top),o=Number.parseFloat(n.bottom);return Number.isFinite(a)&&Math.abs(r.top-(i.start+a))<1||Number.isFinite(o)&&Math.abs(r.bottom-(i.end-o))<1}var _u=3;function vu(e,t,n){let r=[];for(let t=e;t!=null&&t!==n;t=t.parentElement){let e=su.get(t),n=Number(e?.animation.currentTime);e!=null&&Number.isFinite(n)&&r.push({node:t,glide:e,time:n})}if(r.length===0)return t;let i=e.ownerDocument.defaultView,a=()=>r.map(({node:e})=>i?.getComputedStyle(e).translate),o=a();if(o.some(e=>e==null||e===``))return t;let s=e.getBoundingClientRect();for(let t=0;t<_u;t++){let t=a();if(t.every((e,t)=>e===o[t]))break;o=t,s=e.getBoundingClientRect()}let c=0,l=0;return r.forEach(({node:t,glide:n,time:r},i)=>{let[a=0,u=0]=ru(o[i]),d=n.offsetAt(Math.min(Math.max(0,r),n.durationMs)),f=Ou(t,t===e?s:t.getBoundingClientRect(),ku(t));c+=(d.x-a)*f.x,l+=(d.y-u)*f.y}),new DOMRect(s.x+c,s.y+l,s.width,s.height)}function yu(e,t){let n=t.rects.get(e);return n===void 0&&(n=e.getBoundingClientRect(),t.rects.set(e,n)),n}function bu(e,t,n,r,i,a,o){let s=new Map,c=!1,l=(e,t)=>{let a=pu(e,r);if(a===`contents`){for(let n=t?e.firstElementChild:e.lastElementChild;n!=null;n=t?n.nextElementSibling:n.previousElementSibling)if(!l(n,t))return!1;return!0}if(!a)return!0;let u=yu(e,r);return c?(s.set(e,u),!0):u.top>n.end?!1:u.bottom<n.start?t&&i:(s.set(e,u),t||o.add(e),!0)},u=[];for(let n=e;n!=null&&n!==t;n=n.parentElement)u.push(n);let d=0;for(let e=u.length-1;e>0;--e)if(pu(u[e],r)===!1||gu(u[e],r)){d=e;break}for(let[t,n]of u.entries()){c=mu(n.parentElement),n!==e&&t>d&&pu(n,r)===!0&&(s.set(n,yu(n,r)),a.add(n));for(let e=n.nextElementSibling;e!=null&&l(e,!0);e=e.nextElementSibling);if(!(t<d))for(let e=n.previousElementSibling;e!=null&&l(e,!1);e=e.previousElementSibling);}return s}var xu=new Map;function Su(e){let t=xu.get(e);if(t==null){let n={area:e,changed:new Set,before:new Map,candidates:new Map,changedSize:0,reach:0,visible:null,walksToBand:null,ancestors:new Set,befores:new Set,originBefore:null,cache:{rects:new Map,parts:new Map},scheduled:!1,beforeMove:[],start:null};xu.set(e,n);let r=e.ownerDocument.defaultView,i=()=>{xu.get(e)===n&&!n.scheduled&&xu.delete(e)};typeof r?.requestAnimationFrame==`function`?r.requestAnimationFrame(i):setTimeout(i,16),t=n}return t}function Cu(e){let t=e.area.ownerDocument,n=t.timeline?.currentTime;if(n!=null)return(e.start==null||e.start.frame!==Number(n))&&(e.start=Zl(t,Number(n))),e.start}function wu(e,t){let n=e==null?void 0:xu.get(e);if(n!=null)return Cu(n);let r=t.timeline?.currentTime;return r==null?void 0:Zl(t,Number(r))}function Tu(e,t){return e.width>0&&e.height>0&&e.right>t.left&&e.left<t.right&&e.bottom>t.top&&e.top<t.bottom}function Eu(e){let t=e.getBoundingClientRect();return{x:t.left-e.scrollLeft,y:t.top-e.scrollTop}}function Du(e){let t=e.getBoundingClientRect(),n=e.ownerDocument.defaultView,r=n?.innerWidth??t.right,i=n?.innerHeight??t.bottom,a=Math.max(t.left,0),o=Math.max(t.top,0);return new DOMRect(a,o,Math.max(0,Math.min(t.right,r)-a),Math.max(0,Math.min(t.bottom,i)-o))}function Ou(e,t,n={x:1,y:1}){let r=e,i=r.offsetWidth>0&&t.width>0?t.width/r.offsetWidth/n.x:1,a=r.offsetHeight>0&&t.height>0?t.height/r.offsetHeight/n.y:1;return{x:Math.abs(i-1)<.01?1:i||1,y:Math.abs(a-1)<.01?1:a||1}}function ku(e){let t=e.ownerDocument.defaultView?.getComputedStyle(e);if(t==null)return{x:1,y:1};let n=1,r=1,i=t.getPropertyValue(`scale`);if(i!==``&&i!==`none`){let[e=1,t=e]=ru(i);n*=Math.abs(e),r*=Math.abs(t)}let a=/^matrix(3d)?\(([^)]*)\)/.exec(t.getPropertyValue(`transform`));if(a!=null){let e=ru(a[2]),[t=1,i=0,o=0,s=1]=a[1]==null?e:[e[0],e[1],e[4],e[5]];n*=Math.hypot(t,i),r*=Math.hypot(o,s)}return{x:n||1,y:r||1}}function Au(e){xu.delete(e.area);let t=t=>{for(let n of e.beforeMove)n(t)},n=new Set;for(let t of e.changed)for(let r of e.candidates.get(t)??[])r.isConnected&&n.add(r);if(n.size===0){t([]);return}let r=Rl(e.area);if(r.kind===`instant`){t([]);return}let i=Cu(e),a=Du(e.area),o=[];for(let t of n){let n=e.before.get(t);if(n==null||n.width===0&&n.height===0)continue;let r=su.get(t),a=au,s=0,c=au(0);if(r!=null){let e=Number(r.animation.currentTime),n=Number.isFinite(e)?Math.max(0,e):0;n<r.durationMs&&(c=r.offsetAt(n));let{startTime:o}=r.animation,l=i!=null&&o!=null?Math.max(n,i.at-Number(o)):n;l<r.durationMs&&(a=e=>r.offsetAt(l+e),s=r.durationMs-l),r.animation.cancel(),su.delete(t)}o.push({element:t,before:n,carried:a,carriedMs:s,held:c})}let s=new Map,c=Eu(e.area),l=c.x-(e.originBefore?.x??c.x),u=c.y-(e.originBefore?.y??c.y),d=(e,t)=>Math.abs(e+t)<Math.abs(e)?e+t:e;for(let t of o){t.after=vu(t.element,t.element.getBoundingClientRect(),e.area);let n=t.element.parentElement??t.element,r=n.ownerDocument.defaultView?.getComputedStyle(n).direction===`rtl`?t.before.right-t.after.right:t.before.left-t.after.left,i=t.before.top-t.after.top,a=e.ancestors.has(t.element)||e.befores.has(t.element);t.rawX=a?d(r,l):r,t.rawY=a?d(i,u):i,s.set(t.element,t)}for(let e of o){let t=e.after;(Tu(e.before,a)||Tu(t,a))&&(e.isMovedByApp=iu(e.element),e.scale=Ou(e.element,t,ku(e.element)))}t(o.map(e=>e.element));for(let t of o){let{element:n,before:o,carried:c,carriedMs:l,held:u}=t,d=t.after,f=t.rawX??0,p=t.rawY??0;for(let t=n.parentElement;t!=null&&t!==e.area;t=t.parentElement){let e=s.get(t);if(e!=null){f-=e.rawX??0,p-=e.rawY??0;break}}if(Math.abs(f)<Nl&&Math.abs(p)<Nl||!Tu(o,a)&&!Tu(d,a))continue;if(r.kind===`fade`){Jl(n,[{opacity:0},{opacity:1}],{duration:r.durationMs,easing:r.easing},i);continue}if(t.isMovedByApp===!0)continue;let m=t.scale??{x:1,y:1},h=f/m.x-u.x,g=p/m.y-u.y,{progress:_}=r,v=Math.max(_.durationMs,l),y=ou(v,e=>{let t=_.remaining(e),n=c(e);return{x:n.x+h*t,y:n.y+g*t}}),b=Jl(n,Bl(v,e=>{let t=y(e);return{translate:`${t.x}px ${t.y}px`}}),{duration:v,easing:`linear`},i,m);if(b!=null){let e={animation:b,offsetAt:y,durationMs:v,shownBefore:{x:f/m.x,y:p/m.y},scale:m};su.set(n,e),b.addEventListener?.(`finish`,()=>{su.get(n)===e&&su.delete(n)})}}}function ju(e){return typeof e==`function`?e():e.current}function Mu(e,t){return{measureBefore(){let n=ju(t);if(n==null||e==null)return;let r=uu(e,n);if(r==null)return;Ql(n.ownerDocument);let i=Su(r);if(i.changed.has(n))return;i.changed.add(n),i.changedSize+=yu(n,i.cache).height,i.visible??=Du(r),i.originBefore??=Eu(r),i.walksToBand??=r.ownerDocument.defaultView?.getComputedStyle(r).getPropertyValue(`overflow-anchor`)===`none`;let a=Math.max(i.visible.height,i.changedSize),o=a>i.reach?[...i.changed]:[n];a>i.reach&&(i.reach=Math.max(a,i.reach*2));let s=fu(i.visible,i.reach);for(let e of o){let t=bu(e,r,s,i.cache,i.walksToBand,i.ancestors,i.befores);for(let[e,n]of t)i.before.has(e)||i.before.set(e,vu(e,n,r));i.candidates.set(e,[...t.keys()])}},animate(n){let r=ju(t);if(r==null||e==null)return!1;let i=uu(e,r),a=i==null?void 0:xu.get(i);return a==null||!a.changed.has(r)?!1:(n!=null&&a.beforeMove.push(n),a.scheduled||(a.scheduled=!0,queueMicrotask(()=>Wi(()=>Au(a)))),!0)}}}function Nu(e){return e.ownerDocument.defaultView?.getComputedStyle(e).direction===`rtl`}function Pu(e,t,n,r){let i=[n,n,n,n],a=e===`block-start`?0:e===`block-end`?2:e===`inline-end`===r?3:1;return i[a]=t,`inset(${i.join(` `)})`}function Fu(e,t,n){return Pu(e,`${t}px`,`0px`,n)}var Iu={x:0,y:0};function Lu(e,t,{isVisiblePart:n=!1}={}){if(e==null||typeof t.animate!=`function`)return null;let r=uu(e,t);if(r==null||Rl(r).kind===`instant`||!ua(t))return null;Ql(t.ownerDocument);let i=t.getBoundingClientRect();if(!n){let e=ba(t);return e==null?null:{area:r,copy:e,rect:i}}let a=Ta(t);return a==null?null:{area:r,...a}}function Ru(e,t){let n=Rl(e.area);if(n.kind===`instant`)return()=>{};let{copy:r}=e,i=wu(e.area,r.ownerDocument);wa(r,t.host,e.rect,t.copyBefore??null),t.isOpaque===!0&&(t.copyBefore??(r.style.zIndex=`1`),r.style.backgroundColor=Sa(r));let a=[],o=!1,s=()=>{o||(o=!0,a.forEach(e=>e.cancel()),r.remove(),t.onEnd?.())},{copyText:c,rideWith:l}=t,u=c!=null&&n.kind===`glide`?Da(r):[],d=[];if(n.kind===`glide`){let e=l!=null&&l.isCopyMoved!==!1;d.push([r,(n,r)=>({...t.copy(n,r),...e?{translate:`${r.x}px ${r.y}px`}:null})]);for(let e of u)d.push([e,e=>({opacity:c?.(e)??1})]);for(let{element:e,frames:n}of t.incoming)d.push([e,n])}else{d.push([r,e=>({opacity:1-e})]);for(let{element:e}of t.incoming)d.push([e,e=>({opacity:e})])}let f=(e,t,n,r,o)=>{for(let[s,c]of d){let l=Jl(s,o(e=>c(t(e),n(e))),{duration:e,fill:`both`,...r},i);l!=null&&a.push(l)}a.length===0?s():a[0].addEventListener?.(`finish`,s)};if(n.kind===`fade`)return f(n.durationMs,e=>e<=0?0:1,()=>Iu,{easing:n.easing},e=>[e(0),e(n.durationMs)]),s;let{progress:p}=n,m=e=>1-p.remaining(e);return l==null?(f(p.durationMs,m,()=>Iu,{easing:`linear`},e=>Bl(p.durationMs,e)),s):(queueMicrotask(()=>{if(o)return;let e=su.get(l.element),t=e?.durationMs??p.durationMs;f(t,m,t=>{if(e==null)return{x:-l.travel.x,y:-l.travel.y};let n=e.offsetAt(t),r=e.shownBefore;return{x:n.x-r.x,y:n.y-r.y}},{easing:`linear`},e=>Bl(t,e))}),s)}function zu(e,t){let n=e?.[cu]()??null,r=n==null?null:Rl(n);if(n==null||r==null||r.kind===`instant`)return()=>{};let i=wu(n,n.ownerDocument),a=[];for(let{element:e,frames:n}of t){let t=r.kind===`glide`?Jl(e,Vl(r.progress,e=>n(1-e,Iu)),{duration:r.progress.durationMs,easing:`linear`,fill:`both`},i):Jl(e,[{opacity:0},{opacity:1}],{duration:r.durationMs,easing:r.easing},i);t!=null&&a.push(t)}let o=()=>a.forEach(e=>e.cancel());return a[0]?.addEventListener?.(`finish`,o),o}var Bu={end:()=>{},shown:()=>null};function Vu(e,t,{edge:n=`block-end`,shown:r=null}={}){if(t==null||typeof t.animate!=`function`)return null;Ql(t.ownerDocument);let i=t.getBoundingClientRect();return{element:t,rect:i,extent:r??(n.startsWith(`inline`)?i.width:i.height),swap:Lu(e,t)}}function Hu(e,t,n,{edge:r=`block-end`,rideWith:i=null,parts:a}={}){let{rect:o,swap:s,extent:c}=e,l=t?.isConnected===!0?t:e.element;if(!l.isConnected)return Bu;let u=Rl(l),d=l.getBoundingClientRect(),f=r.startsWith(`inline`),p=e=>f?e.width:e.height,m=p(d);if(u.kind===`instant`||Math.abs(m-c)<.5)return Bu;let h=m<c,g=r.endsWith(`end`),_=f&&Nu(l),v=i?.parentElement!=null&&i.parentElement.contains(l)?i.parentElement:l.parentElement,y=v!=null&&i!=null&&i.parentElement===v&&s!=null,b=g?1:-1,x=e=>b*(f?_?-e.x:e.x:e.y),S=(e,t)=>y?c+x(t):c+(m-c)*e,C=e=>Fu(r,e,_),w=()=>Number(l.ownerDocument.timeline?.currentTime??0),T=wu(uu(n,l),l.ownerDocument)?.at??0,E=u.kind===`glide`?u.progress.durationMs:u.durationMs,ee=e=>u.kind===`glide`?1-u.progress.remaining(Math.min(e,E)):0,te=()=>{let e=Math.max(0,w()-T);if(e>=E)return null;let t=ee(Math.max(e,$l(l.ownerDocument)-T)),n=c+(m-c)*ee(e);if(y&&i!=null){let e=eu(i,w()),t=l.getBoundingClientRect();n=f?g===_?t.right-e.right:e.left-t.left:g?e.top-t.top:t.bottom-e.bottom}return{extent:n,progress:t}},D=b*(c-m),ne=y&&i!=null?{element:i,travel:f?{x:_?-D:D,y:0}:{x:0,y:D},isCopyMoved:!1}:void 0,O=new DOMRect(f&&g===(f&&_)?d.right-o.width:d.left,!f&&!g?d.bottom-o.height:d.top,o.width,o.height),re=a?.(h,l)??{},ie=re.copyOpacity??(()=>1);if(h){if(s==null||v==null)return Bu;let e=l;for(;e!=null&&e.parentElement!==v;)e=e.parentElement;let t=e==null?null:{position:e.style.position};return e!=null&&getComputedStyle(e).position===`static`&&(e.style.position=`relative`),{end:Ru({...s,rect:O},{host:v,isOpaque:!0,copyBefore:e??void 0,copy:(e,t)=>({clipPath:C(p(o)-S(e,t)),opacity:ie(e)}),copyText:re.copyText,rideWith:ne,onEnd:()=>{e!=null&&t!=null&&(e.style.position=t.position)},incoming:re.incoming??[]}),shown:te}}let k=[{element:l,frames:(e,t)=>({clipPath:C(m-S(e,t))})},...re.incoming??[]];return{end:s!=null&&v!=null?Ru({...s,rect:O},{host:v,isOpaque:!0,copy:e=>({opacity:ie(e)}),copyText:re.copyText,rideWith:ne,incoming:k}):zu(n,k),shown:te}}function Uu(e){let t=su.get(e);t!=null&&(su.delete(e),t.animation.cancel())}function Wu(e){return Rl(e)}var Gu=(0,z.createContext)(null);Gu.displayName=`SideNavShellContext`;function Ku(e){return lu(()=>e()?.parentElement??null)}function qu(e,t,n,r){let i=Vu(t,e,{edge:`inline-end`,shown:n?.width??null});return i==null?null:{hold:i,swap:i.swap,shown:n??{width:i.extent,labels:+!r,collapsed:+!!r}}}var Ju=e=>Math.min(1,Math.max(0,e));function Yu(e){return Array.from(e.querySelectorAll(`*`)).filter(e=>Array.from(e.childNodes).some(e=>e.nodeType===3&&e.textContent?.trim()))}function Xu(e,t,n,r=null){let{shown:i}=e,a=!1,o=e=>a?i.labels*(1-Ju(e/.4)):i.labels+(1-i.labels)*Ju((e-.4)/.6),s=e=>a?i.collapsed+(1-i.collapsed)*Ju((e-.7)/.3):i.collapsed*(1-Ju(e/.6)),c=Hu(e.hold,t,n,{edge:`inline-end`,rideWith:r,parts:(e,t)=>(a=e,e?{copyOpacity:e=>1-Ju((e-.7)/.3),copyText:o,incoming:[{element:t,frames:e=>({opacity:s(e)})}]}:{copyOpacity:s,incoming:Yu(t).map(e=>({element:e,frames:e=>({opacity:o(e)})}))})});return{end:c.end,shown:()=>{let e=c.shown();return e==null?null:{width:e.extent,labels:o(e.progress),collapsed:s(e.progress)}}}}function Zu({containerRef:e}){return(0,z.useMemo)(()=>lu(()=>e.current),[e])}var Qu=(0,z.createContext)(`default`);Qu.displayName=`TopNavRenderContext`;var q=(0,z.createContext)(null);q.displayName=`TopNavMobileContentContext`;var $u=`astryx-app-shell-main`,J=(0,z.createContext)(!1);J.displayName=`AppShellNestingContext`;var Y=[],X=new Set;function ed(e){return X.add(e),()=>{X.delete(e)}}function td(){for(let e of X)e()}function nd(){let e=(0,z.useId)(),t=(0,z.use)(J),[n]=(0,z.useState)(()=>({})),r=(0,z.useSyncExternalStore)(ed,()=>Y.length===0||Y[0]===n,()=>!0);return O(()=>{if(!t)return Y.push(n),td(),()=>{Y.splice(Y.indexOf(n),1),td()}},[n,t]),r&&!t?$u:`astryx-app-shell-main-${e}`}Te({"@astryx.appShell.mobileNavigation":`Mobile navigation`,"@astryx.appShell.skipToContent":`Skip to content`});var rd=z.Activity!==void 0,id=rd?({mode:e,children:t})=>(0,U.jsx)(z.Activity,{mode:e,children:t}):({children:e})=>(0,U.jsx)(U.Fragment,{children:e}),ad=50,Z={root:{k1xSpc:`x78zum5`,kXwgrk:`xdt5ytf`,kVAEAm:`x1n2onr6`,kVQacm:`x7giv3`,$$css:!0},variantWash:{kWkggS:`x1eiddq6`,$$css:!0},variantSurface:{kWkggS:`x10xzikg`,$$css:!0},variantSection:{kWkggS:`x10xzikg`,$$css:!0},variantElevated:{kWkggS:`x1eiddq6`,$$css:!0},rootFill:{kZKoxP:`xtdtrs8`,$$css:!0},rootAuto:{kAzted:`x1ov3xa9`,$$css:!0},skipLink:{kVAEAm:`x10l6tqk x1xrnuwo`,kzqmXN:`x1i1rx1s x1jqxupm`,kZKoxP:`xjm9jq1 x15cytp8`,k8WAf4:`xt970qd xh2mrf5`,kg3NbH:`xnjsko4 x1cf3d6k`,kogj98:`xkdpibf x1y5lnwp`,kVQacm:`xb3r6kr xomzh7y`,kz4h6p:`x1hyvwdk x1rsz1da`,khDVqt:`xuxw1ft x1hbpcn8`,kMzoRj:`xc342km`,k87sOh:`x13vifvy x1rw3289`,kLqNvP:`x1o0tod xodanix`,kWkggS:`x10xzikg`,kMwMTN:`xjse4m1`,kY2c9j:`x1q2oy4v`,kybGjl:`x1hl2dhg`,k63SB2:`x2mo6ok`,kGuDYH:`xjm74w1`,$$css:!0},mainFocusTarget:{kI3sdo:`x1uvtmcs`,$$css:!0},mainElevatedCorner:{krdFHd:`x9a4xmf`,$$css:!0},mainScrollRegionInset:{kInvED:`x1wfwxd8 x1nw7mhf`,$$css:!0},contentBgSurface:{kWkggS:`x10xzikg`,$$css:!0},contentBgWash:{kWkggS:`x1eiddq6`,$$css:!0},contentBgTransparent:{kWkggS:`xjbqb8w`,kHBbk8:`xc8icb0`,$$css:!0},navAreaWash:{kWkggS:`x1eiddq6`,$$css:!0},navAreaSurface:{kWkggS:`x10xzikg`,$$css:!0},banner:{kmuXW:`x2lah0s`,$$css:!0},headerSticky:{kVAEAm:`x7wzq59`,k87sOh:`x13vifvy`,kY2c9j:`x1vjfegm`,$$css:!0},headerSafeAreaTop:{kLKAdn:`x1mrd0q`,$$css:!0},headerSafeAreaSides:{kZCmMZ:`xs5q72w xqbmqmu`,kwRFfy:`x180b3rg x17h5b6w`,$$css:!0},headerStickyBelowTopInset:{k87sOh:`xz44ai8`,$$css:!0},mainSafeAreaEnd:{kGO01o:`xvmia7e`,$$css:!0},panelAutoFill:{kUk6DE:`x98rzlu`,kVQacm:`xysyzu8`,$$css:!0}};function od(e,t){let n=e;for(;n.parentElement!=null&&!n.parentElement.contains(t);)n=n.parentElement;return n}function sd({variant:e=`elevated`,banner:t,children:n,contentPadding:r,"data-testid":i,height:a=`fill`,mobileNav:o,sideNav:s,topNav:c,xstyle:l,className:u,style:d,ref:f,...p}){let m=Be(),h=_e(),g=o===!1,v=o!=null&&o!==!1&&typeof o==`object`&&!(0,z.isValidElement)(o)?o:null,y=v?.breakpoint??`md`,b=o!=null&&o!==!1&&((0,z.isValidElement)(o)||typeof o==`string`)?o:null,x=v?.content??null,S=v?.hasToggle!==!1,C=v?.isOpen!==void 0,w=h?.__adaptations?.widthBreakpoints,T=y===`none`?`(width < 0px)`:`(width < ${w?.[y]??Me[y]}px)`,E=nt(T,y!==`none`&&v?.defaultIsMobile),[ee,te]=(0,z.useState)(!1),D=v?.isOpen??ee,[ne,O]=(0,z.useState)(D);D&&!ne&&O(!0),(0,z.useEffect)(()=>{if(!rd||D)return;let e=Xe(window,`(prefers-reduced-motion: reduce)`)?0:250+ad,t=setTimeout(()=>{O(!1)},e);return()=>clearTimeout(t)},[D]);let re=D||ne,ie=v?.onOpenChange,k=(0,z.useCallback)(e=>{C||te(e),ie?.(e)},[C,ie]),ae=nd(),oe=(0,z.useCallback)(()=>{document.getElementById(ae)?.focus()},[ae]),[se,ce]=(0,z.useState)(!1),A=(0,z.useRef)(!1),j=(0,z.useCallback)(e=>{e.target===e.currentTarget&&(document.activeElement===e.currentTarget?A.current=!0:(A.current=!1,ce(!1)))},[]),M=a===`fill`,le=a===`auto`,[ue,de]=(0,z.useState)(!1),fe=M&&ue,pe=(0,z.useRef)(null),me=(0,z.useRef)(null),he=(0,z.useRef)(null),ge=(0,z.useRef)(fe),N=(0,z.useCallback)(()=>{let e=ge.current?pe.current:null,t=he.current;t?.node!==e&&(t?.release(),he.current=e==null?null:{node:e,release:yt(e,e)})},[]),P=(0,z.useCallback)(e=>{pe.current=e;let t=M?e:null,n=me.current;n?.node!==t&&(n?.stop(),me.current=t==null?null:{node:t,stop:ks(t,de)},t??de(!1)),N()},[M,N]);(0,z.useEffect)(()=>{ge.current=fe,N()},[fe,N]);let ve=xe(t),ye=xe(c),be=xe(s),F=!g&&(ye||be)&&b==null,I=e===`section`,Se=e===`elevated`,Ce=e===`wash`||e===`elevated`?Z.navAreaWash:e===`surface`?Z.navAreaSurface:void 0,we=Ce??(le&&e===`section`?Z.navAreaSurface:void 0),Te=e===`wash`?Z.contentBgWash:e===`elevated`&&ye&&be&&!E?Z.contentBgTransparent:e===`surface`||e===`elevated`?Z.contentBgSurface:void 0,Ee=Ce??Z.navAreaSurface,De=(0,z.useRef)(null),Oe=(0,z.useRef)(null);(0,z.useEffect)(()=>{if(!le||!De.current||!Oe.current)return;let e=De.current,t=Oe.current,n=()=>{let n=e.getBoundingClientRect().height;t.style.setProperty(`--_app-shell-header-height`,`${n}px`)};return ot(e,()=>n())},[le]);let ke=be&&!E,Ae=b!=null,je=F&&x!=null&&E,L=(0,z.useId)(),Ne=(0,z.useMemo)(()=>({isMobile:E,isMobileNavOpen:D,mobileNavId:L,toggleMobileNav:()=>F&&k(!D),openMobileNav:()=>F&&k(!0),closeMobileNav:()=>k(!1),isMobileNavEnabled:F,hasAutoToggle:S}),[E,D,L,k,F,S]),Pe=be&&S?(0,U.jsx)(jl,{value:`drawer-content`,children:s}):null,Fe=be?(0,U.jsx)(jl,{value:`drawer-content`,children:s}):null,R=ye?E&&!g&&b==null?(0,U.jsx)(q,{value:Pe,children:(0,U.jsx)(Qu,{value:`mobile-bar`,children:c})}):c:null,Ie=ye||ve?(0,U.jsxs)(Es,{padding:0,hasDivider:I&&ye,children:[ve&&(0,U.jsx)(`div`,{...He(Z.banner,Ce),children:t}),ye&&R]}):void 0,Le=xe(Ie)?(0,U.jsx)(`div`,{ref:De,role:`banner`,...tt(Ye(`app-shell-header`,{variant:e}),He(we,le&&Z.headerSticky,Z.headerSafeAreaTop,Z.headerSafeAreaSides)),children:Ie}):void 0,Re=(0,z.useRef)(null),ze=(0,z.useRef)(null),Ve=_(ze,P),Ue=(0,z.useRef)(null),We=Zu({containerRef:Oe}),Ge=(0,z.useRef)(null),Ke=(0,z.useRef)(null),Je=(0,z.useMemo)(()=>({animatedLayout:We,rail:()=>Re.current,column:()=>ze.current==null?null:od(ze.current,Re.current),landContent(){ze.current!=null&&Uu(od(ze.current,Re.current))},beforeRailChange(){let e=ze.current;if(e==null)return;let t=Wu(e),n=null;t.kind===`glide`&&(n=Mu(We,Re),n.measureBefore());let r=1,i=0,a=Ke.current;if(a!=null&&t.kind===`glide`){let n=Math.max(0,$l(e.ownerDocument)-a.startedAt);if(n<t.progress.durationMs){let e=1-t.progress.remaining(n);r=a.entering+(1-a.entering)*e,i=a.held*(1-e)}}Ke.current=null,Ge.current={change:n,swap:Lu(We,e,{isVisiblePart:!0}),before:e.getBoundingClientRect(),held:r,entering:i}},afterRailChange(){let e=Ge.current;Ge.current=null;let t=ze.current;if(e==null||t==null)return()=>{};e.change?.animate();let n=od(t,Re.current),r=t.getBoundingClientRect(),i=getComputedStyle(t).direction===`rtl`,a=i?e.before.right-r.right:e.before.left-r.left,o=[],s=Ue.current??t,c=getComputedStyle(s).backgroundColor,l=!/^rgba\(.*,\s*0\)$|^transparent$/.test(c),u=Wu(t).kind===`glide`;if((i?a>0:a<0)&&u&&l&&o.push({element:s,frames:e=>({boxShadow:`${-a*(1-e)}px 0px 0px 0px ${c}`})}),e.swap==null||n.parentElement==null)return zu(We,o);!u&&l&&(e.swap.copy.style.backgroundColor=c);let{held:d,entering:f}=e;return Ke.current={startedAt:$l(t.ownerDocument),held:d,entering:f},Ru(e.swap,{host:n.parentElement,copy:e=>({opacity:d*(1-e)}),rideWith:{element:n,travel:{x:a,y:0}},incoming:[...Array.from(t.children,e=>({element:e,frames:e=>({opacity:f+(1-f)*e})})),...o]})}}),[We]),Qe=ke?(0,U.jsx)(Uc,{ref:Re,padding:0,hasDivider:I,isScrollable:M,...Ye(`app-shell-sidenav`,{variant:e}),xstyle:[Ce,le&&Ee,le&&Z.panelAutoFill],children:(0,U.jsx)(Gu,{value:Je,children:s})}):void 0,$e=Qe!=null&&le?(0,U.jsx)(`div`,{className:`x2lah0s x7giv3 x7wzq59 xht72ud xpa73km x78zum5 xdt5ytf`,children:Qe}):Qe,et=Se&&ye&&ke;(0,z.useEffect)(()=>{if(!fe)return;let e=null,t=null,n,r=()=>{clearTimeout(n),n=setTimeout(()=>{e=null,t=null},0)},i=n=>{e=n.key===`Tab`?n:null,t=null,r()},a=()=>{e=null,t=null},o=e=>{t=e,r()},s=n=>{let r=e!=null&&e.eventPhase===0,i=t!=null&&t.eventPhase===0,a=r&&e?.defaultPrevented===!1,o=i;r&&(e=null),i&&(t=null);let s=pe.current,c=n.target!=null&&n.target===s;if(A.current){if(A.current=!1,c&&!a)return;if(!c){ce(!1);return}}c&&ce(a||o&&s.matches(`:focus-visible`))};return document.addEventListener(`keydown`,i,!0),document.addEventListener(`pointerdown`,a,!0),document.addEventListener(`focusin`,s,!0),window.addEventListener(`focus`,o),()=>{document.removeEventListener(`keydown`,i,!0),document.removeEventListener(`pointerdown`,a,!0),document.removeEventListener(`focusin`,s,!0),window.removeEventListener(`focus`,o),clearTimeout(n)}},[fe]);let rt=se,it=(0,U.jsx)(Gc,{ref:Ve,padding:r??0,role:`main`,id:ae,tabIndex:fe?0:-1,isScrollable:M,onBlur:j,xstyle:[Te,rt&&et&&Z.mainElevatedCorner,rt?[qe.focusVisible,Z.mainScrollRegionInset]:Z.mainFocusTarget,Z.mainSafeAreaEnd],children:(0,U.jsx)(J,{value:!0,children:n})}),at=(0,U.jsxs)(`div`,{...{0:{className:`xjp7ctv`},1:{className:`x1n2onr6 x78zum5 x98rzlu x2lwn1j x5yr21d`}}[!!et|0],children:[et&&(0,U.jsx)(`div`,{ref:Ue,className:`x10l6tqk x10a8y8t x10xzikg x183tx6i x47corl`}),it]}),st=!g&&S&&E&&!ye&&be?(0,U.jsx)(`div`,{role:Le==null?`banner`:void 0,...tt(Ye(`app-shell-header`,{variant:e}),He(we,le&&Z.headerSticky,Z.headerSafeAreaSides,Le==null&&Z.headerSafeAreaTop,le&&Le!=null&&Z.headerStickyBelowTopInset)),children:(0,U.jsx)(Es,{padding:0,hasDivider:I,children:(0,U.jsxs)(`div`,{className:`x78zum5 x6s0dn4 x1k15mir xf314gf`,role:`navigation`,"aria-label":m(`@astryx.appShell.mobileNavigation`),children:[(0,U.jsx)(jl,{value:`topbar`,children:s}),(0,U.jsx)(Yc,{})]})})}):void 0;return(0,U.jsx)(qc,{value:Ne,children:(0,U.jsx)(`div`,{...p,ref:_(f,Oe),"data-testid":i,...tt(Ye(`app-shell`,{variant:e}),He(Z.root,Kc,e===`wash`?Z.variantWash:e===`surface`?Z.variantSurface:e===`section`?Z.variantSection:Z.variantElevated,M?Z.rootFill:Z.rootAuto,l),u,d),children:(0,U.jsx)(ts,{value:null,children:(0,U.jsx)(rs,{value:ns,children:(0,U.jsxs)(is,{value:null,children:[(0,U.jsx)(`a`,{href:`#${ae}`,onClick:oe,...Ze.focusVisible(Z.skipLink),"data-testid":`skip-to-content`,children:m(`@astryx.appShell.skipToContent`)}),(0,U.jsx)(Cs,{height:a,padding:0,header:(0,U.jsxs)(U.Fragment,{children:[Le,st]}),start:$e,content:at}),Ae&&b,je&&x,E&&!g&&b==null&&!x&&(0,U.jsxs)(id,{mode:re?`visible`:`hidden`,children:[be&&!ye&&(0,U.jsx)(jl,{value:`drawer`,children:s}),ye&&(0,U.jsx)(q,{value:Fe,children:(0,U.jsx)(Qu,{value:`drawer`,children:c})})]})]})})})})})}sd.displayName=`AppShell`;var cd=(0,z.createContext)({isCollapsed:!1,toggle:()=>{},isCollapsible:!1});cd.displayName=`SideNavCollapseContext`;function ld(){return(0,z.use)(cd)}var ud=(0,z.createContext)(null);ud.displayName=`SideNavFocusTargetsContext`,Te({"@astryx.sideNavCollapseButton.collapseSidebar":`Collapse sidebar`,"@astryx.sideNavCollapseButton.expandSidebar":`Expand sidebar`});var dd={chevronMirror:{k1xSpc:`x78zum5`,kGNEyG:`x6s0dn4`,kjj79g:`xl56j7k`,$$css:!0},chevron:{k1xSpc:`x3nfvp2`,kGNEyG:`x6s0dn4`,k1ekBW:`x11xpdln`,kIyJzY:`xuedmi6 x12w9bfk`,kAMwcw:`xlr8y92`,$$css:!0},chevronCollapsed:{k3aq6I:`x19jd1h0`,$$css:!0}};function fd({ref:e,collapsible:t,handleRef:n,label:r,size:i,children:a,onClick:o,...s}){let c=Be(),{isCollapsed:l,toggle:u,isCollapsible:d}=pd(t,n),{isMobile:f}=Jc(),p=(0,z.use)(ud),m=p!=null&&t==null&&n==null,h=(0,z.useCallback)(e=>e==null||!m||p==null?void 0:p.registerToggle(e),[p,m]),g=_(e,h);return!d||f?null:(0,U.jsx)(F,{ref:g,label:r??c(l?`@astryx.sideNavCollapseButton.expandSidebar`:`@astryx.sideNavCollapseButton.collapseSidebar`),variant:`ghost`,size:i,...s,onClick:I(o,u),icon:a??(0,U.jsx)(`span`,{...He(dd.chevronMirror,A.mirror),children:(0,U.jsx)(ke,{icon:`chevronLeft`,size:`sm`,color:`inherit`,xstyle:[dd.chevron,l&&dd.chevronCollapsed]})}),isIconOnly:!0})}fd.displayName=`SideNavCollapseButton`;function pd(e,t){let n=ld(),{isCollapsed:r,onCollapsedChange:i}=e??{},a=(0,z.useCallback)(()=>i?.(!r),[r,i]),o=(0,z.useCallback)(()=>{t?.current?.getCollapseState()?.toggle()},[t]);if(e!=null)return{isCollapsed:r??!1,toggle:a,isCollapsible:!0};if(t==null)return n;let s=t.current?.getCollapseState()??null;return{isCollapsed:s?.isCollapsed??!1,toggle:o,isCollapsible:s?.isCollapsible??!0}}Te({"@astryx.resizable.collapsed":`Collapsed`,"@astryx.resizable.handle.label":`Resize handle`});function md(e,t,n){if(e!==`auto`)return e;let r=t?`end`:`start`;return n?r===`start`?`end`:`start`:r}var Q={handle:{kVAEAm:`x1n2onr6`,kmuXW:`x2lah0s`,k1xSpc:`x78zum5`,kGNEyG:`x6s0dn4`,kjj79g:`xl56j7k`,kWkggS:`x1m4xfpy`,k1ekBW:`x15406qy`,kIyJzY:`xuedmi6`,kAMwcw:`xlr8y92`,$$css:!0},overlay:{kVAEAm:`x10l6tqk`,kY2c9j:`xhtitgo`,kWkggS:`xjbqb8w`,$$css:!0},overlayHorizontal:{kt4wiu:`xtijo5x`,k87sOh:`x13vifvy`,krVfgx:`x1ey2m1c`,kzqmXN:`xkhob8h`,$$css:!0},overlayVertical:{krVfgx:`x1ey2m1c`,kLqNvP:`x1o0tod`,kt4wiu:`xtijo5x`,kZKoxP:`x5kfih4`,$$css:!0},overlayHorizontalOnShortPane:{kzqmXN:`x1a4a7d3`,$$css:!0},overlayVerticalOnShortPane:{kZKoxP:`xcuarzx`,$$css:!0},horizontal:{kzqmXN:`x1i1rx1s`,kZKoxP:`x5yr21d`,kkrTdU:`xicojor x16khyan`,$$css:!0},vertical:{kZKoxP:`xjm9jq1`,kzqmXN:`xh8yej3`,kkrTdU:`xyozgr7 x16khyan`,$$css:!0},noDividerHorizontal:{kWkggS:`xjbqb8w`,kzqmXN:`xnalus7`,$$css:!0},noDividerVertical:{kWkggS:`xjbqb8w`,kZKoxP:`xqtp20y`,$$css:!0},handleHover:{kWkggS:`x1m4xfpy`,$$css:!0},handleActive:{kWkggS:`x7njt3n`,$$css:!0},disabled:{kkrTdU:`xt0e3qv`,kfzvcC:`x47corl`,$$css:!0},hitArea:{kVAEAm:`x10l6tqk`,kY2c9j:`x1vjfegm`,kFalU9:`x5ve5x3`,kfSwDN:`x87ps6o`,$$css:!0},hitAreaHorizontal:{kzqmXN:`x12xnipv`,k87sOh:`x13vifvy`,krVfgx:`x1ey2m1c`,kkrTdU:`xicojor x16khyan`,$$css:!0},hitAreaVertical:{kZKoxP:`x6b6gus`,kLqNvP:`x1o0tod`,kt4wiu:`xtijo5x`,kkrTdU:`xyozgr7 x16khyan`,$$css:!0},hitAreaCenteredY:{k87sOh:`xwa60dl`,k3aq6I:`x1cb1t30`,$$css:!0},hitAreaWithinHandleX:{kzqmXN:`x5wh7xo`,$$css:!0},hitAreaWithinHandleY:{kZKoxP:`x1tvnn7s`,$$css:!0},pill:{kVAEAm:`x10l6tqk`,kY2c9j:`xhtitgo`,kFalU9:`x5ve5x3`,kfSwDN:`x87ps6o`,kaIpWk:`xjspbzw`,kWkggS:`x1m4xfpy`,k1ekBW:`x1wbls7o`,kIyJzY:`xuedmi6`,kAMwcw:`xlr8y92`,k87sOh:`xwa60dl`,$$css:!0},pillGrabHorizontal:{kkrTdU:`xicojor x16khyan`,$$css:!0},pillGrabVertical:{kkrTdU:`xyozgr7 x16khyan`,$$css:!0},pillHorizontal:{kzqmXN:`x1g8rjiy`,kZKoxP:`x18to7ep`,$$css:!0},pillVertical:{kzqmXN:`xgc8j7m`,kZKoxP:`xuoj239`,$$css:!0},pillHidden:{kSiTet:`xg01cxk`,kfzvcC:`x47corl`,$$css:!0},pillVisible:{kSiTet:`x1hc1fzr`,$$css:!0},pillHover:{kSiTet:`x1hc1fzr`,kfzvcC:`x67bb7w`,kWkggS:`x1m4xfpy`,$$css:!0},pillActive:{kSiTet:`x1hc1fzr`,kfzvcC:`x67bb7w`,kWkggS:`x7njt3n`,$$css:!0},pillBridgeRight:{kgeoSG:`x1cpjm7i`,kEoFBp:`x1hmns74`,kGOzcv:`x1nc033x`,kSJ0CW:`x1682cnc x7my30s`,koaPKo:`x5ig7ga x3k25x4`,$$css:!0},pillBridgeLeft:{kgeoSG:`x1cpjm7i`,kEoFBp:`x1hmns74`,kGOzcv:`x1nc033x`,kSJ0CW:`x1vl71nl xnx00js`,koaPKo:`xtyp5od x17kcdp5`,$$css:!0},pillBridgeDown:{kgeoSG:`x1cpjm7i`,kEoFBp:`x1hmns74`,kpsdNU:`xm4btww`,k96MEf:`x1y3wzot`,k5Ofw7:`xxgnp13`,$$css:!0},pillBridgeUp:{kgeoSG:`x1cpjm7i`,kEoFBp:`x1hmns74`,kpsdNU:`xm4btww`,k96MEf:`x1phjwh8`,k5Ofw7:`xhq5o37`,$$css:!0}},hd={kzqmXN:`x5lhr3w`,$$css:!0},gd={kZKoxP:`x16ye13r`,$$css:!0},_d={kLqNvP:`x1o0tod`,k3aq6I:`xsqj5wx`,$$css:!0},vd={k87sOh:`x13vifvy`,k3aq6I:`xsqj5wx`,$$css:!0},yd={hitAreaLengthX:e=>[hd,{"--x-width":(e=>typeof e==`number`?e+`px`:e??void 0)(`min(${Ue[`--spacing-4`]}, ${e}px)`)}],hitAreaLengthY:e=>[gd,{"--x-height":(e=>typeof e==`number`?e+`px`:e??void 0)(`min(${Ue[`--spacing-4`]}, ${e}px)`)}],pillOffsetX:e=>[_d,{"--x-transform":`translate(calc(${e} * (100% + ${Ue[`--spacing-1`]})), -50%)`==null?void 0:`translate(calc(${e} * (100% + ${Ue[`--spacing-1`]})), -50%)`}],pillOffsetY:e=>[vd,{"--x-transform":`translate(-50%, calc(${e} * (100% + ${Ue[`--spacing-1`]})))`==null?void 0:`translate(-50%, calc(${e} * (100% + ${Ue[`--spacing-1`]})))`}]};function bd({direction:e=`horizontal`,position:t=`inline`,isReversed:n=!1,isDisabled:r=!1,hasDivider:i=!1,isAlwaysVisible:a=!0,pillPlacement:o=`auto`,label:s,resizable:c,children:l,xstyle:u,className:d,ref:f,...p}){let m=c==null?void 0:bc(c),h=Be(),g=s??h(`@astryx.resizable.handle.label`),v=(0,z.useRef)(null),[y,b]=(0,z.useState)(!1),[x,S]=(0,z.useState)(!1),C=e===`horizontal`,w=t===`overlay`,T=(0,z.useRef)(m);T.current=m;let E=(0,z.useRef)(void 0),te=(0,z.useRef)({direction:e,isReversed:n,isDisabled:r});te.current={direction:e,isReversed:n,isDisabled:r};let[D,ne]=(0,z.useState)(!1),[re]=(0,z.useState)(()=>nc(()=>v.current,{state(){let e=T.current;return{size:e?._size??0,isCollapsed:e?._isCollapsed??!1,minPx:e?._minSizePx??0,maxPx:e?._maxSizePx??1/0,snaps:e?._snaps??[],collapsedSize:e?._collapsedSize??0,collapsible:e?._collapsible??!1}},start:()=>{E.current=T.current,T.current?._onResizeStart()},move:e=>T.current?._onResizeMove(e),end:()=>{let e=T.current??E.current;E.current=void 0,e?._onResizeEnd()},cancel:()=>{let e=T.current??E.current;E.current=void 0,e?._onResizeCancel?.()},get collapse(){let e=T.current;return e?Cc(e)?.collapse:void 0},get expand(){let e=T.current;return e?Cc(e)?.expand:void 0}},()=>({...te.current,isDisabled:te.current.isDisabled||T.current==null}),ne)),ie=m==null?null:re,k=m?._isCollapsed??!1,ae=md(o,n,k),oe=m?._size,se=k||oe==null||oe<=0?void 0:oe,ce=y||x,j=m?._direction;it(`ResizeHandle`,`direction="${e}" but its useResizable region is "${j}". They must match: the hook measures one axis and the handle drags the other.`,j!=null&&j!==e);let M=m!=null;(0,z.useEffect)(()=>{if(M)return()=>{re.interrupt()}},[M,re]);let{ref:le}=ee(),[,ue]=(0,z.useReducer)(e=>e+1,0),de=(0,z.useRef)(null),fe=(0,z.useCallback)(e=>{if(e==null)return;let t=jc(e);return()=>{t(),de.current=Fc(de.current,null,!1,null)}},[]),pe=m?._isCollapsed??!1;(0,z.useEffect)(()=>M?Nc(ue):void 0,[M]),O(()=>{let e=v.current,t=M&&e!=null?Pc(e,n):null;de.current=Fc(de.current,t,pe,e)});let me=m?._size,he=m?._minSizePx,ge=m?Sc(m)??(m._maxSizePx===1/0?void 0:m._maxSizePx):void 0,N=rc({direction:e,isDisabled:r,value:me==null||he==null?void 0:{size:me,minPx:he,maxPx:ge,isCollapsed:k,collapsedText:h(`@astryx.resizable.collapsed`)}}),P={onPointerDown:e=>{r||ie?.pointerDown(e)},onPointerMove:e=>ie?.pointerMove(e),onPointerUp:e=>ie?.pointerUp(e),onPointerCancel:e=>ie?.pointerCancel(e),onLostPointerCapture:e=>ie?.pointerCancel(e),onPointerEnter:()=>b(!0),onPointerLeave:()=>{D||b(!1)}};return(0,U.jsxs)(`div`,{ref:_(f,v,fe,le),role:`separator`,...N,"aria-label":g,onDoubleClick:()=>{r||ie?.doubleClick()},onFocus:()=>S(!0),onBlur:()=>S(!1),"data-resizing":D||void 0,...tt(Ye(`resize-handle`),Ze.focusVisible(Q.handle,w&&Q.overlay,w&&(C?Q.overlayHorizontal:Q.overlayVertical),w&&se!=null&&(C?Q.overlayHorizontalOnShortPane:Q.overlayVerticalOnShortPane),!w&&(C?Q.horizontal:Q.vertical),!w&&!i&&(C?Q.noDividerHorizontal:Q.noDividerVertical),!w&&i&&ce&&!D&&Q.handleHover,!w&&i&&D&&Q.handleActive,r&&Q.disabled,u),d),...p,onKeyDown:e=>{p.onKeyDown?.(e),r||ie?.keyDown(e)},children:[(0,U.jsx)(`div`,{...He(Q.hitArea,C?Q.hitAreaHorizontal:Q.hitAreaVertical,C?A.centerInline(`0px`):Q.hitAreaCenteredY,w&&(C?Q.hitAreaWithinHandleX:Q.hitAreaWithinHandleY),!w&&se!=null&&(C?yd.hitAreaLengthX(se):yd.hitAreaLengthY(se)),r&&Q.disabled),...P}),l??(0,U.jsx)(`div`,{...tt(Ye(`resize-handle-pill`),He(Q.pill,C?Q.pillHorizontal:Q.pillVertical,ae===`center`?A.centerInline(`-50%`):C?yd.pillOffsetX(ae===`start`?-1:1):yd.pillOffsetY(ae===`start`?-1:1),C?Q.pillGrabHorizontal:Q.pillGrabVertical,a?Q.pillVisible:Q.pillHidden,!a&&ae!==`center`&&(C?ae===`start`?Q.pillBridgeRight:Q.pillBridgeLeft:ae===`start`?Q.pillBridgeDown:Q.pillBridgeUp),ce&&!D&&Q.pillHover,D&&Q.pillActive,r&&Q.disabled)),...P})]})}bd.displayName=`ResizeHandle`;function xd(e){let t=e.cloneNode(!0);for(let e of t.querySelectorAll(`[aria-hidden="true"]`))e.remove();for(let e of t.querySelectorAll(`img[alt], [aria-label]`)){let n=e.getAttribute(e.tagName===`IMG`?`alt`:`aria-label`)??``;e.replaceWith(t.ownerDocument.createTextNode(` ${n} `))}for(let e of t.querySelectorAll(`svg`)){let n=e.querySelector(`title`)?.textContent??``;e.replaceWith(t.ownerDocument.createTextNode(` ${n} `))}return(t.textContent??``).replace(/\s+/g,` `).trim()}function Sd(e){if(e==null)return;let t=e.getAttribute(`aria-labelledby`);if(t!=null&&t.trim()!==``){let n=t.split(/\s+/).map(t=>e.ownerDocument.getElementById(t)).filter(e=>e!=null).map(xd).join(` `).trim();if(n!==``)return n}let n=e.getAttribute(`aria-label`)?.trim();if(n)return n;let r=(`labels`in e?Array.from(e.labels??[]):[]).map(xd).join(` `).trim();if(r!==``)return r;if(e instanceof HTMLInputElement&&[`button`,`submit`,`reset`].includes(e.getAttribute(`type`)??``)){let t=e.value.trim();if(t!==``)return t}let i=xd(e);return i===``?e.getAttribute(`title`)?.trim()||void 0:i}Te({"@astryx.sideNav.label":`Side navigation`,"@astryx.sideNav.resizeSidebar":`Resize sidebar`});var Cd=160;function wd(e,t){let n=typeof e==`object`?e:{},r=typeof t==`object`?t:{},i=r.defaultIsCollapsed!==void 0||r.isCollapsed!==void 0||r.onCollapseChange!==void 0;return{isCollapsible:!!e||i,defaultIsCollapsed:(i?r.defaultIsCollapsed:n.defaultIsCollapsed)??!1,isCollapsed:i?r.isCollapsed:n.isCollapsed,onCollapsedChange:i?r.onCollapseChange:n.onCollapsedChange,conflict:Td(n,r)}}function Td(e,t){let n=[e.defaultIsCollapsed!==void 0&&`defaultIsCollapsed`,e.isCollapsed!==void 0&&`isCollapsed`,e.onCollapsedChange!==void 0&&`onCollapsedChange`].filter(e=>e!==!1),r=[t.defaultIsCollapsed!==void 0&&`defaultIsCollapsed`,t.isCollapsed!==void 0&&`isCollapsed`,t.onCollapseChange!==void 0&&`onCollapseChange`].filter(e=>e!==!1);return n.length===0||r.length===0?null:`${n.map(e=>`collapsible.${e}`).join(`, `)} and ${r.map(e=>`resizable.${e}`).join(`, `)} address the same collapse state. resizable wins; the collapsible ${n.length===1?`value is`:`values are`} ignored.`}function Ed(e,t){let n=typeof e==`object`?e:{},r=[n.defaultIsCollapsed!==void 0&&`defaultIsCollapsed`,n.isCollapsed!==void 0&&`isCollapsed`,n.onCollapsedChange!==void 0&&`onCollapsedChange`].filter(e=>e!==!1),i=t.props._collapsible??!1,a=[];return r.length>0&&a.push(`\`collapsible\` sets ${r.map(e=>`\`${e}\``).join(`, `)}, which the rail ignores: it follows the region.`),e!==void 0&&!!e!==i&&a.push(i?"`collapsible` is false, but the region is collapsible, so the rail still collapses.":"`collapsible` turns collapse on, but the region is not collapsible, so the rail never collapses."),{isCollapsible:i,defaultIsCollapsed:t.isCollapsed,isCollapsed:void 0,onCollapsedChange:void 0,conflict:a.length===0?null:`\`resizable\` is useResizable's output, which owns the collapse state and decides whether the rail collapses. ${a.join(` `)}`}}var Dd={root:{k1xSpc:`x78zum5`,kXwgrk:`xdt5ytf`,kZKoxP:`x5yr21d`,kzqmXN:`x1hfn5x7`,kWkggS:`xgcd1z6`,kB7OPa:`x9f619`,kVQacm:`xb3r6kr`,$$css:!0},rootCollapsed:{kzqmXN:`xhgvbfk`,$$css:!0},resizableHandle:{kInvED:`x1wfwxd8 x1l5jwag`,$$css:!0},topbar:{k1xSpc:`x78zum5`,kXwgrk:`x1q0g3np`,kGNEyG:`x6s0dn4`,kjj79g:`x1qughib`,kZKoxP:`xsdox4t`,kzqmXN:`xh8yej3`,kWkggS:`xgcd1z6`,kB7OPa:`x9f619`,kVQacm:`xb3r6kr`,$$css:!0}},Od=`sm`;function kd(e){let t=e?.timeline?.currentTime;return t==null?null:Number(t)}function Ad({header:e,topContent:t,children:n,footer:r,footerIcons:i,collapsible:a,resizable:o=!1,xstyle:s,className:c,style:l,"data-testid":u,ref:d,handleRef:f,...p}){let m=Be(),h=a!==void 0&&a,g=(typeof h==`object`?h:{}).hasButton??!0,v=xc(o)?o:null,y=typeof o==`object`&&!xc(o)?o:{},b=!!o,x=v==null?wd(h,!xc(o)&&o):Ed(a,v),{isCollapsible:S,onCollapsedChange:C}=x;it(`SideNav`,x.conflict??``,x.conflict!=null);let w=(0,z.useRef)(null),T=(0,z.useId)(),E=p.id??T,{ref:te}=ee(),D=(0,z.useRef)(null),ne=(0,z.useRef)(null),O=(0,z.useRef)(!1),re=(0,z.useCallback)(e=>{e!=null&&e!==ne.current&&(ne.current=e,D.current=kd(e.ownerDocument))},[]),ie=_(d,w,te,re),k=(0,z.useRef)({isCollapsed:!1,toggle:()=>{},isCollapsible:S}),ae=Dc({defaultSize:y.defaultWidth??260,minSize:y.minWidth??180,maxSize:y.maxWidth??480,collapsible:S,collapsedSize:Cd,autoSaveId:y.autoSaveId,defaultIsCollapsed:x.defaultIsCollapsed,isCollapsed:x.isCollapsed,onSizeChange:y.onWidthChange,onCollapseChange:C}),oe=v??ae,se=oe.isCollapsed,ce=(0,z.use)(Gu),A=(0,z.useRef)(null),j=(0,z.useCallback)(()=>A.current??w.current,[]),M=(0,z.useMemo)(()=>Ku(j),[j]),le=(0,z.useRef)(null),ue=(0,z.useRef)(null),de=(0,z.useCallback)(e=>{let t=ue.current,n=t?.rail.shown()??null;t?.rail.end(),t?.endContent(),ue.current=null;let r=O.current;O.current=!1;let i=kd(w.current?.ownerDocument);if(!r&&(i==null||i===D.current)){le.current=null,ce?.landContent();return}le.current=qu(ce?.rail()??j(),ce?.animatedLayout??M,n,e);let a=le.current;a!=null&&a.swap==null&&!e?ce?.landContent():ce?.beforeRailChange()},[ce,M,j]);(0,z.useLayoutEffect)(()=>{let e=le.current;if(le.current=null,e==null)return;let t=ce?.afterRailChange()??(()=>{});ue.current={rail:Xu(e,ce?.rail()??j(),ce?.animatedLayout??M,ce?.column()??null),endContent:t}},[se,ce,M,j]),(0,z.useLayoutEffect)(()=>()=>{ue.current?.rail.end(),ue.current?.endContent()},[]);let fe=(0,z.useRef)(new Map),pe=(0,z.useRef)(new Set),me=(0,z.useMemo)(()=>({registerRow:(e,t)=>(fe.current.set(e,t),()=>{fe.current.delete(e)}),registerToggle:e=>(pe.current.add(e),()=>{pe.current.delete(e)})}),[]),he=zc(se,e=>{let t=w.current,n=e?null:lt(t)??lt(A.current);if(n==null||t==null)return null;let r=Nd(t,fe.current),i=-1;r.forEach((e,t)=>{e.contains(n.element)&&(i=t)});let a=null;if(i<0){let e=jd(t,r),i=e.findIndex(e=>e.contains(n.element));a=i<0?null:{name:Md(e[i])}}return{focus:n,outside:a,wasOnToggle:[...pe.current].some(e=>e.contains(n.element)),isRowKept:i>=0&&fe.current.get(r[i])===!0,keptBefore:r.slice(0,Math.max(i,0)).filter(e=>fe.current.get(e)===!0).length}},(e,t)=>{let n=w.current;if(!t||n==null)return;let r=Nd(n,fe.current),i=e=>e==null?void 0:rt(e)[0],a=[...pe.current].find(e=>n.contains(e)),o;e.wasOnToggle?o=a:e.outside!=null&&(o=jd(n,r).find(t=>Md(t)===e.outside?.name));let s=e.isRowKept?i(r[e.keptBefore]):o??a??i(r[e.keptBefore-1])??i(r[e.keptBefore]);st(e.focus,s)}),ge=(0,z.useCallback)(e=>{he(e),de(e)},[he,de]),N=(0,z.useCallback)(()=>{let e=!se;O.current=!0,k.current={...k.current,isCollapsed:e},e?oe.collapse():oe.expand()},[se,oe]),P=b&&!se;k.current={isCollapsed:se,toggle:N,isCollapsible:S};let _e={isCollapsed:se,toggle:N,isCollapsible:S};(0,z.useImperativeHandle)(f,()=>({getCollapseState:()=>k.current}),[]);let ve=Ml();if(ve===`topbar`)return(0,U.jsxs)(`div`,{"data-testid":u,...tt(Ye(`side-nav`,{mode:`topbar`}),He(Dd.topbar,s),c,l),children:[e,(0,U.jsx)(`div`,{className:`x78zum5 x6s0dn4 xzye2dw xvc5jky`,children:(0,U.jsx)(Ee,{value:Od,children:i})})]});let ye=xe(r)||xe(i);if(ve===`drawer`)return(0,U.jsxs)(Al,{header:e,"data-testid":u,xstyle:s,className:c,style:l,...p,children:[t,n,ye&&(0,U.jsxs)(`div`,{className:`x78zum5 xdt5ytf xr1yuqi x1txdalj x1xye8es`,children:[r,xe(i)&&(0,U.jsx)(`div`,{className:`x78zum5 x6s0dn4 xzye2dw`,children:(0,U.jsx)(Ee,{value:Od,children:i})})]})]});if(ve===`drawer-content`)return(0,U.jsxs)(U.Fragment,{children:[t,n,ye&&(0,U.jsxs)(`div`,{className:`x78zum5 xdt5ytf xr1yuqi x1txdalj x1xye8es`,children:[r,xe(i)&&(0,U.jsx)(`div`,{className:`x78zum5 x6s0dn4 xzye2dw`,children:(0,U.jsx)(Ee,{value:Od,children:i})})]})]});let be=xe(e)||xe(t),F=xe(r)||xe(i),I=S&&g,Se=b?{...l??{},width:se?void 0:oe.size}:l,Ce=(0,U.jsxs)(`nav`,{ref:ie,role:`navigation`,"aria-label":m(`@astryx.sideNav.label`),"data-testid":u,...tt(Ye(`side-nav`),He(Dd.root,se&&Dd.rootCollapsed,s),c,Se),...p,id:E,children:[be&&(0,U.jsxs)(`div`,{...{0:{className:`x78zum5 xdt5ytf x2lah0s x7wzq59 x13vifvy x1vjfegm xgcd1z6 x1xye8es x1wesfrj xf314gf x1txdalj`},1:{className:`x78zum5 xdt5ytf x2lah0s x7wzq59 x13vifvy x1vjfegm xgcd1z6 x1xye8es x1wesfrj xf314gf x1txdalj x6s0dn4`}}[!!se|0],children:[e,xe(t)&&(0,U.jsx)(`div`,{children:t})]}),(0,U.jsx)(`div`,{...{0:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x1xye8es x1wesfrj`},4:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x78zum5 xdt5ytf x6s0dn4 x1xye8es x1wesfrj`},2:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf xfsso4q x1wesfrj`},6:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x78zum5 xdt5ytf x6s0dn4 xfsso4q x1wesfrj`},1:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x1xye8es xy143xn`},5:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x78zum5 xdt5ytf x6s0dn4 x1xye8es xy143xn`},3:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf xfsso4q xy143xn`},7:{className:`x98rzlu x1odjw0f x6ikm8r xf314gf x78zum5 xdt5ytf x6s0dn4 xfsso4q xy143xn`}}[!!se<<2|!!be<<1|!!F],children:n}),(F||I)&&(0,U.jsxs)(`div`,{...{0:{className:`x78zum5 xdt5ytf x2lah0s xr1yuqi x7wzq59 x1ey2m1c xgcd1z6 x1txdalj xf314gf xfsso4q x1wesfrj`},1:{className:`x78zum5 xdt5ytf x2lah0s xr1yuqi x7wzq59 x1ey2m1c xgcd1z6 x1txdalj xf314gf x1wesfrj xexx8yu x6s0dn4`}}[!!se|0],children:[r,(0,U.jsx)(`div`,{...{0:{className:`x78zum5 x6s0dn4 xzye2dw`},1:{className:`x78zum5 x6s0dn4 xzye2dw x3ieub6`}}[!!se|0],children:(0,U.jsxs)(Ee,{value:Od,children:[I&&(0,U.jsx)(fd,{}),i]})})]})]}),we=b?(0,U.jsxs)(`div`,{ref:A,className:`x1n2onr6 x78zum5 x2lah0s x5yr21d x7giv3`,children:[Ce,P&&(0,U.jsx)(bd,{"data-testid":`astryx-sidenav-resize-handle`,direction:`horizontal`,position:`overlay`,pillPlacement:`end`,isAlwaysVisible:!1,xstyle:Dd.resizableHandle,resizable:v??ae.props,label:m(`@astryx.sideNav.resizeSidebar`),"aria-controls":E})]}):Ce,Te=(0,U.jsxs)(U.Fragment,{children:[(0,U.jsx)(Rc,{value:se,onBeforeChange:ge}),we]});return S?(0,U.jsx)(ud,{value:me,children:(0,U.jsx)(cd,{value:_e,children:Te})}):Te}function jd(e,t){return rt(e).filter(e=>!t.some(t=>t.contains(e)))}function Md(e){return Sd(e)??``}function Nd(e,t){return[...t.keys()].filter(t=>e.contains(t)&&t.closest(`[popover]`)==null).sort((e,t)=>e.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_FOLLOWING?-1:1)}Ad.displayName=`SideNav`;function Pd(e){return{onClick:e=>{e.preventDefault(),e.stopPropagation()},onKeyDown:t=>{t.key===`Enter`||t.key===` `?t.preventDefault():e?.(t)}}}function Fd(e,t){return t?Object.fromEntries(Object.entries(e).filter(([e])=>!/^on[A-Z]/.test(e)||/^on(Focus|Blur)/.test(e))):e}var Id={item:{k1xSpc:`x78zum5`,kGNEyG:`x6s0dn4`,kOIVth:`x1txdalj`,kzqmXN:`xh8yej3`,kZKoxP:`x1ueg155`,kg3NbH:`xf314gf`,k8WAf4:`xt970qd`,kaIpWk:`xh6dtrn`,kMzoRj:`xc342km`,ksu8eU:`xng3xce`,kWkggS:`xjbqb8w`,kMwMTN:`x1tgivj0`,kybGjl:`x1hl2dhg`,kkrTdU:`x1ypdohk x16khyan`,kMv6JI:`xjb2p0i`,kGuDYH:`xcr08ib`,k63SB2:`x1sodnla`,kLWn49:`x1kq96og`,k9WMMc:`x1yc453h`,kB7OPa:`x9f619`,$$css:!0},selected:{kWkggS:`x17x4s8c x1jzqe4`,kMwMTN:`x1k5gbb1`,k63SB2:`x1e4wzip`,kRL5z6:`xcu6dpe x1d42zcn`,kSReZ0:`xjtnuge x1vo6n7o`,kNWwhB:`x7qqc7o x1r4mieh`,kCVruK:`x1bev2v5`,kjMJ7V:`xf3x42i x1pjcgc3`,kSPMfz:`x524tya`,$$css:!0},disabled:{kMwMTN:`xnbbluu`,kkrTdU:`xt0e3qv`,kfzvcC:`x47corl`,$$css:!0},sm:{kZKoxP:`x6k0iem`,kg3NbH:`x7a5moj`,$$css:!0},md:{kZKoxP:`x1ueg155`,kg3NbH:`xf314gf`,$$css:!0},lg:{kZKoxP:`xssyfek`,kg3NbH:`xf314gf`,$$css:!0}};function Ld(e){let{closeMobileNav:t}=Jc();return n=>e?(...e)=>{n?.(...e),t()}:n}Te({"@astryx.sideNavItem.collapse":`Collapse {label}`,"@astryx.sideNavItem.expand":`Expand {label}`,"@astryx.sideNavItem.submenuLabel":`{label} submenu`});var Rd={root:{k1xSpc:`x78zum5`,kXwgrk:`xdt5ytf`,kzqmXN:`xh8yej3`,$$css:!0},itemCollapsed:{kjj79g:`xl56j7k`,kzqmXN:`x805d0l`,kg3NbH:`xnjsko4`,$$css:!0},itemCollapsedSm:{kzqmXN:`x1fvqzum`,$$css:!0},itemCollapsedLg:{kzqmXN:`x1m747yf`,$$css:!0},expandChevron:{k1xSpc:`x3nfvp2`,kGNEyG:`x6s0dn4`,kjj79g:`xl56j7k`,kzqmXN:`x1mtvzj0`,kZKoxP:`xvpftlf`,kGuDYH:`x1qlqyl8`,k1ekBW:`x11xpdln`,kIyJzY:`xuedmi6 x12w9bfk`,kAMwcw:`xlr8y92`,kmuXW:`x2lah0s`,$$css:!0},expandChevronExpanded:{k3aq6I:`x19jd1h0`,$$css:!0},expandToggle:{kzqmXN:`x1fvqzum`,kZKoxP:`x6k0iem`,k1xSpc:`x3nfvp2`,kGNEyG:`x6s0dn4`,kjj79g:`xl56j7k`,kmuXW:`x2lah0s`,kmVPX3:`x1717udv`,kogj98:`x1ghz6dp`,kMzoRj:`xc342km`,ksu8eU:`xng3xce`,kWkggS:`xjbqb8w`,kMwMTN:`x1heor9g`,kkrTdU:`x1ypdohk x16khyan`,kaIpWk:`xh6dtrn`,$$css:!0},splitAction:{k1xSpc:`x78zum5`,kGNEyG:`x6s0dn4`,kSGwAc:`xkh2ocl`,kOIVth:`x1txdalj`,kUk6DE:`x98rzlu`,k7Eaqz:`xeuugli`,kMwMTN:`x1heor9g`,kybGjl:`x1hl2dhg`,kmVPX3:`x1717udv`,kogj98:`x1ghz6dp`,kMzoRj:`xc342km`,ksu8eU:`xng3xce`,kWkggS:`xjbqb8w`,kMv6JI:`xjb2p0i`,kGuDYH:`x1qlqyl8`,k63SB2:`x1pd3egz`,kLWn49:`x15bjb6t`,k9WMMc:`x1yc453h`,kkrTdU:`x1ypdohk x16khyan`,$$css:!0},popoverGap:{keTefX:`x11g1kdw`,k71WvV:`xnur1sd`,$$css:!0}},zd={kZCmMZ:`xnvo3vl`,$$css:!0},Bd={kZCmMZ:`xnvo3vl`,$$css:!0},Vd={kZCmMZ:`xnvo3vl`,$$css:!0},Hd={sm:e=>[zd,{"--x-paddingInlineStart":(e=>typeof e==`number`?e+`px`:e??void 0)(`calc(${Ue[`--spacing-1`]} + ${e} * ${Ue[`--spacing-6`]})`)}],md:e=>[Bd,{"--x-paddingInlineStart":(e=>typeof e==`number`?e+`px`:e??void 0)(`calc(${Ue[`--spacing-2`]} + ${e} * ${Ue[`--spacing-6`]})`)}],primary:e=>[Vd,{"--x-paddingInlineStart":(e=>typeof e==`number`?e+`px`:e??void 0)(`calc(${e} * ${Ue[`--spacing-6`]})`)}]},Ud=(0,z.createContext)(0);Ud.displayName=`SideNavItemDepthContext`;var Wd=`sm`,Gd={isCollapsed:!1,toggle:()=>{},isCollapsible:!1};function Kd({href:e,as:t,isDisabled:n,onClick:r,ref:i,children:a,...o}){let s=Fe(t);return e&&!n?(0,U.jsx)(s,{ref:i,href:e,onClick:r,...o,children:a}):(0,U.jsx)(`button`,{ref:i,type:`button`,onClick:r,...Fd(o,!!n),...n?{"aria-disabled":!0,...Pd()}:null,children:a})}function qd(...e){let t=e.filter(Boolean).join(` `);return t===``?void 0:t}function Jd({as:e,label:t,icon:n,selectedIcon:r,isSelected:i=!1,isDisabled:a=!1,href:o,onClick:s,endContent:c,actions:l,children:u,collapsible:d,size:f=`md`,"data-testid":p,ref:m,xstyle:h,...g}){let v=ge(),y=Be(),{isCollapsed:b}=ld(),x=Ml(),S=Ld(x===`drawer`||x===`drawer-content`),C=(0,z.useId)(),w=!!u,T=(0,z.use)(Ud),E=pe({hasLightDismiss:!0,hasAutoFocus:!0,hasCloseButton:!1,dialogLabel:y(`@astryx.sideNavItem.submenuLabel`,{label:t})}),ee=_(m,E.triggerRef),te=b&&!w,D=St({placement:`end`,focusTrigger:`always`,isEnabled:te,isOpen:te?void 0:!1}),ne=_(m,D.ref),O=(0,z.use)(ud),re=!!n&&T===0,ie=(0,z.useCallback)(e=>e==null||O==null?void 0:O.registerRow(e,re),[O,re]),k=_(ie,D.ref),ae=(0,z.useMemo)(()=>typeof d==`object`?d:{},[d]),oe=w&&d!==!1,se=ae.isCollapsed,ce=se!==void 0,[A,j]=(0,z.useState)(ae.defaultIsCollapsed??!1),M=ce?se:A,le=(0,z.useCallback)(()=>{let e=!M;ce||j(e),ae.onCollapsedChange?.(e)},[M,ce,ae]),ue=i&&r?r:n,de=oe&&(!!o||!!s)&&!b,fe=!!l,me=e=>{a?e.preventDefault():oe&&!de&&!b?(e.preventDefault(),le()):S(s)?.(e)},he=e=>{e.preventDefault(),e.stopPropagation(),le()},{triggerProps:N,contentProps:P}=Pe({show:E.show,hide:E.hide,isOpen:E.isOpen,isEnabled:b&&w,clickGuardMs:0,ownsFocus:!1});if(b&&!n)return null;if(b){let n=ue&&Oe(ue,{size:`sm`,color:i?`inherit`:a?`disabled`:`secondary`}),r={...tt(Ye(`side-nav-item`,{size:f,selected:i?`selected`:null,disabled:a?`disabled`:null}),Ze.focusVisible(Id.item,De.backgroundColor,Id[f],Rd.itemCollapsed,f===`sm`&&Rd.itemCollapsedSm,f===`lg`&&Rd.itemCollapsedLg,i&&Id.selected,a&&Id.disabled),g.className),...v},s=g[`aria-label`]!=null&&g[`aria-label`].trim()!==``?g[`aria-label`]:t;if(w)return(0,U.jsxs)(`div`,{ref:ie,...He(Rd.root,h),children:[(0,U.jsx)(`button`,{ref:ee,type:`button`,...g,...N,"aria-label":s,"data-testid":p,...E.triggerProps,...r,children:n}),E.render((0,U.jsxs)(`div`,{className:`x1hviunn xu0wf1k x7a5moj xfb3i0g`,onMouseEnter:P.onMouseEnter,onMouseLeave:P.onMouseLeave,onClick:()=>E.hide(),children:[(0,U.jsx)(`div`,{className:`xf314gf xu0wf1k x141an7d x2mo6ok xv1l7n4 x1ltkj2j`,children:t}),(0,U.jsx)(cd,{value:Gd,children:(0,U.jsx)(Ud,{value:0,children:u})})]}),{placement:`end`,alignment:`start`,xstyle:Rd.popoverGap})]});let c=a,l={"aria-current":i?`page`:void 0,"aria-describedby":qd(g[`aria-describedby`],D.describedBy),"aria-disabled":a||void 0,"aria-label":s,"data-testid":p},d=(0,U.jsx)(Kd,{ref:c?m:ne,href:o,as:e,isDisabled:a,onClick:me,...g,...l,...r,children:n});return(0,U.jsxs)(`div`,{ref:c?k:ie,...He(Rd.root,h),children:[d,D.renderTooltip(t)]})}let _e=(0,U.jsxs)(U.Fragment,{children:[ue&&Oe(ue,{size:`sm`,color:i?`inherit`:a?`disabled`:`secondary`}),!b&&(0,U.jsx)(`span`,{className:`x98rzlu xeuugli xb3r6kr xlyipyv xuxw1ft`,children:t}),!b&&xe(c)&&(0,U.jsx)(`span`,{className:`x2lah0s x78zum5 x6s0dn4`,children:c}),!b&&oe&&!de&&(0,U.jsx)(ke,{icon:`chevronDown`,size:`lg`,color:`inherit`,xstyle:[Rd.expandChevron,!M&&Rd.expandChevronExpanded]})]}),ve=Ye(`side-nav-item`,{size:f,selected:i?`selected`:null,disabled:a?`disabled`:null}),ye=[Id.item,De.backgroundColor,Id[f],i&&Id.selected,a&&Id.disabled],be=T>0&&Hd[f===`sm`?`sm`:`md`](T),F=T>0&&Hd.primary(T),I={className:g.className,style:g.style},Se={...tt(ve,He(...ye)),...v},Ce={...tt(ve,tt(Ze.focusVisible(...ye,be),I)),...v},we={...tt(ve,Ze.focusWithinFirstChild(...ye)),...v},Te=de||fe,Ae;if(Te){let n=de?{"aria-current":i?`page`:void 0}:{"aria-current":i?`page`:void 0,"aria-disabled":a||void 0,"aria-expanded":oe?!M:void 0,"aria-controls":oe?`${C}-children`:void 0};Ae=(0,U.jsxs)(`div`,{"data-testid":p,...fe?we:Se,children:[(0,U.jsx)(Kd,{ref:m,href:o,as:e,isDisabled:a,onClick:me,...g,...n,...tt(fe?Ze.suppressed(Rd.splitAction,F):Ze.focusVisible(Rd.splitAction,F),I),children:_e}),de&&(0,U.jsx)(`button`,{type:`button`,onClick:he,"aria-label":y(M?`@astryx.sideNavItem.expand`:`@astryx.sideNavItem.collapse`,{label:t}),"aria-expanded":!M,"aria-controls":`${C}-children`,...v,...Ze.focusVisible(Rd.expandToggle,De.backgroundColor),children:(0,U.jsx)(ke,{icon:`chevronDown`,size:`lg`,color:`inherit`,xstyle:[Rd.expandChevron,!M&&Rd.expandChevronExpanded]})}),fe&&(0,U.jsx)(`span`,{className:`x2lah0s x78zum5 x6s0dn4 xzye2dw x67bb7w`,children:(0,U.jsx)(Ee,{value:Wd,children:l})})]})}else{let t={"aria-current":i?`page`:void 0,"aria-disabled":a||void 0,"aria-expanded":oe?!M:void 0,"aria-controls":oe?`${C}-children`:void 0,"data-testid":p};Ae=(0,U.jsx)(Kd,{ref:m,href:o,as:e,isDisabled:a,onClick:me,...g,...t,...Ce,children:_e})}return(0,U.jsxs)(`div`,{ref:ie,...He(Rd.root,h),children:[Ae,w&&!b&&(0,U.jsx)(`div`,{id:`${C}-children`,role:`group`,"aria-labelledby":`${C}-label`,"aria-hidden":M,inert:M?!0:void 0,...{0:{className:`xrvj5dj x1tu4anv x1rea2x4 x6kj92 xd00j3c x1eizr8u xsr0bmt xostmdb x1k6wr6r xlr8y92`},1:{className:`xrvj5dj x6kj92 xd00j3c x1eizr8u xsr0bmt xlr8y92 xihq33y xb3r6kr xf1rc6`}}[!!M|0],children:(0,U.jsxs)(`div`,{className:`x2lwn1j`,children:[(0,U.jsx)(`span`,{id:`${C}-label`,hidden:!0,children:t}),(0,U.jsx)(Ud,{value:T+1,children:u})]})})]})}Jd.displayName=`SideNavItem`;var Yd=`(width < 1024px)`;function Xd(e){return(0,U.jsx)(sd,{className:`app-shell`,height:`auto`,variant:`section`,contentPadding:0,topNav:e.topBar,sideNav:e.sideNav,mobileNav:{breakpoint:`lg`,hasToggle:!1,isOpen:e.drawerOpen,onOpenChange:e.onDrawerOpenChange,content:e.drawer},children:e.children})}var Zd=z.forwardRef((e,t)=>(0,U.jsx)(Ad,{ref:t,"aria-label":e.label,header:e.header,footer:e.footer,className:e.className,onKeyDown:e.onKeyDown,children:e.children}));Zd.displayName=`ShellSideNav`;function Qd(e){return(0,U.jsx)(jl,{value:`drawer`,children:(0,U.jsx)(Ad,{header:e.label,topContent:e.topContent,footer:e.footer,className:e.className,onKeyDown:e.onKeyDown,"data-testid":e.testId,children:e.children})})}function $d(e){return(0,U.jsx)(Jd,{label:e.label,icon:Ve(e.icon),href:e.href,isSelected:e.isSelected,"data-testid":e.testId,onClick:t=>{t.metaKey||t.ctrlKey||t.shiftKey||t.altKey||(t.preventDefault(),e.onSelect())}})}function ef(e){return(0,U.jsx)(Yc,{label:e.label,"data-testid":e.testId,className:e.className,children:(0,U.jsx)(N,{name:`menu`})})}var tf=[{id:`models`,key:`nav.models`,icon:`models`},{id:`chat`,key:`nav.chat`,icon:`chat`},{id:`activity`,key:`nav.activity`,icon:`activity`},{id:`settings`,key:`nav.settings`,icon:`settings`}],nf=[{id:`command`,keys:[`⌘/Ctrl`,`K`],separator:`+`,key:`help.shortcut.command`},{id:`new-chat`,keys:[`⌘/Ctrl`,`N`],separator:`+`,key:`help.shortcut.new_chat`},{id:`send`,keys:[`⌘/Ctrl`,`Enter`],separator:`+`,key:`help.shortcut.send`},{id:`escape`,keys:[`Esc`],separator:`+`,key:`help.shortcut.escape`},{id:`navigate`,keys:[`[`,`]`],separator:`/`,key:`help.shortcut.navigate`},{id:`help`,keys:[`?`],separator:`+`,key:`help.shortcut.help`}];function rf(e){let t=(0,z.useRef)(null),n=(0,z.useRef)(e.onCommand),r=(0,z.useRef)(e.onHelp),i=(0,z.useRef)(e.onNewChat),[a,o]=(0,z.useState)(!1);(0,z.useEffect)(()=>{n.current=e.onCommand,r.current=e.onHelp,i.current=e.onNewChat},[e.onCommand,e.onHelp,e.onNewChat]),(0,z.useEffect)(()=>{let e=e=>{let t=e.target,a=t instanceof HTMLElement&&(t.isContentEditable||!!t.closest(`[role="combobox"], [role="listbox"]`)||[`INPUT`,`TEXTAREA`,`SELECT`].includes(t.tagName)),s=t instanceof HTMLElement&&!!t.closest(`dialog[open], [role="dialog"][aria-modal="true"]`);if(e.isComposing||e.altKey||a||s)return;let c=e.metaKey||e.ctrlKey;if(c&&e.key.toLowerCase()===`k`&&(e.preventDefault(),o(!1),n.current()),c&&!e.shiftKey&&e.key.toLowerCase()===`n`){if(e.preventDefault(),e.repeat)return;o(!1),i.current()}(e.key===`?`||e.key===`/`&&e.shiftKey)&&(e.preventDefault(),o(!1),r.current())};return window.addEventListener(`keydown`,e),()=>window.removeEventListener(`keydown`,e)},[]),(0,z.useEffect)(()=>{if(!a||typeof window.matchMedia!=`function`)return;let e=window.matchMedia(Yd),n=()=>{if(e.matches)return;o(!1);let n=0,r=()=>{let e=t.current;if(!e&&n++<120){requestAnimationFrame(r);return}let i=document.activeElement;(!i||i===document.body||i.closest(`dialog`))&&(e?.querySelector(`a[aria-current="page"]`)??e?.querySelector(`a`))?.focus()};requestAnimationFrame(r)};return e.addEventListener(`change`,n),()=>e.removeEventListener(`change`,n)},[a]);let s=t=>{if(t.key!==`[`&&t.key!==`]`||t.altKey||t.metaKey||t.ctrlKey||t.nativeEvent.isComposing)return;t.preventDefault();let n=t.currentTarget,r=tf[(tf.findIndex(t=>t.id===e.route)+(t.key===`]`?1:-1)+tf.length)%tf.length];e.onRouteChange(r.id),requestAnimationFrame(()=>n.querySelector(`a[href="#${r.id}"]`)?.focus())},c=t=>{e.onRouteChange(t),o(!1)},l=tf.map(t=>(0,U.jsx)($d,{label:L(e.locale,t.key),icon:t.icon,href:`#${t.id}`,isSelected:e.route===t.id,testId:R(t.key),onSelect:()=>c(t.id)},t.id)),u=(0,U.jsxs)(`div`,{className:`app-toolbar material-glass`,children:[(0,U.jsx)(ef,{label:L(e.locale,`toolbar.menu`),testId:R(`toolbar.menu`),className:`ds-icon-button mobile-menu-button`}),(0,U.jsx)(sf,{locale:e.locale,className:`brand-mark toolbar-brand`,onHome:()=>c(`models`)}),(0,U.jsxs)(`div`,{className:`toolbar-title`,children:[(0,U.jsx)(`p`,{id:`app-title`,title:L(e.locale,`app.title`),"data-testid":R(`app.title`),children:L(e.locale,`app.title`)}),(0,U.jsx)(`span`,{title:L(e.locale,`app.subtitle`),"data-testid":R(`app.subtitle`),children:L(e.locale,`app.subtitle`)})]}),(0,U.jsx)(af,{locale:e.locale,models:e.loadedModels,onOpenModel:e.onOpenModel,onMore:e.onCommand}),(0,U.jsxs)(`div`,{className:`toolbar-actions`,children:[e.sessionAction,(0,U.jsx)(ae,{label:L(e.locale,`toolbar.command`),icon:`command`,onClick:e.onCommand,"data-testid":R(`toolbar.command`)}),(0,U.jsx)(ae,{label:L(e.locale,`toolbar.help`),icon:`help`,onClick:e.onHelp,"data-testid":R(`toolbar.help`)})]})]});return(0,U.jsx)(Xd,{topBar:u,sideNav:(0,U.jsx)(Zd,{ref:t,label:L(e.locale,`nav.primary`),footer:(0,U.jsx)(cf,{locale:e.locale,connection:e.connection}),className:`app-sidebar desktop-sidebar material-glass`,onKeyDown:s,children:l}),drawer:(0,U.jsx)(Qd,{label:L(e.locale,`nav.primary`),topContent:(0,U.jsx)(sf,{locale:e.locale,onHome:()=>c(`models`)}),footer:(0,U.jsx)(cf,{locale:e.locale,connection:e.connection}),className:`app-sidebar sheet-sidebar`,testId:`mobile-nav-sheet`,onKeyDown:s,children:l}),drawerOpen:a,onDrawerOpenChange:o,children:(0,U.jsxs)(`section`,{className:`app-content-grid ${e.inspector?`has-inspector`:``}`.trim(),children:[(0,U.jsx)(Ne,{className:`app-content`,children:e.children}),e.inspector]})})}function af(e){let t=e.models??[],n=t.slice(0,3),r=t.length-n.length,i=e.models===null?`toolbar.loaded.unknown`:t.length===0?`toolbar.loaded.none`:null;return(0,U.jsx)(`div`,{className:`toolbar-loaded`,role:`group`,"aria-label":L(e.locale,`toolbar.loaded.label`),"data-testid":R(`toolbar.loaded.label`),onFocus:of,children:i===null?(0,U.jsxs)(U.Fragment,{children:[(0,U.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":R(`toolbar.loaded.count`),children:L(e.locale,`toolbar.loaded.count`,{count:String(t.length)})}),n.map(t=>(0,U.jsx)(de,{tone:`ghost`,className:`toolbar-chip`,title:t.name,"data-testid":`toolbar-loaded-chip`,onClick:()=>e.onOpenModel(t.id),children:(0,U.jsxs)(`span`,{className:`toolbar-chip-content`,children:[(0,U.jsx)(`span`,{className:`truncate`,children:t.name}),` `,(0,U.jsx)(M,{state:t.state,appearance:`inline`,children:t.stateLabel})]})},t.id)),r>0?(0,U.jsx)(de,{tone:`ghost`,className:`toolbar-chip toolbar-chip-more`,"aria-label":L(e.locale,`toolbar.loaded.more_label`,{count:String(r)}),"data-testid":R(`toolbar.loaded.more`),onClick:e.onMore,children:L(e.locale,`toolbar.loaded.more`,{count:String(r)})}):null]}):(0,U.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":R(i),children:L(e.locale,i)})})}function of(e){let t=e.target;t instanceof HTMLElement&&typeof t.scrollIntoView==`function`&&t.matches(`:focus-visible`)&&t.scrollIntoView({block:`nearest`,inline:`nearest`})}function sf(e){return(0,U.jsxs)(`a`,{className:e.className??`brand-mark`,href:`#models`,"aria-label":L(e.locale,`nav.home`),onClick:t=>{t.preventDefault(),e.onHome()},children:[(0,U.jsx)(`span`,{className:`brand-symbol`,"aria-hidden":`true`}),(0,U.jsx)(`span`,{className:`brand-name`,children:`mlxcel`})]})}function cf(e){return(0,U.jsxs)(`footer`,{className:`app-sidebar-footer`,children:[(0,U.jsxs)(`div`,{className:`connection-status`,children:[(0,U.jsx)(`span`,{className:`connection-dot`,"data-state":e.connection.state,"aria-hidden":`true`}),(0,U.jsx)(`span`,{"data-testid":R(`connection.ready`),children:e.connection.label})]}),e.connection.details?(0,U.jsxs)(`details`,{className:`connection-details`,"data-testid":R(`connection.footer.details`),children:[(0,U.jsx)(`summary`,{children:L(e.locale,`connection.footer.details`)}),e.connection.summary?(0,U.jsx)(`p`,{"data-testid":R(`connection.authenticated.detail`),children:e.connection.summary}):null,(0,U.jsx)(`p`,{"data-testid":R(`connection.footer.instance`),children:L(e.locale,`connection.footer.instance`,{id:e.connection.details.instance})}),(0,U.jsx)(`p`,{"data-testid":R(`connection.footer.sequence`),children:L(e.locale,`connection.footer.sequence`,{sequence:e.connection.details.sequence})})]}):null]})}var lf=[`mlxcel`,`glass`],uf=[`system`,`light`,`dark`],df={mlxcel:`mlxcel`,glass:`mlxcel-glass`},ff=`mlxcel`,pf=`system`,mf=`(prefers-color-scheme: dark)`;function hf(e){return typeof e==`string`&&lf.includes(e)}function gf(e){return typeof e==`string`&&uf.includes(e)}function _f(){try{return typeof window>`u`||typeof window.matchMedia!=`function`?`light`:window.matchMedia(`(prefers-color-scheme: dark)`).matches?`dark`:`light`}catch{return`light`}}function vf(e,t=_f()){return e===`system`?t:e}function yf(e,t,n=_f()){return{family:e,astryxTheme:df[e],scheme:vf(t,n),preference:t}}function bf(e){if(typeof window>`u`||typeof window.matchMedia!=`function`)return()=>void 0;let t;try{t=window.matchMedia(mf)}catch{return()=>void 0}let n=t=>e(t.matches?`dark`:`light`);return t.addEventListener(`change`,n),()=>t.removeEventListener(`change`,n)}var xf={themeFamily:ff,colorScheme:pf,material:`glass`,glassIntensity:35,reduceMotion:!1,reduceTransparency:!1,highContrast:`system`,locale:`en`},Sf=`mlxcel.webui.appearance`;function Cf(e){let t=typeof e==`number`&&Number.isFinite(e)?e:xf.glassIntensity;return Math.max(0,Math.min(100,Math.round(t)))}function wf(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function Tf(e){return e===`glass`||e===`tinted`||e===`opaque`}function Ef(e){return e===`system`||e===`on`||e===`off`}function Df(e){return Ef(e)?e:e===!0?`on`:e===!1?`off`:xf.highContrast}function Of(e){return gf(e.colorScheme)?e.colorScheme:e.colorScheme===void 0&&gf(e.theme)?e.theme:xf.colorScheme}function kf(e){return e===`en`||e===`ko`}function Af(){if(typeof window>`u`)return xf;let e;try{e=window.localStorage.getItem(Sf)}catch{return xf}if(!e)return xf;try{let t=JSON.parse(e);return wf(t)?{themeFamily:hf(t.themeFamily)?t.themeFamily:xf.themeFamily,colorScheme:Of(t),material:Tf(t.material)?t.material:xf.material,glassIntensity:Cf(t.glassIntensity),reduceMotion:t.reduceMotion===!0,reduceTransparency:t.reduceTransparency===!0,highContrast:Df(t.highContrast),locale:kf(t.locale)?t.locale:xf.locale}:xf}catch{return xf}}function jf(e){try{window.localStorage.setItem(Sf,JSON.stringify(e))}catch{}}function Mf(){return typeof CSS>`u`||typeof CSS.supports!=`function`?!1:CSS.supports(`backdrop-filter: blur(1px)`)||CSS.supports(`-webkit-backdrop-filter: blur(1px)`)}function Nf(e,t,n=_f()){let r=yf(t.themeFamily,t.colorScheme,n);e.dataset.theme=r.scheme,e.dataset.astryxTheme=r.astryxTheme,e.dataset.themeFamily=r.family,e.dataset.colorScheme=r.preference,e.dataset.material=t.reduceTransparency?`opaque`:t.material,e.dataset.glassIntensity=String(Cf(t.glassIntensity)),e.dataset.reduceMotion=String(t.reduceMotion),e.dataset.reduceTransparency=String(t.reduceTransparency),e.dataset.highContrast=t.highContrast,e.dataset.backdropFilter=Mf()?`supported`:`unsupported`,e.lang=t.locale}var Pf={base:{k1xSpc:`xjp7ctv`,kMwMTN:`x1tgivj0`,kMv6JI:`x9ynric`,$$css:!0},light:{kQNsl9:`x19aimcq`,$$css:!0},dark:{kQNsl9:`xntwwlm`,$$css:!0},system:{kQNsl9:`x108lcm5`,$$css:!0},reducedMotion:{"--_astryx-reduced-motion":`x1e3mt4b`,$$css:!0}},Ff=z.createContext(!1);Ff.displayName=`ThemeNestingContext`;var If=new Set,Lf=`@layer reset, astryx-base, astryx-theme;`;function Rf(e){let t=(0,z.useId)();(0,z.useInsertionEffect)(()=>{if(e.__built)return;let n=`astryx-theme-${e.name}`;if(If.has(n))return;`${e.name}`,`${e.name}${e.name}${e.name}${e.name}`;let{prose:r,component:i}=Gr(e);If.add(n);let a=[()=>If.delete(n)];if(r){let n=document.createElement(`style`);n.setAttribute(Je(`theme-prose`),e.name),n.setAttribute(Je(`id`),t),n.textContent=`${Lf}\n@layer reset {\n${r}\n}`,document.head.appendChild(n)}if(i){let n=document.createElement(`style`);n.setAttribute(Je(`theme`),e.name),n.setAttribute(Je(`id`),t),n.textContent=`${Lf}\n@layer astryx-theme {\n${i}\n}`,document.head.appendChild(n)}return(r||i)&&a.push(()=>{let n=document.querySelector(`style[${Je(`theme-prose`)}="${e.name}"][${Je(`id`)}="${t}"]`),r=document.querySelector(`style[${Je(`theme`)}="${e.name}"][${Je(`id`)}="${t}"]`);n?.remove(),r?.remove()}),()=>{for(let e of a)e()}},[e,t])}function zf(e,t,n,r){O(()=>{if(!e&&typeof document<`u`)return t===`light`||t===`dark`?document.documentElement.setAttribute(`data-theme`,t):document.documentElement.removeAttribute(`data-theme`),document.documentElement.setAttribute(Je(`theme`),n),Zi(document.documentElement,r),r?document.documentElement.style.setProperty(he,`1`):document.documentElement.style.removeProperty(he),()=>{document.documentElement.removeAttribute(`data-theme`),document.documentElement.removeAttribute(Je(`theme`)),Zi(document.documentElement,!1),document.documentElement.style.removeProperty(he)}},[e,t,n,r])}function Bf({theme:e,mode:t=`system`,children:n}){let r=(0,z.use)(Ff);me(e),Rf(e);let i=oe(e.__axes?.motion).reducedMotion===`always`;zf(r,t,e.name,i);let a=t===`dark`?Pf.dark:t===`light`?Pf.light:Pf.system,o=(0,z.useMemo)(()=>({theme:e,mode:t}),[e,t]);return(0,U.jsx)(Re,{value:o,children:(0,U.jsx)(Ff,{value:!0,children:(0,U.jsx)(`div`,{...He(Pf.base,a,i&&Pf.reducedMotion),"data-astryx-theme":e.name,...Xi(i),"data-theme":t===`system`?void 0:t,children:n})})})}Bf.displayName=`Theme`;var $={size:`1em`,"aria-hidden":!0},Vf={name:`neutral`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--_astryx-pre-tempo-duration-fast-min":`95ms`,"--_astryx-pre-tempo-duration-fast":`125ms`,"--_astryx-pre-tempo-duration-fast-max":`165ms`,"--_astryx-pre-tempo-duration-medium-min":`225ms`,"--_astryx-pre-tempo-duration-medium":`300ms`,"--_astryx-pre-tempo-duration-medium-max":`400ms`,"--_astryx-pre-tempo-duration-slow-min":`525ms`,"--_astryx-pre-tempo-duration-slow":`700ms`,"--_astryx-pre-tempo-duration-slow-max":`935ms`,"--_astryx-motion-tempo":`1`,"--font-family-body":`Figtree, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,"--font-family-heading":`Figtree, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#525252, #b9b9b9)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#525252, #b9b9b9)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#525252, #b9b9b9)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-syntax-added":`color-mix(in srgb, var(--color-success-muted) 25%, transparent)`,"--color-syntax-removed":`color-mix(in srgb, var(--color-error-muted) 25%, transparent)`,"--color-syntax-diff-header":`transparent`,"--color-syntax-added-marker":`var(--color-syntax-punctuation)`,"--color-syntax-removed-marker":`var(--color-syntax-punctuation)`,"--color-background-surface":`light-dark(#ffffff, #262626)`,"--color-background-body":`light-dark(#f1f1f1, #1b1b1b)`,"--color-background-card":`light-dark(#ffffff, #1b1b1b)`,"--color-background-popover":`light-dark(#ffffff, #1b1b1b)`,"--color-background-muted":`light-dark(#f1f1f1, #1b1b1b)`,"--color-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-accent-muted":`light-dark(#d4d4d4, #474747)`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(#00000080, #000000CC)`,"--color-overlay-hover":`light-dark(#0000000D, #ffffff0D)`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#000000, #ffffff)`,"--color-text-secondary":`light-dark(#474747, #9e9e9e)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #111111)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-info":`light-dark(#0074e2, #6d9cfe)`,"--color-info-muted":`light-dark(#b9d7ff, #88bcff3D)`,"--color-on-info":`light-dark(#ffffff, #111111)`,"--color-text-info":`light-dark(#0067ce, #a1caff)`,"--color-text-success":`light-dark(#2f7d33, #90ca90)`,"--color-text-warning":`light-dark(#836700, #eec448)`,"--color-text-error":`light-dark(#c0262c, #ffaea7)`,"--color-on-info-muted":`light-dark(#003978, #a1caff)`,"--color-on-success-muted":`light-dark(#00490b, #90ca90)`,"--color-on-warning-muted":`light-dark(#4b3900, #eec448)`,"--color-on-error-muted":`light-dark(#76000c, #ffaea7)`,"--color-icon-accent":`light-dark(#1b1b1b, #f1f1f1)`,"--color-icon-primary":`light-dark(#000000, #ffffff)`,"--color-icon-secondary":`light-dark(#6a6a6a, #9e9e9e)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#198100, #64af4c)`,"--color-warning":`#ffce2f`,"--color-error":`light-dark(#c9303a, #ff705d)`,"--color-success-muted":`light-dark(#bce0bb, #90ca903D)`,"--color-error-muted":`light-dark(#ffc4be, #ff98903D)`,"--color-warning-muted":`light-dark(#fae19e, #e2b6233D)`,"--color-border":`light-dark(#00000014, #ffffff1A)`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(#000000, #ffffff)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#0074e2, #6d9cfe)`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`},__localTokenLineage:[`neutral`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},item:{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"tree-list-item":{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"file-input":{"mode:dropzone":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},citation:{"variant:number":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-on-error-muted)`}},badge:{"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`Figtree`,fallbacks:`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`},heading:{family:`Figtree`,fallbacks:`-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:{close:(0,U.jsx)(ie,{...$}),chevronDown:(0,U.jsx)(oi,{...$}),chevronLeft:(0,U.jsx)(ci,{...$}),chevronRight:(0,U.jsx)(ui,{...$}),chevronsLeft:(0,U.jsx)(fi,{...$}),chevronsRight:(0,U.jsx)(mi,{...$}),check:(0,U.jsx)(ii,{...$}),success:(0,U.jsx)(gi,{...$}),error:(0,U.jsx)(vi,{...$}),warning:(0,U.jsx)(ye,{...$}),info:(0,U.jsx)(be,{...$}),calendar:(0,U.jsx)(ei,{...$}),clock:(0,U.jsx)(bi,{...$}),externalLink:(0,U.jsx)(Ei,{...$}),menu:(0,U.jsx)(ve,{...$}),moreHorizontal:(0,U.jsx)(wi,{...$}),search:(0,U.jsx)(ce,{...$}),upload:(0,U.jsx)(Ii,{...$}),arrowUp:(0,U.jsx)(Qr,{...$}),arrowDown:(0,U.jsx)(Jr,{...$}),arrowsUpDown:(0,U.jsx)(Xr,{...$}),funnel:(0,U.jsx)(Ai,{...$}),eyeSlash:(0,U.jsx)(Oi,{...$}),viewColumns:(0,U.jsx)(Si,{...$}),copy:(0,U.jsx)(se,{...$}),checkDouble:(0,U.jsx)(ni,{...$}),wrench:(0,U.jsx)(Ri,{...$}),stop:(0,U.jsx)(Pi,{...$}),microphone:(0,U.jsx)(Mi,{...$})}},Hf={mlxcel:{name:`mlxcel`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--_astryx-pre-tempo-duration-fast-min":`95ms`,"--_astryx-pre-tempo-duration-fast":`125ms`,"--_astryx-pre-tempo-duration-fast-max":`165ms`,"--_astryx-pre-tempo-duration-medium-min":`225ms`,"--_astryx-pre-tempo-duration-medium":`300ms`,"--_astryx-pre-tempo-duration-medium-max":`400ms`,"--_astryx-pre-tempo-duration-slow-min":`525ms`,"--_astryx-pre-tempo-duration-slow":`700ms`,"--_astryx-pre-tempo-duration-slow-max":`935ms`,"--_astryx-motion-tempo":`1`,"--font-family-body":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-heading":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#525252, #b9b9b9)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#525252, #b9b9b9)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#525252, #b9b9b9)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-syntax-added":`color-mix(in srgb, var(--color-success-muted) 25%, transparent)`,"--color-syntax-removed":`color-mix(in srgb, var(--color-error-muted) 25%, transparent)`,"--color-syntax-diff-header":`transparent`,"--color-syntax-added-marker":`var(--color-syntax-punctuation)`,"--color-syntax-removed-marker":`var(--color-syntax-punctuation)`,"--color-background-surface":`light-dark(#ffffff, #181d27)`,"--color-background-body":`light-dark(#f3f4f6, #0e1117)`,"--color-background-card":`light-dark(#ffffff, #181d27)`,"--color-background-popover":`light-dark(#ffffff, #181d27)`,"--color-background-muted":`light-dark(#f8f8fa, #111722)`,"--color-accent":`light-dark(#245bd8, #245bd8)`,"--color-accent-muted":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(rgba(8, 10, 14, 0.42), rgba(8, 10, 14, 0.42))`,"--color-overlay-hover":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#151a24, #f5f7fb)`,"--color-text-secondary":`light-dark(#59647a, #aeb8c9)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#245bd8, #9dbdff)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #ffffff)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-info":`light-dark(#0074e2, #6d9cfe)`,"--color-info-muted":`light-dark(#b9d7ff, #88bcff3D)`,"--color-on-info":`light-dark(#ffffff, #111111)`,"--color-text-info":`light-dark(#0067ce, #a1caff)`,"--color-text-success":`light-dark(#2f7d33, #90ca90)`,"--color-text-warning":`light-dark(#836700, #eec448)`,"--color-text-error":`light-dark(#c0262c, #ffaea7)`,"--color-on-info-muted":`light-dark(#003978, #a1caff)`,"--color-on-success-muted":`light-dark(#00490b, #90ca90)`,"--color-on-warning-muted":`light-dark(#4b3900, #eec448)`,"--color-on-error-muted":`light-dark(#76000c, #ffaea7)`,"--color-icon-accent":`light-dark(#245bd8, #9dbdff)`,"--color-icon-primary":`light-dark(#151a24, #f5f7fb)`,"--color-icon-secondary":`light-dark(#59647a, #aeb8c9)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#137333, #9be6b0)`,"--color-warning":`light-dark(#8a5200, #ffd28a)`,"--color-error":`light-dark(#b42318, #ffb4a8)`,"--color-success-muted":`light-dark(#eaf7ee, rgba(25, 108, 55, 0.22))`,"--color-error-muted":`light-dark(#fff0ed, rgba(152, 43, 28, 0.24))`,"--color-warning-muted":`light-dark(#fff4d8, rgba(139, 92, 0, 0.22))`,"--color-border":`light-dark(rgba(33, 42, 61, 0.12), rgba(255, 255, 255, 0.12))`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(#000000, #ffffff)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`,"--focus-outline-color":`light-dark(#245bd8, #9dbdff)`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#245bd8, #245bd8)`,"--astryx-theme-neutral-color-status-fill-success":`light-dark(#137333, #9be6b0)`,"--astryx-theme-neutral-color-status-fill-error":`light-dark(#b42318, #ffb4a8)`,"--astryx-theme-neutral-color-status-muted-accent":`light-dark(rgba(36, 42, 55, 0.08), rgba(255, 255, 255, 0.10))`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`,"--astryx-theme-neutral-color-status-fill-success":`mlxcel`,"--astryx-theme-neutral-color-status-fill-error":`mlxcel`,"--astryx-theme-neutral-color-status-muted-accent":`mlxcel`},__localTokenLineage:[`neutral`,`mlxcel`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},item:{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"tree-list-item":{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"file-input":{"mode:dropzone":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},citation:{"variant:number":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-on-error-muted)`}},badge:{"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`},heading:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:Vf.icons},glass:{name:`mlxcel-glass`,__built:!0,tokens:{"--font-size-4xs":`0.375rem`,"--font-size-3xs":`0.4375rem`,"--font-size-2xs":`0.5rem`,"--font-size-xs":`0.625rem`,"--font-size-sm":`0.75rem`,"--font-size-base":`0.875rem`,"--font-size-lg":`1.0625rem`,"--font-size-xl":`1.25rem`,"--font-size-2xl":`1.5rem`,"--font-size-3xl":`1.8125rem`,"--font-size-4xl":`2.1875rem`,"--font-size-5xl":`2.625rem`,"--text-heading-1-size":`var(--font-size-2xl)`,"--text-heading-1-weight":`var(--font-weight-semibold)`,"--text-heading-1-leading":`1.3333`,"--text-heading-2-size":`var(--font-size-xl)`,"--text-heading-2-weight":`var(--font-weight-semibold)`,"--text-heading-2-leading":`1.4`,"--text-heading-3-size":`var(--font-size-lg)`,"--text-heading-3-weight":`var(--font-weight-bold)`,"--text-heading-3-leading":`1.4118`,"--text-heading-4-size":`var(--font-size-base)`,"--text-heading-4-weight":`var(--font-weight-bold)`,"--text-heading-4-leading":`1.4286`,"--text-heading-5-size":`var(--font-size-sm)`,"--text-heading-5-weight":`var(--font-weight-semibold)`,"--text-heading-5-leading":`1.6667`,"--text-heading-6-size":`var(--font-size-xs)`,"--text-heading-6-weight":`var(--font-weight-semibold)`,"--text-heading-6-leading":`1.6`,"--text-body-size":`var(--font-size-base)`,"--text-body-weight":`var(--font-weight-normal)`,"--text-body-leading":`1.4286`,"--text-large-size":`var(--font-size-lg)`,"--text-large-weight":`var(--font-weight-semibold)`,"--text-large-leading":`1.4118`,"--text-label-size":`var(--font-size-base)`,"--text-label-weight":`var(--font-weight-medium)`,"--text-label-leading":`1.4286`,"--text-code-size":`var(--font-size-base)`,"--text-code-weight":`var(--font-weight-normal)`,"--text-code-leading":`1.4286`,"--text-supporting-size":`var(--font-size-sm)`,"--text-supporting-weight":`var(--font-weight-normal)`,"--text-supporting-leading":`1.6667`,"--text-display-1-size":`var(--font-size-5xl)`,"--text-display-1-weight":`var(--font-weight-normal)`,"--text-display-1-leading":`1.2381`,"--text-display-2-size":`var(--font-size-4xl)`,"--text-display-2-weight":`var(--font-weight-normal)`,"--text-display-2-leading":`1.2571`,"--text-display-3-size":`var(--font-size-3xl)`,"--text-display-3-weight":`var(--font-weight-normal)`,"--text-display-3-leading":`1.3793`,"--duration-fast-min":`95ms`,"--duration-fast":`125ms`,"--duration-fast-max":`165ms`,"--duration-medium-min":`225ms`,"--duration-medium":`300ms`,"--duration-medium-max":`400ms`,"--duration-slow-min":`525ms`,"--duration-slow":`700ms`,"--duration-slow-max":`935ms`,"--_astryx-pre-tempo-duration-fast-min":`95ms`,"--_astryx-pre-tempo-duration-fast":`125ms`,"--_astryx-pre-tempo-duration-fast-max":`165ms`,"--_astryx-pre-tempo-duration-medium-min":`225ms`,"--_astryx-pre-tempo-duration-medium":`300ms`,"--_astryx-pre-tempo-duration-medium-max":`400ms`,"--_astryx-pre-tempo-duration-slow-min":`525ms`,"--_astryx-pre-tempo-duration-slow":`700ms`,"--_astryx-pre-tempo-duration-slow-max":`935ms`,"--_astryx-motion-tempo":`1`,"--font-family-body":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-heading":`-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,"--font-family-code":`ui-monospace, "SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`,"--color-syntax-keyword":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-string":`light-dark(#0b5615, #a4d6a3)`,"--color-syntax-comment":`light-dark(#525252, #b9b9b9)`,"--color-syntax-number":`light-dark(#733100, #ffb37e)`,"--color-syntax-function":`light-dark(#00458c, #a1caff)`,"--color-syntax-type":`light-dark(#6b187c, #efa8ff)`,"--color-syntax-variable":`light-dark(#111111, #e2e2e2)`,"--color-syntax-operator":`light-dark(#525252, #b9b9b9)`,"--color-syntax-constant":`light-dark(#733100, #ffb37e)`,"--color-syntax-tag":`light-dark(#8a0011, #ffaea7)`,"--color-syntax-attribute":`light-dark(#584400, #eec448)`,"--color-syntax-property":`light-dark(#005348, #90d7c8)`,"--color-syntax-punctuation":`light-dark(#525252, #b9b9b9)`,"--color-syntax-background":`light-dark(#ffffff, #111111)`,"--color-syntax-added":`color-mix(in srgb, var(--color-success-muted) 25%, transparent)`,"--color-syntax-removed":`color-mix(in srgb, var(--color-error-muted) 25%, transparent)`,"--color-syntax-diff-header":`transparent`,"--color-syntax-added-marker":`var(--color-syntax-punctuation)`,"--color-syntax-removed-marker":`var(--color-syntax-punctuation)`,"--color-background-surface":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-body":`light-dark(#e6ebf3, #0b0d12)`,"--color-background-card":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-popover":`light-dark(#fbfbfd, #1c1c1f)`,"--color-background-muted":`light-dark(#f1f2f6, #232327)`,"--color-accent":`light-dark(#0058b9, #0a5bc0)`,"--color-accent-muted":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`,"--color-neutral":`light-dark(#0000000F, #ffffff1A)`,"--color-overlay":`light-dark(rgba(8, 10, 14, 0.32), rgba(8, 10, 14, 0.32))`,"--color-overlay-hover":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`,"--color-overlay-pressed":`light-dark(#0000001A, #ffffff1A)`,"--color-text-primary":`light-dark(#1d1d1f, #f5f5f7)`,"--color-text-secondary":`light-dark(#545458, #b8b8bc)`,"--color-text-disabled":`light-dark(#919191, #525252)`,"--color-text-accent":`light-dark(#0058b9, #6cb2ff)`,"--color-on-dark":`#ffffff`,"--color-on-light":`#111111`,"--color-on-accent":`light-dark(#ffffff, #ffffff)`,"--color-on-success":`light-dark(#ffffff, #111111)`,"--color-on-error":`light-dark(#ffffff, #111111)`,"--color-on-warning":`#111111`,"--color-info":`light-dark(#0074e2, #6d9cfe)`,"--color-info-muted":`light-dark(#b9d7ff, #88bcff3D)`,"--color-on-info":`light-dark(#ffffff, #111111)`,"--color-text-info":`light-dark(#0067ce, #a1caff)`,"--color-text-success":`light-dark(#2f7d33, #90ca90)`,"--color-text-warning":`light-dark(#836700, #eec448)`,"--color-text-error":`light-dark(#c0262c, #ffaea7)`,"--color-on-info-muted":`light-dark(#003978, #a1caff)`,"--color-on-success-muted":`light-dark(#00490b, #90ca90)`,"--color-on-warning-muted":`light-dark(#4b3900, #eec448)`,"--color-on-error-muted":`light-dark(#76000c, #ffaea7)`,"--color-icon-accent":`light-dark(#0058b9, #6cb2ff)`,"--color-icon-primary":`light-dark(#1d1d1f, #f5f5f7)`,"--color-icon-secondary":`light-dark(#545458, #b8b8bc)`,"--color-icon-disabled":`light-dark(#919191, #525252)`,"--color-success":`light-dark(#1b6f3c, #4cd964)`,"--color-warning":`light-dark(#8a4f00, #ffb340)`,"--color-error":`light-dark(#b3261e, #ff7b72)`,"--color-success-muted":`light-dark(rgba(52, 199, 89, 0.12), rgba(50, 215, 75, 0.14))`,"--color-error-muted":`light-dark(rgba(255, 59, 48, 0.10), rgba(255, 69, 58, 0.16))`,"--color-warning-muted":`light-dark(rgba(255, 149, 0, 0.12), rgba(255, 159, 10, 0.14))`,"--color-border":`light-dark(rgba(60, 60, 67, 0.14), rgba(255, 255, 255, 0.12))`,"--color-border-emphasized":`light-dark(#d4d4d4, #525252)`,"--color-skeleton":`light-dark(#f1f1f1, #525252)`,"--color-shadow":`light-dark(#0000001A, #0000004D)`,"--color-tint-hover":`light-dark(#000000, #ffffff)`,"--color-background-red":`light-dark(#ffc4be, #5b2b28)`,"--color-border-red":`light-dark(#ffaea7, #fa6762)`,"--color-icon-red":`light-dark(#76000c, #ff9890)`,"--color-text-red":`light-dark(#76000c, #ffaea7)`,"--color-background-orange":`light-dark(#ffc7a1, #503424)`,"--color-border-orange":`light-dark(#ffc7a1, #df843f)`,"--color-icon-orange":`light-dark(#622700, #ff9e55)`,"--color-text-orange":`light-dark(#622700, #ffb37e)`,"--color-background-yellow":`light-dark(#fae19e, #453a1c)`,"--color-border-yellow":`light-dark(#e3c36c, #c29900)`,"--color-icon-yellow":`light-dark(#4b3900, #e2b623)`,"--color-text-yellow":`light-dark(#4b3900, #eec448)`,"--color-background-green":`light-dark(#bce0bb, #2b422b)`,"--color-border-green":`light-dark(#aad4a9, #6ab26b)`,"--color-icon-green":`light-dark(#00490b, #90ca90)`,"--color-text-green":`light-dark(#00490b, #90ca90)`,"--color-background-teal":`light-dark(#a9e2d6, #28413c)`,"--color-border-teal":`light-dark(#90d7c8, #4fb1a0)`,"--color-icon-teal":`light-dark(#00463d, #81c9bb)`,"--color-text-teal":`light-dark(#00463d, #90d7c8)`,"--color-background-cyan":`light-dark(#9ae2f4, #274046)`,"--color-border-cyan":`light-dark(#85d5e9, #49adc4)`,"--color-icon-cyan":`light-dark(#004351, #71c7dd)`,"--color-text-cyan":`light-dark(#004351, #85d5e9)`,"--color-background-blue":`light-dark(#b9d7ff, #253c5a)`,"--color-border-blue":`light-dark(#a1caff, #5aa0f8)`,"--color-icon-blue":`light-dark(#003978, #88bcff)`,"--color-text-blue":`light-dark(#003978, #a1caff)`,"--color-background-purple":`light-dark(#f7d5ff, #4a2f51)`,"--color-border-purple":`light-dark(#f3bfff, #d885eb)`,"--color-icon-purple":`light-dark(#5c0e6c, #e496f6)`,"--color-text-purple":`light-dark(#5c0e6c, #eaacf8)`,"--color-background-pink":`light-dark(#ffc0d7, #572b3d)`,"--color-border-pink":`light-dark(#ffc0d7, #fc78b1)`,"--color-icon-pink":`light-dark(#70003f, #fd92bd)`,"--color-text-pink":`light-dark(#70003f, #ffa9ca)`,"--color-background-gray":`light-dark(#e2e2e2, #303030)`,"--color-border-gray":`light-dark(#d4d4d4, #262626)`,"--color-icon-gray":`light-dark(#474747, #9e9e9e)`,"--color-text-gray":`light-dark(#1b1b1b, #d4d4d4)`,"--radius-none":`0px`,"--radius-inner":`0.375rem`,"--radius-element":`0.625rem`,"--radius-container":`0.75rem`,"--radius-page":`1.75rem`,"--radius-full":`9999px`,"--shadow-low":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 25%)), 0 4px 8px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 40%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 8%))`,"--shadow-med":`0 2px 4px light-dark(oklch(0 0 0 / 5%), oklch(0 0 0 / 35%)), 0 4px 12px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 12%))`,"--shadow-high":`0 4px 6px light-dark(oklch(0 0 0 / 10%), oklch(0 0 0 / 50%)), 0 12px 24px light-dark(oklch(0 0 0 / 15%), oklch(0 0 0 / 70%)), inset 0 0 0 1px light-dark(transparent, oklch(1 0 0 / 15%))`,"--shadow-inset-hover":`inset 0px 0px 0px 2px #0074e24D`,"--shadow-inset-selected":`inset 0px 0px 0px 2px #0074e280`,"--shadow-inset-success":`inset 0px 0px 0px 2px #2f7d334D`,"--shadow-inset-warning":`inset 0px 0px 0px 2px #f8d36a4D`,"--shadow-inset-error":`inset 0px 0px 0px 2px #de47454D`,"--focus-outline-color":`light-dark(#0058b9, #6cb2ff)`},localTokens:{"--astryx-theme-neutral-color-status-fill-accent":`light-dark(#0058b9, #0a5bc0)`,"--astryx-theme-neutral-color-status-fill-success":`light-dark(#1b6f3c, #4cd964)`,"--astryx-theme-neutral-color-status-fill-error":`light-dark(#b3261e, #ff7b72)`,"--astryx-theme-neutral-color-status-muted-accent":`light-dark(rgba(0, 88, 185, 0.10), rgba(64, 156, 255, 0.18))`},__localTokenOwners:{"--astryx-theme-neutral-color-status-fill-accent":`neutral`,"--astryx-theme-neutral-color-status-fill-success":`mlxcel-glass`,"--astryx-theme-neutral-color-status-fill-error":`mlxcel-glass`,"--astryx-theme-neutral-color-status-muted-accent":`mlxcel-glass`},__localTokenLineage:[`neutral`,`mlxcel-glass`],components:{heading:{"level:1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-1-size)`,fontWeight:`var(--text-heading-1-weight)`,lineHeight:`var(--text-heading-1-leading)`},"level:2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-2-size)`,fontWeight:`var(--text-heading-2-weight)`,lineHeight:`var(--text-heading-2-leading)`},"level:3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-3-size)`,fontWeight:`var(--text-heading-3-weight)`,lineHeight:`var(--text-heading-3-leading)`},"level:4":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-4-size)`,fontWeight:`var(--text-heading-4-weight)`,lineHeight:`var(--text-heading-4-leading)`},"level:5":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-5-size)`,fontWeight:`var(--text-heading-5-weight)`,lineHeight:`var(--text-heading-5-leading)`},"level:6":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-heading-6-size)`,fontWeight:`var(--text-heading-6-weight)`,lineHeight:`var(--text-heading-6-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},text:{"type:body":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-body-size)`,lineHeight:`var(--text-body-leading)`},"type:large":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-large-size)`,lineHeight:`var(--text-large-leading)`},"type:label":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-label-size)`,lineHeight:`var(--text-label-leading)`},"type:code":{fontFamily:`var(--font-family-code)`,fontSize:`var(--text-code-size)`,lineHeight:`var(--text-code-leading)`},"type:supporting":{fontFamily:`var(--font-family-body)`,fontSize:`var(--text-supporting-size)`,lineHeight:`var(--text-supporting-leading)`},"type:display-1":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-1-size)`,lineHeight:`var(--text-display-1-leading)`},"type:display-2":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-2-size)`,lineHeight:`var(--text-display-2-leading)`},"type:display-3":{fontFamily:`var(--font-family-heading)`,fontSize:`var(--text-display-3-size)`,lineHeight:`var(--text-display-3-leading)`}},item:{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"tree-list-item":{selected:{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},"file-input":{"mode:dropzone":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},citation:{"variant:number":{"--color-text-secondary":`light-dark(#474747, #c6c6c6)`}},button:{"variant:destructive":{backgroundColor:`var(--color-error-muted)`,color:`var(--color-on-error-muted)`}},badge:{"variant:neutral":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`},"variant:red":{backgroundColor:`var(--color-background-red)`,color:`var(--color-text-red)`},"variant:orange":{backgroundColor:`var(--color-background-orange)`,color:`var(--color-text-orange)`},"variant:yellow":{backgroundColor:`var(--color-background-yellow)`,color:`var(--color-text-yellow)`},"variant:green":{backgroundColor:`var(--color-background-green)`,color:`var(--color-text-green)`},"variant:teal":{backgroundColor:`var(--color-background-teal)`,color:`var(--color-text-teal)`},"variant:cyan":{backgroundColor:`var(--color-background-cyan)`,color:`var(--color-text-cyan)`},"variant:blue":{backgroundColor:`var(--color-background-blue)`,color:`var(--color-text-blue)`},"variant:purple":{backgroundColor:`var(--color-background-purple)`,color:`var(--color-text-purple)`},"variant:pink":{backgroundColor:`var(--color-background-pink)`,color:`var(--color-text-pink)`},"variant:gray":{backgroundColor:`var(--color-background-gray)`,color:`var(--color-text-gray)`}},"segmented-control":{base:{padding:`var(--spacing-1)`}},"segmented-control-item":{"size:sm":{height:`calc(var(--size-element-sm) - 8px)`},"size:md":{height:`calc(var(--size-element-md) - 8px)`},"size:lg":{height:`calc(var(--size-element-lg) - 8px)`},selected:{boxShadow:`none`}},switch:{base:{"--color-background-gray":`var(--color-border-emphasized)`}},"progress-bar":{base:{"--color-background-muted":`var(--color-border-emphasized)`}},card:{base:{padding:`var(--spacing-3)`}},section:{base:{padding:`var(--spacing-3)`}}},__onDark:{tokens:{"color-scheme":`dark`,"--color-text-primary":`var(--color-on-dark)`,"--color-icon-primary":`var(--color-on-dark)`,"--color-accent":`var(--color-on-dark)`}},__onLight:{tokens:{"color-scheme":`light`,"--color-text-primary":`var(--color-on-light)`,"--color-icon-primary":`var(--color-on-light)`,"--color-accent":`var(--color-on-light)`}},__adaptations:{widthBreakpoints:{sm:640,md:768,lg:1024,xl:1280,"2xl":1536},rules:[{when:{pointer:`coarse`,width:{below:`lg`}},value:{typography:{scale:{base:16,ratio:1.1736}}}}]},__axes:{typography:{scale:{base:14,ratio:1.2},body:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`},heading:{family:`-apple-system`,fallbacks:`BlinkMacSystemFont, "SF Pro Text", "Segoe UI", system-ui, sans-serif`,weights:{3:`bold`,4:`bold`}},code:{family:`ui-monospace`,fallbacks:`"SF Mono", Monaco, Consolas, "Liberation Mono", "Courier New", monospace`}},motion:{fast:125,medium:300,slow:700,ratio:.75}},icons:Vf.icons}};function Uf(e){return(0,U.jsx)(Bf,{theme:Hf[e.family],mode:e.scheme,children:(0,U.jsx)(Le,{locale:e.locale,children:e.children})})}var Wf=0,Gf=0,Kf=new Set;function qf(e){return Kf.add(e),()=>Kf.delete(e)}function Jf(){Gf+=1,Wf=Gf;for(let e of Kf)e()}function Yf(){return(0,z.useSyncExternalStore)(qf,()=>Wf)}function Xf(){if(Wf===0)return!1;Wf=0;for(let e of Kf)e();return!0}var Zf=`(min-width: 1100px)`,Qf=null;function $f(){return typeof window.matchMedia==`function`?(Qf?.matchMedia!==window.matchMedia&&(Qf={matchMedia:window.matchMedia,list:window.matchMedia(Zf)}),Qf.list):null}function ep(e){let t=$f();return t?(t.addEventListener(`change`,e),()=>t.removeEventListener(`change`,e)):()=>void 0}function tp(){return(0,z.useSyncExternalStore)(ep,()=>$f()?.matches??!0)}function np(e,t,n){let r=e.filter(e=>e.runtime.server_instance_id===t.server_instance_id&&e.runtime.model_id===t.model_id&&e.receivedAt>n-3e5&&e.receivedAt<=n),i=r.at(-1);return i!==void 0&&Math.floor(i.receivedAt/2e3)===Math.floor(n/2e3)&&r.pop(),[...r,{receivedAt:n,runtime:t}].slice(-150)}function rp(e,t){let n=[...e.values()].sort((e,t)=>t.updated_at.localeCompare(e.updated_at)),r=n.filter(e=>!ip(e)),i=n.filter(e=>ip(e)&&Date.parse(e.updated_at)>t-36e5).slice(0,200);return new Map([...r,...i].map(e=>[e.operation_id,e]))}function ip(e){return[`succeeded`,`failed`,`cancelled`].includes(e.state)}var ap=`webui.ui-api.v1`;function op(){return{schemaVersion:ap,auth:{status:`signed-out`,tokenPresent:!1},connection:`idle`,bootstrap:null,catalog:[],catalogSequence:null,operations:new Map,runtimes:new Map,runtimeHistory:[],selectedModelId:null,serverInstanceId:null,lastEventId:null,lastSequence:null,lastUpdatedAt:null,lastSuccessfulAt:null,error:null,pendingReconciliations:new Map,resourceFences:{catalog:null,operationsSnapshot:null,operations:new Map,models:new Map,runtimes:new Map}}}function sp(e,t){return t.type===`login-start`?{...e,auth:{status:`authenticating`,tokenPresent:!0},connection:`bootstrapping`,error:null}:t.type===`login-success`?cp(e,t.bootstrap,t.now):t.type===`logout`?{...op(),lastUpdatedAt:t.now}:t.type===`select-model`?{...e,selectedModelId:t.modelId,runtimeHistory:[]}:t.type===`catalog`?lp(e,t.response,t.now):t.type===`operations-snapshot`?up(e,t.response,t.now):t.type===`operation`?dp(e,t.operation,t.sequence,t.now):t.type===`runtime`?fp(e,t.runtime,t.sequence,t.now):t.type===`event`?pp(e,t.event,t.now):t.type===`pending`?{...e,pendingReconciliations:xp(e.pendingReconciliations,t.item.idempotencyKey,t.item)}:t.type===`reconciled`?{...e,pendingReconciliations:Sp(e.pendingReconciliations,t.idempotencyKey)}:{...e,connection:t.connection,error:t.error??null,lastUpdatedAt:t.now}}function cp(e,t,n){let r=t.server.server_instance_id;return e.serverInstanceId!==null&&e.serverInstanceId!==r?{...op(),auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n}:{...e,auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n,error:null}}function lp(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return _p(e,t.server_instance_id,n);if(e.catalogSequence!==null&&t.snapshot_sequence<e.catalogSequence)return e;let r=e.catalogSequence===t.snapshot_sequence;return vp({...e,connection:e.connection===`bootstrapping`?`ready`:e.connection,catalog:yp(r?e.catalog:[],t),catalogSequence:t.snapshot_sequence,serverInstanceId:t.server_instance_id,lastSequence:bp({...e.resourceFences,catalog:t.snapshot_sequence}),error:null,resourceFences:{...e.resourceFences,catalog:t.snapshot_sequence}},n)}function up(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return _p(e,t.server_instance_id,n);if(e.resourceFences.operationsSnapshot!==null&&t.snapshot_sequence<e.resourceFences.operationsSnapshot)return e;let r=e.resourceFences.operationsSnapshot===t.snapshot_sequence,i=r?new Map(e.operations):new Map,a=r?new Map(e.resourceFences.operations):new Map;for(let e of t.items)i.set(e.operation_id,e),a.set(e.operation_id,t.snapshot_sequence);let o=rp(i,n),s={...e.resourceFences,operationsSnapshot:t.snapshot_sequence,operations:new Map([...a].filter(([e])=>o.has(e)))};return vp({...e,operations:o,serverInstanceId:t.server_instance_id,lastSequence:bp(s),error:null,resourceFences:s},n)}function dp(e,t,n,r){if(n!==null&&e.resourceFences.operationsSnapshot!==null&&n<=e.resourceFences.operationsSnapshot)return e;let i=e.resourceFences.operations.get(t.operation_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=rp(xp(e.operations,t.operation_id,t),r),o=n===null?e.resourceFences.operations:xp(e.resourceFences.operations,t.operation_id,n),s={...e.resourceFences,operations:new Map([...o].filter(([e])=>a.has(e)))};return vp({...e,operations:a,lastSequence:bp(s),error:null,resourceFences:s},r)}function fp(e,t,n,r){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return _p(e,t.server_instance_id,r);let i=e.resourceFences.runtimes.get(t.model_id);if(n!==null&&i!==void 0&&n<i)return e;let a=n===null?e.resourceFences.runtimes:xp(e.resourceFences.runtimes,t.model_id,n),o={...e.resourceFences,runtimes:a};return vp({...e,runtimes:xp(e.runtimes,t.model_id,t),runtimeHistory:t.model_id===e.selectedModelId?np(e.runtimeHistory,t,r):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastSequence:bp(o),error:null,resourceFences:o},r)}function pp(e,t,n){if(t.schema_version!==`webui.ui-api.v1`)return{...e,connection:`schema-mismatch`,error:{code:`schema_mismatch`,message:`Unsupported WebUI schema ${t.schema_version}`,retryable:!1},lastUpdatedAt:n};if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return _p(e,t.server_instance_id,n);if(t.type===`heartbeat`)return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:e.connection===`polling`?`polling`:`streaming`,lastUpdatedAt:n};if(t.type===`server_restart`||t.type===`gap`||t.type===`reset`)return gp(e,t,n);if(t.type===`operation`)return{...dp(e,t.payload.operation,t.sequence,n),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`};if(t.type===`model_revision`)return mp(e,t,n);if(t.type===`runtime`)return hp(e,t,n);if(t.type===`snapshot`){let r={...e.resourceFences,catalog:t.payload.catalog_changed?t.sequence:e.resourceFences.catalog};return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:bp(r),connection:`streaming`,lastUpdatedAt:n,resourceFences:r}}return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`,lastUpdatedAt:n}}function mp(e,t,n){let r=t.payload.model_id,i=e.resourceFences.models.get(r);if(i!==void 0&&t.sequence<=i)return e;let a=e.catalog.map(e=>e.identity.id===r&&t.payload.revision>=e.identity.revision?{...e,identity:{...e.identity,revision:t.payload.revision},lifecycle:t.payload.lifecycle}:e),o={...e.resourceFences,models:xp(e.resourceFences.models,r,t.sequence)};return vp({...e,catalog:a,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:bp(o),connection:`streaming`,resourceFences:o},n)}function hp(e,t,n){let r=t.payload.runtime,i=e.resourceFences.runtimes.get(r.model_id);if(i!==void 0&&t.sequence<=i)return e;let a={...e.resourceFences,runtimes:xp(e.resourceFences.runtimes,r.model_id,t.sequence)};return vp({...e,runtimes:xp(e.runtimes,r.model_id,r),runtimeHistory:r.model_id===e.selectedModelId?np(e.runtimeHistory,r,n):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:bp(a),connection:`streaming`,resourceFences:a},n)}function gp(e,t,n){let r=t.type===`server_restart`?null:e.lastSuccessfulAt;return{...op(),auth:e.auth,connection:`stale`,serverInstanceId:t.server_instance_id,selectedModelId:e.selectedModelId,lastEventId:t.event_id,error:{code:t.type,message:t.payload.reason,retryable:t.payload.resnapshot},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n,lastSuccessfulAt:r}}function _p(e,t,n){return{...op(),auth:e.auth,connection:`stale`,serverInstanceId:t,selectedModelId:e.selectedModelId,error:{code:`server_restarted`,message:`The mlxcel server restarted; refresh the authoritative snapshot before continuing.`,retryable:!0},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n}}function vp(e,t){return{...e,lastUpdatedAt:t,lastSuccessfulAt:t}}function yp(e,t){if(e.length>0){let n=new Map(e.map(e=>[e.identity.id,e]));for(let e of t.items)n.set(e.identity.id,e);return Array.from(n.values()).sort((e,t)=>e.identity.display_name.localeCompare(t.identity.display_name))}return t.items}function bp(e){let t=[e.catalog,e.operationsSnapshot,...e.operations.values(),...e.models.values(),...e.runtimes.values()].filter(e=>e!==null);return t.length===0?null:Math.min(...t)}function xp(e,t,n){let r=new Map(e);return r.set(t,n),r}function Sp(e,t){let n=new Map(e);return n.delete(t),n}var Cp=2e3,wp=1e4,Tp=3e4,Ep=6e4,Dp=6e4,Op=new Set([`snapshot`,`model_revision`]),kp=class{client;dispatch;getSnapshot;clock;visibility;random;timer=null;observationTimer=null;inflight=null;eventAbort=null;stopped=!0;failures=0;generation=0;needsFullSnapshot=!0;lastFullSnapshotAt=null;unsubscribe;constructor(e){this.client=e.client,this.dispatch=e.dispatch,this.getSnapshot=e.getSnapshot,this.clock=e.clock??Fp,this.visibility=e.visibility??Ip,this.random=e.random??Math.random,this.unsubscribe=this.visibility.subscribe(()=>this.visibilityChanged())}start(){this.stopped&&(this.stopped=!1,this.generation+=1,this.needsFullSnapshot=!0,this.reschedule(0))}stop(){this.generation+=1,this.stopped=!0,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null}selectionChanged(){this.cancelObservation(),this.reschedule(0)}fullSnapshotDue(e){let t=this.getSnapshot();return t.bootstrap===null||t.catalogSequence===null||this.needsFullSnapshot||this.lastFullSnapshotAt===null||this.eventAbort===null?!0:e-this.lastFullSnapshotAt>=Dp}cancelObservation(){this.generation+=1,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null}visibilityChanged(){this.cancelObservation(),this.needsFullSnapshot=!0,this.stopped||(this.dispatch({type:`connection`,connection:`stale`,now:this.clock.now()}),this.reschedule(0))}dispose(){this.stop(),this.unsubscribe()}async resnapshot(){this.needsFullSnapshot=!0,await this.refresh()}async refresh(){if(this.inflight!==null)return;let e=new AbortController;this.inflight=e;let t=this.generation,n=this.clock.now(),r=this.clock.setTimeout(()=>{e.abort(),this.generation===t&&this.dispatch({type:`connection`,connection:`stale`,error:{code:`observation_timeout`,message:`Runtime observation timed out; retrying.`,retryable:!0},now:this.clock.now()})},wp);this.observationTimer=r;try{if(this.getSnapshot().auth.status===`signed-out`)return;let r=this.fullSnapshotDue(n),i=null,a=[];if(r){if(i=await this.client.bootstrap(e.signal),!this.isCurrent(e,t)||this.getSnapshot().auth.status===`signed-out`||(this.dispatch({type:`login-success`,bootstrap:i,now:n}),a=await this.catalogSnapshot(e.signal),!this.isCurrent(e,t)))return;for(let e of a)this.dispatch({type:`catalog`,response:e,now:n})}let o=await this.operationSnapshot(e.signal);if(!this.isCurrent(e,t))return;let s=a.at(-1),c=o[0];if(s!==void 0&&c!==void 0&&c.server_instance_id!==s.server_instance_id){this.dispatch({type:`connection`,connection:`stale`,error:{code:`snapshot_mismatch`,message:`Operation and catalog snapshots came from different server instances.`,retryable:!0},now:n});return}let l=o.flatMap(e=>(this.dispatch({type:`operations-snapshot`,response:e,now:n}),e.items)),u=this.getSnapshot().selectedModelId,d=r&&i!==null&&u!==null&&a.every(e=>e.server_instance_id===i.server.server_instance_id)&&!a.some(e=>e.items.some(e=>e.identity.id===u));if(d&&this.dispatch({type:`select-model`,modelId:null}),d||await this.refreshSelectedRuntime(e.signal,t),r&&(this.needsFullSnapshot=!1,this.lastFullSnapshotAt=n),!this.isCurrent(e,t))return;let f=this.reconcilePending(l,n);this.failures=0,f||this.dispatch({type:`connection`,connection:`ready`,now:n}),this.eventAbort===null&&!this.visibility.hidden()&&this.startEvents()}catch(n){!e.signal.aborted&&this.generation===t&&(this.failures+=1,this.dispatch({type:`connection`,connection:jp(n),error:Mp(n),now:this.clock.now()}))}finally{this.clock.clearTimeout(r),this.observationTimer===r&&(this.observationTimer=null),this.inflight===e&&(this.inflight=null),!this.stopped&&this.generation===t&&this.reschedule(this.pollDelay())}}noteUnknownPost(e){this.dispatch({type:`pending`,item:e}),this.reschedule(0)}startEvents(){this.eventAbort?.abort();let e=new AbortController;this.eventAbort=e;let t=this.generation,n=Ap(this.getSnapshot());this.client.events({onEvent:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.handleEvent(n)},onRetryAfter:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.reschedule(n)}},e.signal,n).then(()=>{this.generation===t&&(this.eventAbort===e&&(this.eventAbort=null),!e.signal.aborted&&!this.stopped&&this.reschedule(this.backoffDelay()))}).catch(n=>{e.signal.aborted||this.stopped||this.generation!==t||(this.failures+=1,this.eventAbort=null,this.dispatch({type:`connection`,connection:jp(n),error:Mp(n),now:this.clock.now()}),this.reschedule(this.backoffDelay()))})}handleEvent(e){this.dispatch({type:`event`,event:e,now:this.clock.now()}),Op.has(e.type)&&(this.needsFullSnapshot=!0),(e.type===`server_restart`||e.type===`gap`||e.type===`reset`)&&e.payload.resnapshot&&(this.needsFullSnapshot=!0,this.eventAbort?.abort(),this.eventAbort=null,this.refresh())}async catalogSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.catalog(n===void 0?{}:{cursor:n},e);if(r===null?r=i:Pp(`catalog`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operationSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null?r=i:Pp(`operations`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async refreshSelectedRuntime(e,t){let n=this.getSnapshot().selectedModelId;if(n===null)return;let r;try{r=await this.client.runtime(n,e)}catch(e){throw typeof e==`object`&&e&&`status`in e&&e.status===404?new Np(`Selected model changed during observation; refreshing the catalog.`):e}e.aborted||this.generation!==t||this.getSnapshot().selectedModelId===n&&this.dispatch({type:`runtime`,runtime:r,sequence:r.snapshot_sequence,now:this.clock.now()})}reconcilePending(e,t){let n=!1,r=new Set(e.map(e=>e.operation_id));for(let e of this.getSnapshot().pendingReconciliations.values())e.operationId!==null&&r.has(e.operationId)?this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}):t-e.createdAt>=Ep&&(this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}),this.dispatch({type:`connection`,connection:`stale`,error:{code:`unknown_post_unresolved`,message:`A previous control request could not be matched to an operation after reconciliation.`,retryable:!0},now:t}),n=!0);return n}reschedule(e){this.stopped||this.visibility.hidden()||(this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=this.clock.setTimeout(()=>{this.timer=null,this.refresh()},e))}pollDelay(){return Cp}backoffDelay(){let e=Math.min(Tp,500*2**Math.min(6,this.failures));return Math.round(e/2+this.random()*(e/2))}isCurrent(e,t){return this.inflight===e&&this.generation===t&&!e.signal.aborted}};function Ap(e){let t=bp(e.resourceFences);if(e.serverInstanceId!==null&&t!==null)return{lastEventId:null,serverInstanceId:e.serverInstanceId,afterSequence:t};if(e.lastEventId!==null)return{lastEventId:e.lastEventId,serverInstanceId:null,afterSequence:null}}function jp(e){if(e instanceof Np)return`stale`;let t=typeof e==`object`&&e&&`status`in e&&typeof e.status==`number`?e.status:null;return t===401||e instanceof Error&&/401|unauthorized/i.test(e.message)?`unauthorized`:t===403||e instanceof Error&&/403|forbidden/i.test(e.message)?`forbidden`:e instanceof DOMException&&e.name===`AbortError`?`stale`:`offline`}function Mp(e){return e instanceof Np?{code:e.code,message:e.message,retryable:!0}:{code:`sync_error`,message:e instanceof Error?e.message.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`):`Unknown WebUI client error`,retryable:!0}}var Np=class extends Error{code=`snapshot_mismatch`;constructor(e){super(e),this.name=`SnapshotConsistencyError`}};function Pp(e,t,n){if(n.server_instance_id!==t.server_instance_id)throw new Np(`${e} pagination crossed a server restart; refresh the authoritative snapshot.`);if(n.snapshot_sequence!==t.snapshot_sequence)throw new Np(`${e} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`)}var Fp={setTimeout:(e,t)=>globalThis.setTimeout(e,t),clearTimeout:e=>globalThis.clearTimeout(e),now:()=>Date.now()},Ip={hidden:()=>typeof document<`u`&&document.hidden,subscribe:e=>typeof document>`u`?()=>void 0:(document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e))},Lp=z.createContext(null),Rp=z.createContext(null);function zp({children:e,apiBase:t,fetchImpl:n}){let[r,i]=z.useReducer(sp,void 0,op),a=z.useRef(r);a.current=r;let o=z.useRef(0),s=z.useRef(null),c=z.useMemo(()=>new Do({apiBase:t,fetchImpl:n,onUnauthorized:()=>{o.current+=1,s.current?.stop(),i({type:`logout`,now:Date.now()})}}),[t,n]);z.useEffect(()=>{let e=new kp({client:c,dispatch:i,getSnapshot:()=>a.current});return s.current=e,()=>{e.dispose(),s.current=null,c.abortAll()}},[c]);let l=z.useMemo(()=>({getTokenCount:(e,t,n)=>c.tokenCount(d(e),t,n),getSettings:(e,t)=>c.settings(d(e),t),patchSettings:(e,t,n)=>c.patchSettings(d(e),t,n),getModelProps:(e,t)=>c.modelProps(d(e),t),login:async e=>{o.current+=1;let t=o.current;c.abortAll(),c.setBearerToken(e),i({type:`login-start`});try{let e=await c.bootstrap();if(o.current!==t)return;i({type:`login-success`,bootstrap:e,now:Date.now()}),s.current?.start()}catch(e){throw o.current===t&&(c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})),e}},logout:()=>{o.current+=1,s.current?.stop(),c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})},refresh:async()=>{await s.current?.resnapshot()},selectModel:e=>{i({type:`select-model`,modelId:e}),s.current?.selectionChanged()},loadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},unloadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},downloadModel:async e=>{await u(`download`,e.idempotency_key,void 0,()=>c.download(e))},removeModel:async e=>{await u(`removal`,e.idempotency_key,e.model_id,()=>c.removeModel(e))},refreshRuntime:async e=>{let t=o.current,n=await c.runtime(e);return o.current===t&&i({type:`runtime`,runtime:n,sequence:n.snapshot_sequence,now:Date.now()}),n},refreshCatalog:async e=>{await u(`catalog-refresh`,e,void 0,()=>c.refreshCatalog(e))},cancelOperation:async e=>{let t=o.current;await c.cancelOperation(e),o.current===t&&await s.current?.resnapshot()},streamChatCompletions:async(e,t,n,r)=>{await c.chatCompletions(d(e),t,n,r)},streamResponses:async(e,t,n,r)=>{await c.responses(d(e),t,n,r)}}),[c]);async function u(e,t,n,r){let i=o.current;try{let a=await r();if(o.current!==i)return;s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:a.operation_id,modelId:n,createdAt:Date.now()})}catch(r){throw o.current===i&&!(r instanceof To)&&s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:null,modelId:n,createdAt:Date.now()}),r}}function d(e){let t=a.current.catalog.find(t=>t.identity.id===e);if(t===void 0)throw Error(`Selected model is not present in the catalog snapshot.`);if(t.identity.inference_id.length===0)throw Error(`Selected model does not expose an inference model id.`);return t.identity.inference_id}return(0,U.jsx)(Rp.Provider,{value:l,children:(0,U.jsx)(Lp.Provider,{value:r,children:e})})}function Bp(){let e=z.useContext(Lp);if(e===null)throw Error(`useWebUi must be used within WebUiProvider.`);return e}function Vp(){let e=z.useContext(Rp);if(e===null)throw Error(`useWebUiActions must be used within WebUiProvider.`);return e}var Hp=Object.freeze({conversations:50,turnsPerConversation:200,totalTurns:1e3,textCharacters:262144,jsonBytes:16777216,imageBytes:8388608,imagesPerTurn:8,toolsPerTurn:128}),Up=`mlxcel.webui.chat.v1`,Wp=`history`,Gp=`conversations`,Kp=1,qp=2,Jp=[`id`,`modelId`,`modelRevision`,`inferenceId`,`modelName`,`prompt`,`content`,`reasoning`,`tools`,`status`,`finishReason`,`usage`,`ttftMs`,`elapsedMs`,`error`,`parameters`,`images`],Yp=[...Jp,`startedAt`],Xp=new Set([`temperature`,`top_p`,`top_k`,`min_p`,`max_tokens`,`seed`,`repetition_penalty`,`presence_penalty`,`frequency_penalty`]),Zp=class extends Error{constructor(e=`Conversation history is invalid or exceeds the local history limits.`){super(e),this.name=`HistoryValidationError`}},Qp=class extends Error{constructor(e,t){super(e,t),this.name=`HistoryStorageError`}};function $p(){throw new Zp}function em(e,t){if(!e||typeof e!=`object`||Array.isArray(e))return $p();let n=Object.getPrototypeOf(e);if(n!==Object.prototype&&n!==null)return $p();let r=Object.keys(e);return r.length!==t.length||r.some(e=>!t.includes(e))?$p():e}function tm(e,t=Hp.textCharacters){return typeof e!=`string`||e.length>t?$p():e}function nm(e){let t=tm(e,512);return!t||Array.from(t).some(e=>e.charCodeAt(0)<32||e.charCodeAt(0)===127)?$p():t}function rm(e,t=!1){return typeof e!=`number`||!Number.isFinite(e)||e<0||t&&!Number.isSafeInteger(e)?$p():e}var im=864e13;function am(e){let t=rm(e,!0);return t>im?$p():t}function om(e,t){return e===null?null:t(e)}function sm(e,t){return!Array.isArray(e)||e.length>t?$p():e}function cm(e){return new TextEncoder().encode(e).byteLength}function lm(e,t,n){let r=0,i=0,a=0,o=new Set;return sm(e,Hp.conversations).map(e=>{let s=em(e,[`id`,`title`,`systemPrompt`,`turns`,`updatedAt`]),c=nm(s.id);if(o.has(c))return $p();o.add(c);let l=new Set,u=sm(s.turns,Hp.turnsPerConversation).map(e=>{if(++r>Hp.totalTurns)return $p();let o=em(e,n===1?Jp:Yp),s=nm(o.id);if(l.has(s))return $p();l.add(s);let c=tm(o.status,32);if(![`streaming`,`complete`,`cancelled`,`interrupted`,`error`].includes(c))return $p();let u={};if(!o.parameters||typeof o.parameters!=`object`||Array.isArray(o.parameters)||![Object.prototype,null].includes(Object.getPrototypeOf(o.parameters)))return $p();for(let[e,t]of Object.entries(o.parameters)){if(!Xp.has(e)||typeof t!=`number`||!Number.isFinite(t))return $p();u[e]=t}let d=new Set,f=sm(o.tools,Hp.toolsPerTurn).map(e=>{let t=em(e,[`index`,`id`,`name`,`arguments`]),n=rm(t.index,!0);return d.has(n)?$p():(d.add(n),{index:n,id:tm(t.id,512),name:tm(t.name,512),arguments:tm(t.arguments)})}),p=sm(o.images,Hp.imagesPerTurn).map(e=>{let t=em(e,[`name`,`type`,`dataUrl`]),n=tm(t.type,64);if(![`image/png`,`image/jpeg`,`image/webp`].includes(n))return $p();let r=tm(t.dataUrl,Hp.imageBytes*2),a=`data:${n};base64,`;if(!r.startsWith(a))return $p();let o=r.slice(a.length);return!o||o.length%4!=0||!/^[A-Za-z0-9+/]*={0,2}$/u.test(o)||(i+=o.length*3/4-(o.endsWith(`==`)?2:+!!o.endsWith(`=`)),i>Hp.imageBytes)?$p():{name:tm(t.name,512),type:n,dataUrl:r}}),m={id:s,modelId:nm(o.modelId),modelRevision:rm(o.modelRevision,!0),inferenceId:nm(o.inferenceId),modelName:tm(o.modelName,512),prompt:tm(o.prompt),content:tm(o.content),reasoning:tm(o.reasoning),tools:f,status:c===`streaming`?`interrupted`:c,finishReason:om(o.finishReason,tm),usage:om(o.usage,e=>{let t=em(e,[`prompt_tokens`,`completion_tokens`,`total_tokens`]);return{prompt_tokens:rm(t.prompt_tokens,!0),completion_tokens:rm(t.completion_tokens,!0),total_tokens:rm(t.total_tokens,!0)}}),ttftMs:om(o.ttftMs,rm),elapsedMs:om(o.elapsedMs,rm),error:om(o.error,tm),parameters:u,images:t.includeImages?p:[],startedAt:n===1?0:am(o.startedAt)};return m.status===`complete`&&(!m.finishReason||!m.content&&!m.reasoning&&!m.tools.length)||(a+=cm(JSON.stringify(m)),a>Hp.jsonBytes)?$p():m});return{id:c,title:tm(s.title,512),systemPrompt:tm(s.systemPrompt),turns:u,updatedAt:rm(s.updatedAt,!0)}})}function um(e,t={}){return dm(e,t,qp)}function dm(e,t,n){let r=lm(e,t,n);return cm(JSON.stringify(r))>Hp.jsonBytes?$p():r}function fm(e,t={}){let n=JSON.stringify({version:qp,conversations:um(e,t)});return cm(n)>Hp.jsonBytes?$p():n}function pm(e,t={}){if(cm(e)>Hp.jsonBytes)return $p();let n;try{n=JSON.parse(e)}catch{return $p()}let r=em(n,[`version`,`conversations`]);return r.version!==1&&r.version!==2?$p():dm(r.conversations,t,r.version)}function mm(e){return new Qp(e instanceof DOMException&&e.name===`QuotaExceededError`?`Local history storage is full. Export or clear history, or disable persistence; the current conversation remains in memory.`:`Local history storage is unavailable. The current conversation remains in memory.`,{cause:e})}function hm(e={}){let t=!1,n=!1,r=0,i,a,o=new Set,s=()=>{let t=e.indexedDB??globalThis.indexedDB;if(!t)throw new Qp(`This browser does not support local history storage.`);return t},c=()=>{r++;for(let e of o)try{e.abort()}catch{}o.clear(),i?.close(),i=void 0,a=void 0},l=()=>{if(i)return Promise.resolve(i);if(a)return a;let e=r;return a=new Promise((n,a)=>{let o=s().open(Up,Kp);o.onupgradeneeded=()=>{o.result.objectStoreNames.contains(Wp)||o.result.createObjectStore(Wp)},o.onerror=()=>a(mm(o.error));let l=!1;o.onblocked=()=>{l=!0,a(new Qp(`Close other WebUI tabs to access local history.`))},o.onsuccess=()=>{l||!t||e!==r?(o.result.close(),a(new Qp(`Local history persistence was disabled.`))):(i=o.result,i.onversionchange=c,n(i))}}),a.catch(t=>{throw e===r&&(a=void 0),t})};async function u(e,i){if(!t||n)return;let a=r;try{let s=await l();return!t||n||r!==a?void 0:await new Promise((t,n)=>{let r=s.transaction(Wp,e);o.add(r);let a=i(r.objectStore(Wp));r.oncomplete=()=>{o.delete(r),t(a.result)},r.onabort=r.onerror=()=>{o.delete(r),n(mm(r.error??a.error))}})}catch(e){throw e instanceof Qp?e:mm(e)}}return{setEnabled(e){t!==e&&(t=e,e||c())},async load(e={}){let t=await u(`readonly`,e=>e.get(Gp));return t===void 0?[]:pm(tm(t,Hp.jsonBytes),e)},async save(e,n={}){if(!t)return;let r=fm(e,n);await u(`readwrite`,e=>e.put(r,Gp))},async clear(){if(n)throw new Qp(`Local history is already being cleared.`);n=!0;let e=!1;c();try{await new Promise((t,r)=>{let i=s().deleteDatabase(Up);i.onsuccess=()=>{n=!1,t()},i.onerror=()=>{n=!1,r(mm(i.error))},i.onblocked=()=>{e=!0,r(new Qp(`Close other WebUI tabs to clear local history.`))}})}catch(t){throw e||(n=!1),t instanceof Qp?t:mm(t)}},close:c}}var gm=[],_m=0,vm=0,ym=0,bm=new Set;function xm(e){return bm.add(e),()=>{bm.delete(e)}}function Sm(){return _m}function Cm(e){return{id:crypto.randomUUID(),title:e,systemPrompt:``,turns:[],updatedAt:Date.now()}}var wm=new WeakMap;function Tm(e){let t=wm.get(e);if(t!==void 0)return t;let n=new TextEncoder().encode(JSON.stringify(e)).byteLength;return wm.set(e,n),n}function Em(e){return e.length>Hp.conversations||e.reduce((e,t)=>e+t.turns.length,0)>Hp.totalTurns?!1:2+Math.max(0,e.length-1)+e.reduce((e,t)=>e+new TextEncoder().encode(JSON.stringify({...t,turns:[]})).byteLength+t.turns.reduce((e,t)=>e+Tm(t),0)+Math.max(0,t.turns.length-1),0)<=Hp.jsonBytes}function Dm(){for(let e of bm)e()}function Om(e){gm=e,Dm()}function km(e){if(!Em(e))throw Error(`Conversation memory limit exceeded.`);_m++,vm=0,Om(e)}function Am(){ym++,vm=ym,Dm()}function jm(){return(0,z.useSyncExternalStore)(xm,()=>vm)}function Mm(){let e=vm!==0;return vm=0,e}function Nm(e){let t=gm.some(t=>t.id===e.id)?gm.map(t=>t.id===e.id?e:t):[...gm,e];return Em(t)?(Om(t),!0):!1}function Pm(){return(0,z.useSyncExternalStore)(xm,()=>gm)}var Fm=e=>[`succeeded`,`failed`,`cancelled`].includes(e.state),Im=e=>e.auth.status===`authenticated`&&e.bootstrap!==null&&e.catalogSequence!==null&&[`ready`,`streaming`,`polling`].includes(e.connection);function Lm(e,t){return Im(e)&&e.bootstrap?.actions[t]?.state===`enabled`}function Rm(e,t){return[...e.operations.values()].some(e=>!Fm(e)&&e.target.target_kind===`model`&&(e.target.model_id===t||e.target.eviction_target_id===t))||[...e.pendingReconciliations.values()].some(e=>e.modelId===t)}function zm(e,t){return Lm(e,`load`)&&t.supported&&t.complete&&t.metadata.support.runnable_on_backend&&!t.lifecycle.busy&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Rm(e,t.identity.id)}function Bm(e,t){return Lm(e,`unload`)&&t.lifecycle.state===`ready`&&!Rm(e,t.identity.id)}function Vm(e,t){return Lm(e,`cache_delete`)&&t.identity.source===`cache`&&t.removable&&t.removal.eligible&&!t.lifecycle.busy&&t.lifecycle.active_requests===0&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Rm(e,t.identity.id)}function Hm(e,t){return Im(e)&&t.lifecycle.state===`ready`&&t.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available)}function Um(e,t){return e.catalog.filter(n=>n.identity.id!==t&&Bm(e,n)&&!n.lifecycle.busy&&n.lifecycle.active_requests===0&&n.lifecycle.draining_requests===0)}function Wm(e,t){if(e===null)return L(t,`models.library.unknown`);if(e===0)return`0 B`;let n=Math.min(4,Math.floor(Math.log(e)/Math.log(1024)));return`${(e/1024**n).toLocaleString(t,{maximumFractionDigits:1})} ${[`B`,`KiB`,`MiB`,`GiB`,`TiB`][n]}`}function Gm(e,t){return e instanceof To?e.envelope?.error.code===`stale_revision`?L(t,`models.library.stale`):e.message:L(t,`models.library.pending`)}var Km={column:`name`,direction:`asc`},qm={ready:0,loading:1,draining:2,unloading:3,failed:4,unloaded:5},Jm=new Intl.Collator(void 0,{numeric:!0});function Ym(e){let t=e.lifecycle.state;return t===`ready`?0:t===`loading`||t===`draining`||t===`unloading`?1:2}function Xm(e){return e.metadata.quantization??e.metadata.dtype??null}function Zm(e,t){return t===`size`?e.metadata.disk_bytes:t===`quantization`?Xm(e):t===`state`?qm[e.lifecycle.state]:e.identity.display_name}function Qm(e){return(t,n)=>{let r=Ym(t)-Ym(n);if(r!==0)return r;let i=Zm(t,e.column),a=Zm(n,e.column);if(i===null||a===null){if(i!==a)return i===null?1:-1}else{let t=typeof i==`number`&&typeof a==`number`?i-a:Jm.compare(String(i),String(a));if(t!==0)return e.direction===`asc`?t:-t}return Jm.compare(t.identity.display_name,n.identity.display_name)||t.identity.id.localeCompare(n.identity.id)}}function $m(e,t,n=Km){let r=t.query.trim().toLocaleLowerCase();return e.filter(e=>(!r||[e.identity.display_name,e.identity.inference_id,e.metadata.architecture??``].some(e=>e.toLocaleLowerCase().includes(r)))&&(!t.source||e.identity.source===t.source)&&(!t.task||e.capabilities.some(e=>e.task===t.task))&&(!t.status||e.lifecycle.state===t.status)).sort(Qm(n))}function eh(e){let t=[...new Set(e.capabilities.map(e=>e.task))],n=t=>{let n=e.metadata.output_tasks.indexOf(t);return n===-1?e.metadata.output_tasks.length:n};return t.map((e,t)=>({task:e,index:t})).sort((e,t)=>n(e.task)-n(t.task)||e.index-t.index).map(e=>e.task)}function th(e,t){let n=[...e.operations.values()].filter(e=>e.kind===`model_load`&&e.target.target_kind===`model`&&e.target.model_id===t).sort((e,t)=>t.created_at.localeCompare(e.created_at))[0];return n?.state===`failed`&&n.error?.code===`conflict`}function nh(e,t=5){return[...e.operations.values()].filter(e=>e.kind===`download`&&e.target.target_kind===`download`&&e.state!==`succeeded`).sort((e,t)=>Number(Fm(e))-Number(Fm(t))||t.created_at.localeCompare(e.created_at)).slice(0,t)}function rh(e){let t=e.split(`/`);return t.length===2&&t.every(e=>e.length<=96&&/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(e))}function ih(e){return e.length===0||e.length<=128&&e.split(`/`).every(e=>e!==`.`&&e!==`..`&&/^[A-Za-z0-9._~+-]+$/.test(e))}var ah={filter:{query:``,source:``,task:``,status:``},sort:Km,instance:null},oh=ah,sh=new Set;function ch(e){return sh.add(e),()=>sh.delete(e)}function lh(e){oh=e;for(let e of sh)e()}function uh(e){let t=(0,z.useSyncExternalStore)(ch,()=>oh);return t.instance===null||t.instance===e?t:ah}function dh(e){return oh.instance===null||oh.instance===e?oh:ah}function fh(e,t){lh({...dh(e),filter:t,instance:e})}function ph(e,t){lh({...dh(e),sort:t,instance:e})}function mh(){lh(ah)}var hh=[`appearance`,`requests`,`model`,`server`],gh=`appearance`;function _h(e){return hh.includes(e??``)?e:gh}function vh(e){return`#settings/${e}`}var yh=[`models`,`chat`,`activity`,`settings`,`gallery`];function bh(e){let t=null,n=()=>{if(t)return t;let n=e();return t=n,n.then(e=>{Object.assign(n,{status:`fulfilled`,value:e})},e=>{Object.assign(n,{status:`rejected`,reason:e})}),n};function r(e){let t=(0,z.use)(n());return(0,U.jsx)(t,{...e})}return{Route:r,preload:n}}var xh=bh(()=>P(()=>import(`./screen-DY8g_dbJ.js`).then(e=>e.ModelsLibrary),__vite__mapDeps([0,1,2,3,4,5,6,7,8,9,10,11,12]),import.meta.url)),Sh=bh(()=>P(()=>import(`./chat-CVipOb7r.js`).then(e=>e.Chat),__vite__mapDeps([13,1,2,3,4,5,6,7,8,9,10,14,12,11,15,16]),import.meta.url)),Ch=bh(()=>P(()=>import(`./activity-D7FGoQJ6.js`).then(e=>e.ActivityPage),__vite__mapDeps([17,1,2,3,4,5,6,7,8,9,10,15,14]),import.meta.url)),wh=bh(()=>P(()=>import(`./settings-screen-CJRvqCAJ.js`).then(e=>e.SettingsScreen),__vite__mapDeps([18,1,2,3,4,5,6,7,8,9,10,16,11]),import.meta.url)),Th=bh(()=>P(()=>import(`./gallery-B2Rl3xjE.js`).then(e=>e.DesignGallery),__vite__mapDeps([19,1,2,3,4,5,6,7,8,9,10,14]),import.meta.url)),Eh=xh.Route,Dh=Sh.Route,Oh=Ch.Route,kh=wh.Route,Ah=Th.Route,jh=class extends z.Component{state={failed:!1,resetKey:this.props.resetKey};static getDerivedStateFromError(){return{failed:!0}}static getDerivedStateFromProps(e,t){return e.resetKey===t.resetKey?null:{failed:!1,resetKey:e.resetKey}}render(){if(!this.state.failed)return this.props.children;let e=this.props.locale;return(0,U.jsx)(je,{title:L(e,`route.load_failed.title`),body:L(e,`route.load_failed.body`),testId:R(`route.load_failed.title`),action:(0,U.jsx)(de,{onClick:()=>window.location.reload(),children:L(e,`common.reload`)})})}};function Mh(){return Promise.all([xh.preload(),Sh.preload(),Ch.preload(),wh.preload(),Th.preload()])}function Nh(e){return e.auth.status===`authenticated`&&e.connection!==`schema-mismatch`}function Ph(e=window.location.hash){let t=e.replace(/^#/,``).replace(/^\//,``),[n,...r]=t.split(`/`);return n===`settings`?{route:`settings`,section:_h(r.join(`/`)||void 0)}:{route:yh.includes(t)?t:`models`,section:gh}}function Fh(){let e=Bp(),t=Vp(),[n,r]=(0,z.useState)(()=>Ph()),i=n.route,[a,o]=(0,z.useState)(Af),[s,c]=(0,z.useState)(null),[l,u]=(0,z.useState)(null),d=(0,z.useRef)(0);(0,z.useEffect)(()=>()=>{d.current+=1},[]),(0,z.useEffect)(()=>{let e=()=>{Mh().catch(()=>void 0)};if(typeof window.requestIdleCallback==`function`){let t=window.requestIdleCallback(e,{timeout:2e3});return()=>window.cancelIdleCallback(t)}let t=window.setTimeout(e,500);return()=>window.clearTimeout(t)},[]),(0,z.useEffect)(()=>{let e=()=>r(Ph());return window.addEventListener(`hashchange`,e),()=>window.removeEventListener(`hashchange`,e)},[]);let[f,p]=(0,z.useState)(_f),m=vf(a.colorScheme,f);(0,z.useLayoutEffect)(()=>{Nf(document.documentElement,a,f),jf(a)},[a,f]),(0,z.useEffect)(()=>{if(a.colorScheme===`system`)return p(_f()),bf(p)},[a.colorScheme]),(0,z.useEffect)(()=>{let e=()=>{document.documentElement.dataset.documentHidden=String(document.hidden)};return e(),document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e)},[]),(0,z.useEffect)(()=>{e.auth.status===`authenticated`&&u(null)},[e.auth.status]),(0,z.useEffect)(()=>{e.auth.status===`signed-out`&&km([])},[e.auth.status]);let h=e=>{r({route:e,section:gh}),window.history.replaceState(null,``,`#${e}`)},g=e=>{r({route:`settings`,section:e}),window.history.replaceState(null,``,vh(e))},_=e=>{let n=d.current+1;d.current=n,u(null),t.login(e).catch(e=>{d.current===n&&(e instanceof DOMException&&e.name===`AbortError`||u(Po(e)))})},v=()=>{d.current+=1,u(null),mh(),t.logout()},y=()=>{u(null),t.refresh()},b=()=>{v(),window.location.reload()},x=n=>{let r=zo(e,n);r!==e.selectedModelId&&t.selectModel(r),r!==null&&Jf(),h(`models`),c(null)},S=()=>{Nh(e)&&Am(),h(`chat`),c(null)},C=Ro(e),w=e.auth.status===`authenticated`&&e.catalogSequence!==null?C.map(e=>({id:e.identity.id,name:e.identity.display_name,state:e.lifecycle.state,stateLabel:Xo(a.locale,e.lifecycle.state)})):null,T=Ih(n,a,o,g,{snapshot:e,authFailure:l,login:_,logout:v,retry:y,recoverSchema:b});return(0,U.jsxs)(Uf,{family:a.themeFamily,scheme:m,locale:a.locale,children:[(0,U.jsx)(rf,{locale:a.locale,route:i,onRouteChange:h,onCommand:()=>c(`command`),onHelp:()=>c(`help`),onNewChat:S,onOpenModel:x,loadedModels:w,connection:{label:Bo(a.locale,e),state:e.connection,details:Vo(a.locale,e),summary:e.bootstrap===null?null:qo(a.locale,e)},sessionAction:e.auth.tokenPresent?(0,U.jsx)(ae,{label:L(a.locale,`toolbar.logout`),icon:`key`,onClick:v,"data-testid":R(`toolbar.logout`)}):null,inspector:null,children:(0,U.jsx)(jh,{locale:a.locale,resetKey:i,children:(0,U.jsx)(z.Suspense,{fallback:null,children:T})})}),(0,U.jsx)(es,{open:s===`command`,locale:a.locale,catalog:e.catalog,loaded:C,onClose:()=>c(null),onNavigate:e=>{h(e),c(null)},onNewChat:S,onOpenModel:x}),(0,U.jsxs)(Se,{open:s===`help`,title:L(a.locale,`help.title`),onClose:()=>c(null),testId:`help-dialog`,closeLabel:L(a.locale,`common.close`),children:[(0,U.jsx)(`p`,{"data-testid":R(`help.body`),children:L(a.locale,`help.body`)}),(0,U.jsx)(`ul`,{className:`shortcut-list`,"data-testid":`help-shortcuts`,children:nf.map(e=>(0,U.jsxs)(`li`,{"data-testid":`help-shortcut-${e.id}`,children:[(0,U.jsx)(`span`,{className:`shortcut-keys`,children:e.keys.map((t,n)=>(0,U.jsxs)(z.Fragment,{children:[n>0?` ${e.separator} `:null,(0,U.jsx)(`kbd`,{children:t})]},t))}),(0,U.jsx)(`span`,{children:L(a.locale,e.key)})]},e.id))})]})]})}function Ih(e,t,n,r,i){let{route:a}=e;return a===`models`?(0,U.jsx)(Lh,{locale:t.locale,context:i}):a===`chat`?(0,U.jsx)(Rh,{locale:t.locale,context:i}):a===`activity`?(0,U.jsx)(zh,{locale:t.locale,context:i}):a===`settings`?(0,U.jsx)(kh,{appearance:t,setAppearance:n,section:e.section,onSectionChange:r}):(0,U.jsx)(Ah,{locale:t.locale})}function Lh(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,U.jsx)(Eh,{locale:e.locale}):(0,U.jsx)(Uo,{locale:e.locale,title:L(e.locale,`models.title`),titleTestId:R(`models.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function Rh(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,U.jsx)(Dh,{locale:e.locale}):(0,U.jsx)(Uo,{locale:e.locale,title:L(e.locale,`chat.title`),titleTestId:R(`chat.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function zh(e){return e.context.snapshot.auth.status===`authenticated`&&![`schema-mismatch`,`forbidden`,`unauthorized`].includes(e.context.snapshot.connection)?(0,U.jsx)(Oh,{locale:e.locale}):(0,U.jsx)(Uo,{locale:e.locale,title:L(e.locale,`activity.title`),titleTestId:R(`activity.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}var Bh=document.getElementById(`root`);if(Bh===null)throw Error(`Missing #root element for mlxcel WebUI.`);(0,Ot.createRoot)(Bh).render((0,U.jsx)(z.StrictMode,{children:(0,U.jsx)(zp,{children:(0,U.jsx)(Fh,{})})}));export{qi as $,Pm as A,tp as B,ih as C,km as D,Cm as E,Bp as F,To as G,Fd as H,Vp as I,Na as J,Ia as K,ip as L,hm as M,fm as N,Sm as O,pm as P,Gi as Q,Xf as R,rh as S,Mm as T,Sl as U,Pd as V,Xo as W,Qi as X,Pa as Y,Ji as Z,Um as _,Km as a,Xm as b,Hm as c,Bm as d,th as f,Gm as g,eh as h,uh as i,jm as j,Nm as k,Vm as l,Im as m,fh as n,Lm as o,Qm as p,La as q,ph as r,Wm as s,hh as t,zm as u,$m as v,nh as w,Fm as x,Rm as y,Yf as z};