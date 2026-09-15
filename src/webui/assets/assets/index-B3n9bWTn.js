var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},c=(n,r,o)=>(o=n==null?{}:e(i(n)),s(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n)),l=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.portal`),r=Symbol.for(`react.fragment`),i=Symbol.for(`react.strict_mode`),a=Symbol.for(`react.profiler`),o=Symbol.for(`react.consumer`),s=Symbol.for(`react.context`),c=Symbol.for(`react.forward_ref`),l=Symbol.for(`react.suspense`),u=Symbol.for(`react.memo`),d=Symbol.for(`react.lazy`),f=Symbol.for(`react.activity`),p=Symbol.for(`react.view_transition`),m=Symbol.iterator;function h(e){return typeof e!=`object`||!e?null:(e=m&&e[m]||e[`@@iterator`],typeof e==`function`?e:null)}var g={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},_=Object.assign,v={};function y(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}y.prototype.isReactComponent={},y.prototype.setState=function(e,t){if(typeof e!=`object`&&typeof e!=`function`&&e!=null)throw Error(`takes an object of state variables to update or a function which returns an object of state variables.`);this.updater.enqueueSetState(this,e,t,`setState`)},y.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,`forceUpdate`)};function b(){}b.prototype=y.prototype;function x(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}var ee=x.prototype=new b;ee.constructor=x,_(ee,y.prototype),ee.isPureReactComponent=!0;var te=Array.isArray;function ne(){}var S={H:null,A:null,T:null,S:null},C=Object.prototype.hasOwnProperty;function w(e,n,r){var i=r.ref;return{$$typeof:t,type:e,key:n,ref:i===void 0?null:i,props:r}}function re(e,t){return w(e.type,t,e.props)}function T(e){return typeof e==`object`&&!!e&&e.$$typeof===t}function ie(e){var t={"=":`=0`,":":`=2`};return`$`+e.replace(/[=:]/g,function(e){return t[e]})}var E=/\/+/g;function ae(e,t){return typeof e==`object`&&e&&e.key!=null?ie(``+e.key):t.toString(36)}function oe(e){switch(e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason;default:switch(typeof e.status==`string`?e.then(ne,ne):(e.status=`pending`,e.then(function(t){e.status===`pending`&&(e.status=`fulfilled`,e.value=t)},function(t){e.status===`pending`&&(e.status=`rejected`,e.reason=t)})),e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason}}throw e}function D(e,r,i,a,o){var s=typeof e;(s===`undefined`||s===`boolean`)&&(e=null);var c=!1;if(e===null)c=!0;else switch(s){case`bigint`:case`string`:case`number`:c=!0;break;case`object`:switch(e.$$typeof){case t:case n:c=!0;break;case d:return c=e._init,D(c(e._payload),r,i,a,o)}}if(c)return o=o(e),c=a===``?`.`+ae(e,0):a,te(o)?(i=``,c!=null&&(i=c.replace(E,`$&/`)+`/`),D(o,r,i,``,function(e){return e})):o!=null&&(T(o)&&(o=re(o,i+(o.key==null||e&&e.key===o.key?``:(``+o.key).replace(E,`$&/`)+`/`)+c)),r.push(o)),1;c=0;var l=a===``?`.`:a+`:`;if(te(e))for(var u=0;u<e.length;u++)a=e[u],s=l+ae(a,u),c+=D(a,r,i,s,o);else if(u=h(e),typeof u==`function`)for(e=u.call(e),u=0;!(a=e.next()).done;)a=a.value,s=l+ae(a,u++),c+=D(a,r,i,s,o);else if(s===`object`){if(typeof e.then==`function`)return D(oe(e),r,i,a,o);throw r=String(e),Error(`Objects are not valid as a React child (found: `+(r===`[object Object]`?`object with keys {`+Object.keys(e).join(`, `)+`}`:r)+`). If you meant to render a collection of children, use an array instead.`)}return c}function O(e,t,n){if(e==null)return e;var r=[],i=0;return D(e,r,``,``,function(e){return t.call(n,e,i++)}),r}function se(e){if(e._status===-1){var t=e._result,n=t();n.then(function(t){(e._status===0||e._status===-1)&&(e._status=1,e._result=t,n.status===void 0&&(n.status=`fulfilled`,n.value=t))},function(t){(e._status===0||e._status===-1)&&(e._status=2,e._result=t,n.status===void 0&&(n.status=`rejected`,n.reason=t))}),e._status===-1&&(e._status=0,e._result=n)}if(e._status===1)return e._result.default;throw e._result}var ce=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)};function le(e){var t=S.T,n={};n.types=t===null?null:t.types,S.T=n;try{var r=e(),i=S.S;i!==null&&i(n,r),typeof r==`object`&&r&&typeof r.then==`function`&&r.then(ne,ce)}catch(e){ce(e)}finally{t!==null&&n.types!==null&&(t.types=n.types),S.T=t}}function ue(e){var t=S.T;if(t!==null){var n=t.types;n===null?t.types=[e]:n.indexOf(e)===-1&&n.push(e)}else le(ue.bind(null,e))}var de={map:O,forEach:function(e,t,n){O(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return O(e,function(){t++}),t},toArray:function(e){return O(e,function(e){return e})||[]},only:function(e){if(!T(e))throw Error(`React.Children.only expected to receive a single React element child.`);return e}};e.Activity=f,e.Children=de,e.Component=y,e.Fragment=r,e.Profiler=a,e.PureComponent=x,e.StrictMode=i,e.Suspense=l,e.ViewTransition=p,e.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=S,e.__COMPILER_RUNTIME={__proto__:null,c:function(e){return S.H.useMemoCache(e)}},e.addTransitionType=ue,e.cache=function(e){return function(){return e.apply(null,arguments)}},e.cacheSignal=function(){return null},e.cloneElement=function(e,t,n){if(e==null)throw Error(`The argument must be a React element, but you passed `+e+`.`);var r=_({},e.props),i=e.key;if(t!=null)for(a in t.key!==void 0&&(i=``+t.key),t)!C.call(t,a)||a===`key`||a===`__self`||a===`__source`||a===`ref`&&t.ref===void 0||(r[a]=t[a]);var a=arguments.length-2;if(a===1)r.children=n;else if(1<a){for(var o=Array(a),s=0;s<a;s++)o[s]=arguments[s+2];r.children=o}return w(e.type,i,r)},e.createContext=function(e){return e={$$typeof:s,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null},e.Provider=e,e.Consumer={$$typeof:o,_context:e},e},e.createElement=function(e,t,n){var r,i={},a=null;if(t!=null)for(r in t.key!==void 0&&(a=``+t.key),t)C.call(t,r)&&r!==`key`&&r!==`__self`&&r!==`__source`&&(i[r]=t[r]);var o=arguments.length-2;if(o===1)i.children=n;else if(1<o){for(var s=Array(o),c=0;c<o;c++)s[c]=arguments[c+2];i.children=s}if(e&&e.defaultProps)for(r in o=e.defaultProps,o)i[r]===void 0&&(i[r]=o[r]);return w(e,a,i)},e.createRef=function(){return{current:null}},e.forwardRef=function(e){return{$$typeof:c,render:e}},e.isValidElement=T,e.lazy=function(e){return{$$typeof:d,_payload:{_status:-1,_result:e},_init:se}},e.memo=function(e,t){return{$$typeof:u,type:e,compare:t===void 0?null:t}},e.startTransition=le,e.unstable_useCacheRefresh=function(){return S.H.useCacheRefresh()},e.use=function(e){return S.H.use(e)},e.useActionState=function(e,t,n){return S.H.useActionState(e,t,n)},e.useCallback=function(e,t){return S.H.useCallback(e,t)},e.useContext=function(e){return S.H.useContext(e)},e.useDebugValue=function(){},e.useDeferredValue=function(e,t){return S.H.useDeferredValue(e,t)},e.useEffect=function(e,t){return S.H.useEffect(e,t)},e.useEffectEvent=function(e){return S.H.useEffectEvent(e)},e.useId=function(){return S.H.useId()},e.useImperativeHandle=function(e,t,n){return S.H.useImperativeHandle(e,t,n)},e.useInsertionEffect=function(e,t){return S.H.useInsertionEffect(e,t)},e.useLayoutEffect=function(e,t){return S.H.useLayoutEffect(e,t)},e.useMemo=function(e,t){return S.H.useMemo(e,t)},e.useOptimistic=function(e,t){return S.H.useOptimistic(e,t)},e.useReducer=function(e,t,n){return S.H.useReducer(e,t,n)},e.useRef=function(e){return S.H.useRef(e)},e.useState=function(e){return S.H.useState(e)},e.useSyncExternalStore=function(e,t,n){return S.H.useSyncExternalStore(e,t,n)},e.useTransition=function(){return S.H.useTransition()},e.version=`19.3.0`})),u=o(((e,t)=>{t.exports=l()})),d=o((e=>{function t(e,t){var n=e.length;e.push(t);a:for(;0<n;){var r=n-1>>>1,a=e[r];if(0<i(a,t))e[r]=t,e[n]=a,n=r;else break a}}function n(e){return e.length===0?null:e[0]}function r(e){if(e.length===0)return null;var t=e[0],n=e.pop();if(n!==t){e[0]=n;a:for(var r=0,a=e.length,o=a>>>1;r<o;){var s=2*(r+1)-1,c=e[s],l=s+1,u=e[l];if(0>i(c,n))l<a&&0>i(u,c)?(e[r]=u,e[l]=n,r=l):(e[r]=c,e[s]=n,r=s);else if(l<a&&0>i(u,n))e[r]=u,e[l]=n,r=l;else break a}}return t}function i(e,t){var n=e.sortIndex-t.sortIndex;return n===0?e.id-t.id:n}if(e.unstable_now=void 0,typeof performance==`object`&&typeof performance.now==`function`){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var c=[],l=[],u=1,d=null,f=3,p=!1,m=!1,h=!1,g=!1,_=typeof setTimeout==`function`?setTimeout:null,v=typeof clearTimeout==`function`?clearTimeout:null,y=typeof setImmediate<`u`?setImmediate:null;function b(e){for(var i=n(l);i!==null;){if(i.callback===null)r(l);else if(i.startTime<=e)r(l),i.sortIndex=i.expirationTime,t(c,i);else break;i=n(l)}}function x(e){if(h=!1,b(e),!m){if(n(c)!==null)m=!0,ee||(ee=!0,re());else{var t=n(l);t!==null&&E(x,t.startTime-e)}}}var ee=!1,te=-1,ne=5,S=-1;function C(){return g?!0:!(e.unstable_now()-S<ne)}function w(){if(g=!1,ee){var t=e.unstable_now();S=t;var i=!0;try{a:{m=!1,h&&(h=!1,v(te),te=-1),p=!0;var a=f;try{b:{for(b(t),d=n(c);d!==null&&!(d.expirationTime>t&&C());){var o=d.callback;if(typeof o==`function`){d.callback=null,f=d.priorityLevel;var s=o(d.expirationTime<=t);if(t=e.unstable_now(),typeof s==`function`){d.callback=s,b(t),i=!0;break b}d===n(c)&&r(c),b(t)}else r(c);d=n(c)}if(d!==null)i=!0;else{var u=n(l);u!==null&&E(x,u.startTime-t),i=!1}}break a}finally{d=null,f=a,p=!1}i=void 0}}finally{i?re():ee=!1}}}var re;if(typeof y==`function`)re=function(){y(w)};else if(typeof MessageChannel<`u`){var T=new MessageChannel,ie=T.port2;T.port1.onmessage=w,re=function(){ie.postMessage(null)}}else re=function(){_(w,0)};function E(t,n){te=_(function(){t(e.unstable_now())},n)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(e){e.callback=null},e.unstable_forceFrameRate=function(e){0>e||125<e?console.error(`forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported`):ne=0<e?Math.floor(1e3/e):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_next=function(e){switch(f){case 1:case 2:case 3:var t=3;break;default:t=f}var n=f;f=t;try{return e()}finally{f=n}},e.unstable_requestPaint=function(){g=!0},e.unstable_runWithPriority=function(e,t){switch(e){case 1:case 2:case 3:case 4:case 5:break;default:e=3}var n=f;f=e;try{return t()}finally{f=n}},e.unstable_scheduleCallback=function(r,i,a){var o=e.unstable_now();switch(typeof a==`object`&&a?(a=a.delay,a=typeof a==`number`&&0<a?o+a:o):a=o,r){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=a+s,r={id:u++,callback:i,priorityLevel:r,startTime:a,expirationTime:s,sortIndex:-1},a>o?(r.sortIndex=a,t(l,r),n(c)===null&&r===n(l)&&(h?(v(te),te=-1):h=!0,E(x,a-o))):(r.sortIndex=s,t(c,r),m||p||(m=!0,ee||(ee=!0,re()))),r},e.unstable_shouldYield=C,e.unstable_wrapCallback=function(e){var t=f;return function(){var n=f;f=t;try{return e.apply(this,arguments)}finally{f=n}}}})),f=o(((e,t)=>{t.exports=d()})),p=o((e=>{var t=u();function n(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function r(){}var i={d:{f:r,r:function(){throw Error(n(522))},D:r,C:r,L:r,m:r,X:r,S:r,M:r},p:0,findDOMNode:null},a=Symbol.for(`react.portal`),o=Symbol.for(`react.recoverable`),s=Symbol.for(`react.optimistic_key`);function c(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:a,key:r==null?null:r===s?s:``+r,children:e,containerInfo:t,implementation:n}}var l=t.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function d(e,t){if(e===`font`)return``;if(typeof t==`string`)return t===`use-credentials`?t:``}e.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=i,e.browser=function(e){return{$$typeof:o,_reason:e}},e.createPortal=function(e,t){var r=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)throw Error(n(299));return c(e,t,null,r)},e.flushSync=function(e){var t=l.T,n=i.p;try{if(l.T=null,i.p=2,e)return e()}finally{l.T=t,i.p=n,i.d.f()}},e.preconnect=function(e,t){typeof e==`string`&&(t?(t=t.crossOrigin,t=typeof t==`string`?t===`use-credentials`?t:``:void 0):t=null,i.d.C(e,t))},e.prefetchDNS=function(e){typeof e==`string`&&i.d.D(e)},e.preinit=function(e,t){if(typeof e==`string`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin),a=typeof t.integrity==`string`?t.integrity:void 0,o=typeof t.fetchPriority==`string`?t.fetchPriority:void 0;n===`style`?i.d.S(e,typeof t.precedence==`string`?t.precedence:void 0,{crossOrigin:r,integrity:a,fetchPriority:o}):n===`script`&&i.d.X(e,{crossOrigin:r,integrity:a,fetchPriority:o,nonce:typeof t.nonce==`string`?t.nonce:void 0})}},e.preinitModule=function(e,t){if(typeof e==`string`){if(typeof t==`object`&&t){if(t.as==null||t.as===`script`){var n=d(t.as,t.crossOrigin);i.d.M(e,{crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}}else t??i.d.M(e)}},e.preload=function(e,t){if(typeof e==`string`&&typeof t==`object`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin);i.d.L(e,n,{crossOrigin:r,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,type:typeof t.type==`string`?t.type:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0,referrerPolicy:typeof t.referrerPolicy==`string`?t.referrerPolicy:void 0,imageSrcSet:typeof t.imageSrcSet==`string`?t.imageSrcSet:void 0,imageSizes:typeof t.imageSizes==`string`?t.imageSizes:void 0,media:typeof t.media==`string`?t.media:void 0})}},e.preloadModule=function(e,t){if(typeof e==`string`){if(t){var n=d(t.as,t.crossOrigin);i.d.m(e,{as:typeof t.as==`string`&&t.as!==`script`?t.as:void 0,crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}else i.d.m(e)}},e.requestFormReset=function(e){i.d.r(e)},e.unstable_batchedUpdates=function(e,t){return e(t)},e.useFormState=function(e,t,n){return l.H.useFormState(e,t,n)},e.useFormStatus=function(){return l.H.useHostTransitionStatus()},e.version=`19.3.0`})),m=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=p()})),h=o((e=>{var t=f(),n=u(),r=m();function i(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function a(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function o(e){for(var t=e,n=t;n&&!n.alternate;)t=n,t.flags&4098&&(e=t.return),n=t.return;for(;t.return;)t=t.return;return t.tag===3?e:null}function s(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function c(e){if(e.tag===31){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function l(e){if(o(e)!==e)throw Error(i(188))}function d(e){var t=e.alternate;if(!t){if(t=o(e),t===null)throw Error(i(188));return t===e?e:null}for(var n=e,r=t;;){var a=n.return;if(a===null)break;var s=a.alternate;if(s===null){if(r=a.return,r!==null){n=r;continue}break}if(a.child===s.child){for(s=a.child;s;){if(s===n)return l(a),e;if(s===r)return l(a),t;s=s.sibling}throw Error(i(188))}if(n.return!==r.return)n=a,r=s;else{for(var c=!1,u=a.child;u;){if(u===n){c=!0,n=a,r=s;break}if(u===r){c=!0,r=a,n=s;break}u=u.sibling}if(!c){for(u=s.child;u;){if(u===n){c=!0,n=s,r=a;break}if(u===r){c=!0,r=s,n=a;break}u=u.sibling}if(!c)throw Error(i(189))}}if(n.alternate!==r)throw Error(i(190))}if(n.tag!==3)throw Error(i(188));return n.stateNode.current===n?e:t}function p(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e;for(e=e.child;e!==null;){if(t=p(e),t!==null)return t;e=e.sibling}return null}function h(e,t,n,r,i,a){for(;e!==null;){if((e.tag===5||e.tag===27||e.tag===6)&&n(e,r,i,a)||(e.tag!==22||e.memoizedState===null)&&(t||e.tag!==5&&e.tag!==27)&&h(e.child,t,n,r,i,a))return!0;e=e.sibling}return!1}function g(e){for(e=e.return;e!==null;){if(e.tag===3||e.tag===5||e.tag===27)return e;e=e.return}return null}function _(e){var t=!1;for(e=e.return;e!==null&&(e.tag===4&&(t=!0),e.tag!==3&&e.tag!==5&&e.tag!==27);)e=e.return;return t}function v(e){var t=[null,null],n=g(e);return n===null||y(t,e,n.child,{foundSelf:!1}),t}function y(e,t,n,r){for(;n!==null;){if(n===t)r.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(r.foundSelf)return e[1]=n,!0;e[0]=n}else if((n.tag!==22||n.memoizedState===null)&&y(e,t,n.child,r))return!0;n=n.sibling}return!1}function b(e){switch(e.tag){case 5:case 27:case 6:return e.stateNode;case 3:return e.stateNode.containerInfo;default:throw Error(i(559))}}var x=null,ee=null;function te(e,t,n){return e===n||e===t&&(x=e,!0)}function ne(e,t,n){return e===n?(ee=e,!1):e===t&&(ee!==null&&(x=e),!0)}function S(e){if(e===null)return null;do e=e===null?null:e.return;while(e&&e.tag!==5&&e.tag!==27&&e.tag!==3);return e||null}function C(e,t,n){for(var r=0,i=e;i;i=n(i))r++;i=0;for(var a=t;a;a=n(a))i++;for(;0<r-i;)e=n(e),r--;for(;0<i-r;)t=n(t),i--;for(;r--;){if(e===t||t!==null&&e===t.alternate)return e;e=n(e),t=n(t)}return null}var w=Object.assign,re=Symbol.for(`react.element`),T=Symbol.for(`react.transitional.element`),ie=Symbol.for(`react.portal`),E=Symbol.for(`react.fragment`),ae=Symbol.for(`react.strict_mode`),oe=Symbol.for(`react.profiler`),D=Symbol.for(`react.consumer`),O=Symbol.for(`react.context`),se=Symbol.for(`react.forward_ref`),ce=Symbol.for(`react.suspense`),le=Symbol.for(`react.suspense_list`),ue=Symbol.for(`react.memo`),de=Symbol.for(`react.lazy`),fe=Symbol.for(`react.activity`),pe=Symbol.for(`react.legacy_hidden`),k=Symbol.for(`react.memo_cache_sentinel`),A=Symbol.for(`react.view_transition`),me=Symbol.for(`react.recoverable`),he=Symbol.iterator;function ge(e){return typeof e!=`object`||!e?null:(e=he&&e[he]||e[`@@iterator`],typeof e==`function`?e:null)}var _e=Symbol.for(`react.client.reference`);function ve(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===_e?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case E:return`Fragment`;case oe:return`Profiler`;case ae:return`StrictMode`;case ce:return`Suspense`;case le:return`SuspenseList`;case fe:return`Activity`;case A:return`ViewTransition`}if(typeof e==`object`)switch(e.$$typeof){case ie:return`Portal`;case O:return e.displayName||`Context`;case D:return(e._context.displayName||`Context`)+`.Consumer`;case se:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case ue:return t=e.displayName||null,t===null?ve(e.type)||`Memo`:t;case de:t=e._payload,e=e._init;try{return ve(e(t))}catch{}}return null}var ye=Array.isArray,j=n.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,M=r.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,be={pending:!1,data:null,method:null,action:null},xe=[],N=-1;function P(e){return{current:e}}function F(e){0>N||(e.current=xe[N],xe[N]=null,N--)}function I(e,t){N++,xe[N]=e.current,e.current=t}var L=P(null),Se=P(null),Ce=P(null),we=P(null);function Te(e,t){switch(I(Ce,t),I(Se,e),I(L,null),t.nodeType){case 9:case 11:e=(e=t.documentElement)&&(e=e.namespaceURI)?up(e):0;break;default:if(e=t.tagName,t=t.namespaceURI)t=up(t),e=dp(t,e);else switch(e){case`svg`:e=1;break;case`math`:e=2;break;default:e=0}}F(L),I(L,e)}function Ee(){F(L),F(Se),F(Ce)}function De(e){var t=e.memoizedState;t!==null&&(sh._currentValue=t.memoizedState,I(we,e)),t=L.current;var n=dp(t,e.type);t!==n&&(I(Se,e),I(L,n))}function Oe(e){Se.current===e&&(F(L),F(Se)),we.current===e&&(F(we),sh._currentValue=be)}var ke,Ae;function je(e){if(ke===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);ke=t&&t[1]||``,Ae=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+ke+e+Ae}var Me=!1;function Ne(e,t){if(!e||Me)return``;Me=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}n=!1;try{var i=Object.getOwnPropertyDescriptor(e.prototype,`props`);Object.defineProperty(e.prototype,"props",{configurable:!0,set:function(){throw Error()}}),n=!0,new e}finally{n&&(i===void 0?delete e.prototype.props:Object.defineProperty(e.prototype,"props",i))}}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,"name",{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Me=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?je(n):``}function Pe(e,t){switch(e.tag){case 26:case 27:case 5:return je(e.type);case 16:return je(`Lazy`);case 13:return e.child!==t&&t!==null?je(`Suspense Fallback`):je(`Suspense`);case 19:return je(`SuspenseList`);case 0:case 15:return Ne(e.type,!1);case 11:return Ne(e.type.render,!1);case 1:return Ne(e.type,!0);case 31:return je(`Activity`);case 30:return je(`ViewTransition`);default:return``}}function Fe(e){try{var t=``,n=null;do t+=Pe(e,n),n=e,e=e.return;while(e);return t}catch(e){return`
Error generating stack: `+e.message+`
`+e.stack}}var Ie=Object.prototype.hasOwnProperty,Le=t.unstable_scheduleCallback,Re=t.unstable_cancelCallback,ze=t.unstable_shouldYield,Be=t.unstable_requestPaint,Ve=t.unstable_now,He=t.unstable_getCurrentPriorityLevel,Ue=t.unstable_ImmediatePriority,We=t.unstable_UserBlockingPriority,Ge=t.unstable_NormalPriority,Ke=t.unstable_LowPriority,R=t.unstable_IdlePriority,qe=t.log,Je=t.unstable_setDisableYieldValue,Ye=null,Xe=null;function Ze(e){if(typeof qe==`function`&&Je(e),Xe&&typeof Xe.setStrictMode==`function`)try{Xe.setStrictMode(Ye,e)}catch{}}var Qe=Math.clz32?Math.clz32:tt,$e=Math.log,et=Math.LN2;function tt(e){return e>>>=0,e===0?32:31-($e(e)/et|0)|0}var nt=256,rt=262144,it=4194304;function at(e){var t=e&42;if(t!==0)return t;switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return e&-e;case 262144:case 524288:case 1048576:case 2097152:return e&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return e&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return e}}function ot(e,t,n){var r=e.pendingLanes;if(r===0)return 0;var i=0,a=e.suspendedLanes,o=e.pingedLanes;e=e.warmLanes;var s=r&134217727;return s===0?(s=r&~a,s===0?o===0?n||(n=r&~e,n!==0&&(i=at(n))):i=at(o):i=at(s)):(r=s&~a,r===0?(o&=s,o===0?n||(n=s&~e,n!==0&&(i=at(n))):i=at(o)):i=at(r)),i===0?0:t!==0&&t!==i&&(t&a)===0&&(a=i&-i,n=t&-t,a>=n||a===32&&n&4194048)?t:i}function st(e,t){return(e.pendingLanes&~(e.suspendedLanes&~e.pingedLanes)&t)===0}function ct(e,t){t&8&&(t|=t&32);var n=e.entangledLanes;if(n!==0)for(e=e.entanglements,n&=t;0<n;){var r=31-Qe(n),i=1<<r;t|=e[r],n&=~i}return t}function lt(e,t){switch(e){case 1:case 2:case 4:case 8:case 64:return t+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function ut(){var e=it;return it<<=1,!(it&62914560)&&(it=4194304),e}function dt(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function ft(e,t){e.pendingLanes|=t,t!==268435456&&(e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0)}function pt(e,t,n,r,i,a){var o=e.pendingLanes;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0,e.expiredLanes&=n,e.entangledLanes&=n,e.errorRecoveryDisabledLanes&=n,e.shellSuspendCounter=0;var s=e.entanglements,c=e.expirationTimes,l=e.hiddenUpdates;for(n=o&~n;0<n;){var u=31-Qe(n),d=1<<u;s[u]=0,c[u]=-1;var f=l[u];if(f!==null)for(l[u]=null,u=0;u<f.length;u++){var p=f[u];p!==null&&(p.lane&=-536870913)}n&=~d}r!==0&&mt(e,r,0),a!==0&&i===0&&e.tag!==0&&(e.suspendedLanes|=a&~(o&~t))}function mt(e,t,n){e.pendingLanes|=t,e.suspendedLanes&=~t;var r=31-Qe(t);e.entangledLanes|=t,e.entanglements[r]=e.entanglements[r]|1073741824|n&261930}function ht(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Qe(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}function gt(e,t){var n=t&-t;return n=n&42?1:_t(n),(n&(e.suspendedLanes|t))===0?n:0}function _t(e){switch(e){case 2:e=1;break;case 8:e=4;break;case 32:e=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:e=128;break;case 268435456:e=134217728;break;default:e=0}return e}function vt(e){return e&=-e,2<e?8<e?e&134217727?32:268435456:8:2}function yt(){var e=M.p;return e===0?(e=window.event,e===void 0?32:Ch(e.type)):e}function bt(e,t){var n=M.p;try{return M.p=e,t()}finally{M.p=n}}var xt=Math.random().toString(36).slice(2),St=`__reactFiber$`+xt,Ct=`__reactProps$`+xt,wt=`__reactContainer$`+xt,Tt=`__reactEvents$`+xt,Et=`__reactListeners$`+xt,Dt=`__reactHandles$`+xt,Ot=`__reactResources$`+xt,kt=`__reactMarker$`+xt,At=`__reactLoad$`+xt;function jt(e){delete e[St],delete e[Ct],delete e[Et],delete e[Dt]}function Mt(e){var t;if(t=e[St])return t;for(var n=e.parentNode;n;){if(t=n[wt]||n[St]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=fm(e);e!==null;){if(n=e[St])return n;e=fm(e)}return t}e=n,n=e.parentNode}return null}function Nt(e){if(e=e[St]||e[wt]){var t=e.tag;if(t===5||t===6||t===13||t===31||t===26||t===27||t===3)return e}return null}function Pt(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e.stateNode;throw Error(i(33))}function Ft(e){var t=e[Ot];return t||=e[Ot]={hoistableStyles:new Map,hoistableScripts:new Map},t}function It(e){e[kt]=!0}function Lt(e){e[At]=void 0}var Rt=new Set,zt={};function Bt(e,t){Vt(e,t),Vt(e+`Capture`,t)}function Vt(e,t){for(zt[e]=t,e=0;e<t.length;e++)Rt.add(t[e])}var Ht=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),Ut={},Wt={};function Gt(e){return Ie.call(Wt,e)?!0:Ie.call(Ut,e)?!1:Ht.test(e)?Wt[e]=!0:(Ut[e]=!0,!1)}var z=!1;function Kt(){var e=z;return z=!1,e}function qt(e,t,n){if(Gt(t)){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:e.removeAttribute(t);return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`){e.removeAttribute(t);return}}e.setAttribute(t,n)}}}function Jt(e,t,n){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(t);return}e.setAttribute(t,n)}}function Yt(e,t,n,r){if(r===null)e.removeAttribute(n);else{switch(typeof r){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(n);return}e.setAttributeNS(t,n,r)}}function Xt(e){switch(typeof e){case`bigint`:case`boolean`:case`number`:case`string`:case`undefined`:return e;case`object`:return e;default:return``}}function Zt(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()===`input`&&(t===`checkbox`||t===`radio`)}function Qt(e,t,n){var r=Object.getOwnPropertyDescriptor(e.constructor.prototype,t);if(!e.hasOwnProperty(t)&&r!==void 0&&typeof r.get==`function`&&typeof r.set==`function`){var i=r.get,a=r.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(e){n=``+e,a.call(this,e)}}),Object.defineProperty(e,t,{enumerable:r.enumerable}),{getValue:function(){return n},setValue:function(e){n=``+e},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function $t(e){if(!e._valueTracker){var t=Zt(e)?`checked`:`value`;e._valueTracker=Qt(e,t,``+e[t])}}function en(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r=``;return e&&(r=Zt(e)?e.checked?`true`:`false`:e.value),e=r,e!==n&&(t.setValue(e),!0)}var tn=/[\n"\\]/g;function nn(e){return e.replace(tn,function(e){return`\\`+e.charCodeAt(0).toString(16)+` `})}function rn(e,t,n,r,i,a,o,s){e.name=``,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`?e.type=o:e.removeAttribute(`type`),t==null?o!==`submit`&&o!==`reset`||e.removeAttribute(`value`):o===`number`?(t===0&&e.value===``||e.value!=t)&&(e.value=``+Xt(t)):e.value!==``+Xt(t)&&(e.value=``+Xt(t)),t==null?n==null?r!=null&&e.removeAttribute(`value`):on(e,Xt(n)):o===`number`&&e.value==t?on(e,Xt(e.value)):on(e,Xt(t)),i==null&&a!=null&&(e.defaultChecked=!!a),i!=null&&(e.checked=i&&typeof i!=`function`&&typeof i!=`symbol`),s!=null&&typeof s!=`function`&&typeof s!=`symbol`&&typeof s!=`boolean`?e.name=``+Xt(s):e.removeAttribute(`name`)}function an(e,t,n,r,i,a,o,s){if(a!=null&&typeof a!=`function`&&typeof a!=`symbol`&&typeof a!=`boolean`&&(e.type=a),t!=null||n!=null){if(!(a!==`submit`&&a!==`reset`||t!=null)){$t(e);return}n=n==null?``:``+Xt(n),t=t==null?n:``+Xt(t),s||t===e.value||(e.value=t),e.defaultValue=t}r??=i,r=typeof r!=`function`&&typeof r!=`symbol`&&!!r,e.checked=s?e.checked:!!r,e.defaultChecked=!!r,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`&&(e.name=o),$t(e)}function on(e,t){e.defaultValue!==``+t&&(e.defaultValue=``+t)}function sn(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t[`$`+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty(`$`+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=``+Xt(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function cn(e,t,n){if(t!=null&&(t=``+Xt(t),t!==e.value&&(e.value=t),n==null)){e.defaultValue!==t&&(e.defaultValue=t);return}e.defaultValue=n==null?``:``+Xt(n)}function ln(e,t,n,r){if(t==null){if(r!=null){if(n!=null)throw Error(i(92));if(ye(r)){if(1<r.length)throw Error(i(93));r=r[0]}n=r}n??=``,t=n}n=Xt(t),e.defaultValue=n,r=e.textContent,r===n&&r!==``&&r!==null&&(e.value=r),$t(e)}function un(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var dn=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `));function fn(e,t,n){var r=t.indexOf(`--`)===0;n==null||typeof n==`boolean`||n===``?r?e.setProperty(t,``):t===`float`?e.cssFloat=``:e[t]=``:r?e.setProperty(t,n):typeof n!=`number`||n===0||dn.has(t)?t===`float`?e.cssFloat=n:e[t]=(``+n).trim():e[t]=n+`px`}function pn(e,t,n){if(t!=null&&typeof t!=`object`)throw Error(i(62));if(e=e.style,n!=null){for(var r in n)!n.hasOwnProperty(r)||t!=null&&t.hasOwnProperty(r)||(r.indexOf(`--`)===0?e.setProperty(r,``):r===`float`?e.cssFloat=``:e[r]=``,z=!0);for(var a in t)r=t[a],t.hasOwnProperty(a)&&n[a]!==r&&(fn(e,a,r),z=!0)}else for(var o in t)t.hasOwnProperty(o)&&fn(e,o,t[o])}function mn(e){if(e.indexOf(`-`)===-1)return!1;switch(e){case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:return!1;default:return!0}}var hn=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`maskType`,`mask-type`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),gn=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function _n(e){return gn.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}function vn(){}var yn=null;function bn(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var xn=null,Sn=null;function Cn(e){var t=Nt(e);if(t&&(e=t.stateNode)){var n=e[Ct]||null;a:switch(e=t.stateNode,t.type){case`input`:if(rn(e,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),t=n.name,n.type===`radio`&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll(`input[name="`+nn(``+t)+`"][type="radio"]`),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var a=r[Ct]||null;if(!a)throw Error(i(90));rn(r,a.value,a.defaultValue,a.defaultValue,a.checked,a.defaultChecked,a.type,a.name)}}for(t=0;t<n.length;t++)r=n[t],r.form===e.form&&en(r)}break a;case`textarea`:cn(e,n.value,n.defaultValue);break a;case`select`:t=n.value,t!=null&&sn(e,!!n.multiple,t,!1)}}}var wn=!1;function Tn(e,t,n){if(wn)return e(t,n);wn=!0;try{return e(t)}finally{if(wn=!1,(xn!==null||Sn!==null)&&(zd(),xn&&(t=xn,e=Sn,Sn=xn=null,Cn(t),e)))for(t=0;t<e.length;t++)Cn(e[t])}}function En(e,t){var n=e.stateNode;if(n===null)return null;var r=n[Ct]||null;if(r===null)return null;n=r[t];a:switch(t){case`onClick`:case`onClickCapture`:case`onDoubleClick`:case`onDoubleClickCapture`:case`onMouseDown`:case`onMouseDownCapture`:case`onMouseMove`:case`onMouseMoveCapture`:case`onMouseUp`:case`onMouseUpCapture`:case`onMouseEnter`:(r=!r.disabled)||(e=e.type,r=e!==`button`&&e!==`input`&&e!==`select`&&e!==`textarea`),e=!r;break a;default:e=!1}if(e)return null;if(n&&typeof n!=`function`)throw Error(i(231,t,typeof n));return n}var Dn=!(typeof window>`u`||window.document===void 0||window.document.createElement===void 0),On=!1;if(Dn)try{var kn={};Object.defineProperty(kn,"passive",{get:function(){On=!0}}),window.addEventListener(`test`,kn,kn),window.removeEventListener(`test`,kn,kn)}catch{On=!1}var An=null,jn=null,Mn=null;function Nn(){if(Mn)return Mn;var e,t=jn,n=t.length,r,i=`value`in An?An.value:An.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Mn=i.slice(e,1<r?1-r:void 0)}function Pn(e){var t=e.keyCode;return`charCode`in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Fn(){return!0}function In(){return!1}function Ln(e){function t(t,n,r,i,a){for(var o in this._reactName=t,this._targetInst=r,this.type=n,this.nativeEvent=i,this.target=a,this.currentTarget=null,e)e.hasOwnProperty(o)&&(t=e[o],this[o]=t?t(i):i[o]);return this.isDefaultPrevented=(i.defaultPrevented==null?!1===i.returnValue:i.defaultPrevented)?Fn:In,this.isPropagationStopped=In,this}return w(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var e=this.nativeEvent;e&&(e.preventDefault?e.preventDefault():typeof e.returnValue!=`unknown`&&(e.returnValue=!1),this.isDefaultPrevented=Fn)},stopPropagation:function(){var e=this.nativeEvent;e&&(e.stopPropagation?e.stopPropagation():typeof e.cancelBubble!=`unknown`&&(e.cancelBubble=!0),this.isPropagationStopped=Fn)},persist:function(){},isPersistent:Fn}),t}var Rn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},zn=Ln(Rn),Bn=w({},Rn,{view:0,detail:0}),Vn=Ln(Bn),Hn,Un,Wn,Gn=w({},Bn,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:nr,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return`movementX`in e?e.movementX:(e!==Wn&&(Wn&&e.type===`mousemove`?(Hn=e.screenX-Wn.screenX,Un=e.screenY-Wn.screenY):Un=Hn=0,Wn=e),Hn)},movementY:function(e){return`movementY`in e?e.movementY:Un}}),Kn=Ln(Gn),qn=Ln(w({},Gn,{dataTransfer:0})),Jn=Ln(w({},Bn,{relatedTarget:0})),Yn=Ln(w({},Rn,{animationName:0,elapsedTime:0,pseudoElement:0})),Xn=Ln(w({},Rn,{clipboardData:function(e){return`clipboardData`in e?e.clipboardData:window.clipboardData}})),Zn=Ln(w({},Rn,{data:0})),Qn={Esc:`Escape`,Spacebar:` `,Left:`ArrowLeft`,Up:`ArrowUp`,Right:`ArrowRight`,Down:`ArrowDown`,Del:`Delete`,Win:`OS`,Menu:`ContextMenu`,Apps:`ContextMenu`,Scroll:`ScrollLock`,MozPrintableKey:`Unidentified`},$n={8:`Backspace`,9:`Tab`,12:`Clear`,13:`Enter`,16:`Shift`,17:`Control`,18:`Alt`,19:`Pause`,20:`CapsLock`,27:`Escape`,32:` `,33:`PageUp`,34:`PageDown`,35:`End`,36:`Home`,37:`ArrowLeft`,38:`ArrowUp`,39:`ArrowRight`,40:`ArrowDown`,45:`Insert`,46:`Delete`,112:`F1`,113:`F2`,114:`F3`,115:`F4`,116:`F5`,117:`F6`,118:`F7`,119:`F8`,120:`F9`,121:`F10`,122:`F11`,123:`F12`,144:`NumLock`,145:`ScrollLock`,224:`Meta`},er={Alt:`altKey`,Control:`ctrlKey`,Meta:`metaKey`,Shift:`shiftKey`};function tr(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=er[e])?!!t[e]:!1}function nr(){return tr}var rr=Ln(w({},Bn,{key:function(e){if(e.key){var t=Qn[e.key]||e.key;if(t!==`Unidentified`)return t}return e.type===`keypress`?(e=Pn(e),e===13?`Enter`:String.fromCharCode(e)):e.type===`keydown`||e.type===`keyup`?$n[e.keyCode]||`Unidentified`:``},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:nr,charCode:function(e){return e.type===`keypress`?Pn(e):0},keyCode:function(e){return e.type===`keydown`||e.type===`keyup`?e.keyCode:0},which:function(e){return e.type===`keypress`?Pn(e):e.type===`keydown`||e.type===`keyup`?e.keyCode:0}})),ir=Ln(w({},Gn,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0})),ar=Ln(w({},Rn,{submitter:0})),or=Ln(w({},Bn,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:nr})),sr=Ln(w({},Rn,{propertyName:0,elapsedTime:0,pseudoElement:0})),cr=Ln(w({},Gn,{deltaX:function(e){return`deltaX`in e?e.deltaX:`wheelDeltaX`in e?-e.wheelDeltaX:0},deltaY:function(e){return`deltaY`in e?e.deltaY:`wheelDeltaY`in e?-e.wheelDeltaY:`wheelDelta`in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0})),lr=Ln(w({},Rn,{newState:0,oldState:0,source:0})),ur=[9,13,27,32],dr=Dn&&`CompositionEvent`in window,fr=null;Dn&&`documentMode`in document&&(fr=document.documentMode);var pr=Dn&&`TextEvent`in window&&!fr,mr=Dn&&(!dr||fr&&8<fr&&11>=fr),hr=` `,gr=!1;function _r(e,t){switch(e){case`keyup`:return ur.indexOf(t.keyCode)!==-1;case`keydown`:return t.keyCode!==229;case`keypress`:case`mousedown`:case`focusout`:return!0;default:return!1}}function vr(e){return e=e.detail,typeof e==`object`&&`data`in e?e.data:null}var yr=!1;function br(e,t){switch(e){case`compositionend`:return vr(t);case`keypress`:return t.which===32?(gr=!0,hr):null;case`textInput`:return e=t.data,e===hr&&gr?null:e;default:return null}}function xr(e,t){if(yr)return e===`compositionend`||!dr&&_r(e,t)?(e=Nn(),Mn=jn=An=null,yr=!1,e):null;switch(e){case`paste`:return null;case`keypress`:if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case`compositionend`:return mr&&t.locale!==`ko`?null:t.data;default:return null}}var Sr={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function Cr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t===`input`?!!Sr[e.type]:t===`textarea`}function wr(e,t,n,r){xn?Sn?Sn.push(r):Sn=[r]:xn=r,t=Jf(t,`onChange`),0<t.length&&(n=new zn(`onChange`,`change`,null,n,r),e.push({event:n,listeners:t}))}var Tr=null,Er=null;function Dr(e){Vf(e,0)}function Or(e){if(en(Pt(e)))return e}function kr(e,t){if(e===`change`)return t}var Ar=!1;if(Dn){var jr;if(Dn){var Mr=`oninput`in document;if(!Mr){var Nr=document.createElement(`div`);Nr.setAttribute(`oninput`,`return;`),Mr=typeof Nr.oninput==`function`}jr=Mr}else jr=!1;Ar=jr&&(!document.documentMode||9<document.documentMode)}function Pr(){Tr&&(Tr.detachEvent(`onpropertychange`,Fr),Er=Tr=null)}function Fr(e){if(e.propertyName===`value`&&Or(Er)){var t=[];wr(t,Er,e,bn(e)),Tn(Dr,t)}}function Ir(e,t,n){e===`focusin`?(Pr(),Tr=t,Er=n,Tr.attachEvent(`onpropertychange`,Fr)):e===`focusout`&&Pr()}function Lr(e){if(e===`selectionchange`||e===`keyup`||e===`keydown`)return Or(Er)}function Rr(e,t){if(e===`click`)return Or(t)}function zr(e,t){if(e===`input`||e===`change`)return Or(t)}function Br(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var Vr=typeof Object.is==`function`?Object.is:Br;function Hr(e,t){if(Vr(e,t))return!0;if(typeof e!=`object`||!e||typeof t!=`object`||!t)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!Ie.call(t,i)||!Vr(e[i],t[i]))return!1}return!0}function Ur(e){if(e||=typeof document<`u`?document:void 0,e===void 0)return null;try{return e.activeElement||e.body}catch{return e.body}}function Wr(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Gr(e,t){var n=Wr(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}a:{for(;n;){if(n.nextSibling){n=n.nextSibling;break a}n=n.parentNode}n=void 0}n=Wr(n)}}function Kr(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Kr(e,t.parentNode):`contains`in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function qr(e){e=e!=null&&e.ownerDocument!=null&&e.ownerDocument.defaultView!=null?e.ownerDocument.defaultView:window;for(var t=Ur(e.document);t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href==`string`}catch{n=!1}if(n)e=t.contentWindow;else break;t=Ur(e.document)}return t}function Jr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t===`input`&&(e.type===`text`||e.type===`search`||e.type===`tel`||e.type===`url`||e.type===`password`)||t===`textarea`||e.contentEditable===`true`)}var Yr=Dn&&`documentMode`in document&&11>=document.documentMode,Xr=null,Zr=null,Qr=null,$r=!1;function ei(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;$r||Xr==null||Xr!==Ur(r)||(r=Xr,`selectionStart`in r&&Jr(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Qr&&Hr(Qr,r)||(Qr=r,r=Jf(Zr,`onSelect`),0<r.length&&(t=new zn(`onSelect`,`select`,null,t,n),e.push({event:t,listeners:r}),t.target=Xr)))}function ti(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n[`Webkit`+e]=`webkit`+t,n[`Moz`+e]=`moz`+t,n}var ni={animationend:ti(`Animation`,`AnimationEnd`),animationiteration:ti(`Animation`,`AnimationIteration`),animationstart:ti(`Animation`,`AnimationStart`),transitionrun:ti(`Transition`,`TransitionRun`),transitionstart:ti(`Transition`,`TransitionStart`),transitioncancel:ti(`Transition`,`TransitionCancel`),transitionend:ti(`Transition`,`TransitionEnd`)},ri={},ii={};Dn&&(ii=document.createElement(`div`).style,`AnimationEvent`in window||(delete ni.animationend.animation,delete ni.animationiteration.animation,delete ni.animationstart.animation),`TransitionEvent`in window||delete ni.transitionend.transition);function ai(e){if(ri[e])return ri[e];if(!ni[e])return e;var t=ni[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in ii)return ri[e]=t[n];return e}var oi=ai(`animationend`),si=ai(`animationiteration`),ci=ai(`animationstart`),li=ai(`transitionrun`),ui=ai(`transitionstart`),di=ai(`transitioncancel`),fi=ai(`transitionend`),pi=new Map,mi=`abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel`.split(` `);mi.push(`scrollEnd`);function hi(e,t){pi.set(e,t),Bt(t,[e])}var gi=0;function _i(e,t){if(e.name!=null&&e.name!==`auto`)return e.name;if(t.autoName!==null)return t.autoName;e=bd.identifierPrefix;var n=gi++;return e=`_`+e+`t_`+n.toString(32)+`_`,t.autoName=e}function vi(e){if(e==null||typeof e==`string`)return e;var t=null,n=Od;if(n!==null)for(var r=0;r<n.length;r++){var i=e[n[r]];if(i!=null){if(i===`none`)return`none`;t=t==null?i:t+(` `+i)}}return t??e.default}function yi(e,t){return e=vi(e),t=vi(t),t==null?e===`auto`?null:e:t===`auto`?null:t}var bi=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)},xi=[],Si=0,Ci=0;function wi(){for(var e=Si,t=Ci=Si=0;t<e;){var n=xi[t];xi[t++]=null;var r=xi[t];xi[t++]=null;var i=xi[t];xi[t++]=null;var a=xi[t];if(xi[t++]=null,r!==null&&i!==null){var o=r.pending;o===null?i.next=i:(i.next=o.next,o.next=i),r.pending=i}a!==0&&Oi(n,i,a)}}function Ti(e,t,n,r){xi[Si++]=e,xi[Si++]=t,xi[Si++]=n,xi[Si++]=r,Ci|=r,e.lanes|=r,e=e.alternate,e!==null&&(e.lanes|=r)}function Ei(e,t,n,r){return Ti(e,t,n,r),ki(e)}function Di(e,t){return Ti(e,null,null,t),ki(e)}function Oi(e,t,n){e.lanes|=n;var r=e.alternate;r!==null&&(r.lanes|=n);for(var i=!1,a=e.return;a!==null;)a.childLanes|=n,r=a.alternate,r!==null&&(r.childLanes|=n),a.tag===22&&(e=a.stateNode,e===null||e._visibility&1||(i=!0)),e=a,a=a.return;return e.tag===3?(a=e.stateNode,i&&t!==null&&(i=31-Qe(n),e=a.hiddenUpdates,r=e[i],r===null?e[i]=[t]:r.push(t),t.lane=n|536870912),a):null}function ki(e){if(50<kd)throw kd=0,Ad=null,Error(i(185));for(var t=e.return;t!==null;)e=t,t=e.return;return e.tag===3?e.stateNode:null}var Ai={};function ji(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Mi(e,t,n,r){return new ji(e,t,n,r)}function Ni(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Pi(e,t){var n=e.alternate;return n===null?(n=Mi(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&1206910976,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n.refCleanup=e.refCleanup,n}function Fi(e,t){e.flags&=1206910978;var n=e.alternate;return n===null?(e.childLanes=0,e.lanes=t,e.child=null,e.subtreeFlags=0,e.memoizedProps=null,e.memoizedState=null,e.updateQueue=null,e.dependencies=null,e.stateNode=null):(e.childLanes=n.childLanes,e.lanes=n.lanes,e.child=n.child,e.subtreeFlags=0,e.deletions=null,e.memoizedProps=n.memoizedProps,e.memoizedState=n.memoizedState,e.updateQueue=n.updateQueue,e.type=n.type,t=n.dependencies,e.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),e}function Ii(e,t,n,r,a,o){var s=0;if(r=e,typeof r==`function`)Ni(r)&&(s=1);else if(typeof r==`string`)s=qm(e,n,L.current)?26:e===`html`||e===`head`||e===`body`?27:5;else a:switch(r){case fe:return e=Mi(31,n,t,a),e.elementType=fe,e.lanes=o,e;case E:return Li(n.children,a,o,t);case ae:s=8,a|=24;break;case oe:return e=Mi(12,n,t,a|2),e.elementType=oe,e.lanes=o,e;case ce:return e=Mi(13,n,t,a),e.elementType=ce,e.lanes=o,e;case le:return e=Mi(19,n,t,a),e.elementType=le,e.lanes=o,e;case pe:case A:return e=a|32,e=Mi(30,n,t,e),e.elementType=A,e.lanes=o,e.stateNode={autoName:null,paired:null,clones:null,ref:null},e;default:if(typeof r==`object`&&r)switch(r.$$typeof){case O:s=10;break a;case D:s=9;break a;case se:s=11;break a;case ue:s=14;break a;case de:s=16,r=null;break a}s=29,n=Error(i(130,e===null?`null`:typeof e,``)),r=null}return t=Mi(s,n,t,a),t.elementType=e,t.type=r,t.lanes=o,t}function Li(e,t,n,r){return e=Mi(7,e,r,t),e.lanes=n,e}function Ri(e,t,n){return e=Mi(6,e,null,t),e.lanes=n,e}function zi(e){var t=Mi(18,null,null,0);return t.stateNode=e,t}function Bi(e,t,n){return t=Mi(4,e.children===null?[]:e.children,e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}var Vi=new WeakMap;function Hi(e,t){if(typeof e==`object`&&e){var n=Vi.get(e);return n===void 0?(t={value:e,source:t,stack:Fe(t)},Vi.set(e,t),t):n}return{value:e,source:t,stack:Fe(t)}}var Ui=[],Wi=0,Gi=null,Ki=0,qi=[],Ji=0,Yi=null,Xi=1,Zi=``;function Qi(e,t){Ui[Wi++]=Ki,Ui[Wi++]=Gi,Gi=e,Ki=t}function $i(e,t,n){qi[Ji++]=Xi,qi[Ji++]=Zi,qi[Ji++]=Yi,Yi=e;var r=Xi;e=Zi;var i=32-Qe(r)-1;r&=~(1<<i),n+=1;var a=32-Qe(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Xi=1<<32-Qe(t)+i|n<<i|r,Zi=a+e}else Xi=1<<a|n<<i|r,Zi=e}function ea(e){e.return!==null&&(Qi(e,1),$i(e,1,0))}function ta(e){for(;e===Gi;)Gi=Ui[--Wi],Ui[Wi]=null,Ki=Ui[--Wi],Ui[Wi]=null;for(;e===Yi;)Yi=qi[--Ji],qi[Ji]=null,Zi=qi[--Ji],qi[Ji]=null,Xi=qi[--Ji],qi[Ji]=null}function na(e,t){qi[Ji++]=Xi,qi[Ji++]=Zi,qi[Ji++]=Yi,Xi=t.id,Zi=t.overflow,Yi=e}var ra=null,B=null,V=!1,ia=null,aa=!1,oa=Error(i(519));function sa(e){throw pa(Hi(Error(i(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?`text`:`HTML`,``)),e)),oa}function ca(e){var t=e.stateNode,n=e.type,r=e.memoizedProps;switch(t[St]=e,t[Ct]=r,n){case`dialog`:Q(`cancel`,t),Q(`close`,t);break;case`iframe`:case`object`:case`embed`:Q(`load`,t);break;case`video`:case`audio`:for(n=0;n<zf.length;n++)Q(zf[n],t);break;case`source`:Q(`error`,t);break;case`img`:case`image`:case`link`:Q(`error`,t),Q(`load`,t);break;case`details`:Q(`toggle`,t);break;case`input`:Q(`invalid`,t),an(t,r.value,r.defaultValue,r.checked,r.defaultChecked,r.type,r.name,!0);break;case`select`:Q(`invalid`,t);break;case`textarea`:Q(`invalid`,t),ln(t,r.value,r.defaultValue,r.children)}n=r.children,typeof n!=`string`&&typeof n!=`number`&&typeof n!=`bigint`||t.textContent===``+n||!0===r.suppressHydrationWarning||ep(t.textContent,n)?(r.popover!=null&&(Q(`beforetoggle`,t),Q(`toggle`,t)),r.onScroll!=null&&Q(`scroll`,t),r.onScrollEnd!=null&&Q(`scrollend`,t),r.onClick!=null&&(t.onclick=vn),t=!0):t=!1,t||sa(e,!0)}function la(e){for(ra=e.return;ra;)switch(ra.tag){case 5:case 31:case 13:aa=!1;return;case 27:case 3:aa=!0;return;default:ra=ra.return}}function ua(e){if(e!==ra)return!1;if(!V)return la(e),V=!0,!1;var t=e.tag,n;if((n=t!==3&&t!==27)&&((n=t===5)&&(n=e.type,n=n===`form`||n===`button`||pp(e.type,e.memoizedProps)),n=!n),n&&B&&sa(e),la(e),t===13){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));B=dm(e)}else if(t===31){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));B=dm(e)}else t===27?(t=B,Sp(e.type)?(e=um,um=null,B=e):B=t):B=ra?lm(e.stateNode.nextSibling):null;return!0}function da(){B=ra=null,V=!1}function fa(){var e=ia;return e!==null&&(fd===null?fd=e:fd.push.apply(fd,e),ia=null),e}function pa(e){ia===null?ia=[e]:ia.push(e)}var ma=P(null),ha=null,ga=null;function _a(e,t,n){I(ma,t._currentValue),t._currentValue=n}function va(e){e._currentValue=ma.current,F(ma)}function ya(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)===t?r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t):(e.childLanes|=t,r!==null&&(r.childLanes|=t)),e===n)break;e=e.return}}function ba(e,t,n,r){var a=e.child;for(a!==null&&(a.return=e);a!==null;){var o=a.dependencies;if(o!==null){var s=a.child;o=o.firstContext;a:for(;o!==null;){var c=o;o=a;for(var l=0;l<t.length;l++)if(c.context===t[l]){o.lanes|=n,c=o.alternate,c!==null&&(c.lanes|=n),ya(o.return,n,e),r||(s=null);break a}o=c.next}}else if(a.tag===18){if(s=a.return,s===null)throw Error(i(341));s.lanes|=n,o=s.alternate,o!==null&&(o.lanes|=n),ya(s,n,e),s=null}else a.tag===13&&a.memoizedState!==null&&a.memoizedState.dehydrated===null?(a.lanes|=n,s=a.alternate,s!==null&&(s.lanes|=n),ya(a.return,n,e),s=a.child,s=s===null?null:s.sibling):s=a.child;if(s!==null)s.return=a;else for(s=a;s!==null;){if(s===e){s=null;break}if(a=s.sibling,a!==null){a.return=s.return,s=a;break}s=s.return}a=s}}function xa(e,t,n,r){e=null;for(var a=t,o=!1;a!==null;){if(!o){if(a.flags&524288)o=!0;else if(a.flags&262144)break}if(a.tag===10){var s=a.alternate;if(s===null)throw Error(i(387));if(s=s.memoizedProps,s!==null){var c=a.type;Vr(a.pendingProps.value,s.value)||(e===null?e=[c]:e.push(c))}}else if(a===we.current){if(s=a.alternate,s===null)throw Error(i(387));s.memoizedState.memoizedState!==a.memoizedState.memoizedState&&(e===null?e=[sh]:e.push(sh))}a=a.return}return e!==null&&ba(t,e,n,r),t.flags|=262144,e!==null}function Sa(e){for(e=e.firstContext;e!==null;){if(!Vr(e.context._currentValue,e.memoizedValue))return!0;e=e.next}return!1}function Ca(e){ha=e,ga=null,e=e.dependencies,e!==null&&(e.firstContext=null)}function wa(e){return Ea(ha,e)}function Ta(e,t){return ha===null&&Ca(e),Ea(e,t)}function Ea(e,t){var n=t._currentValue;if(t={context:t,memoizedValue:n,next:null},ga===null){if(e===null)throw Error(i(308));ga=t,e.dependencies={lanes:0,firstContext:t},e.flags|=524288}else ga=ga.next=t;return n}var Da=typeof AbortController<`u`?AbortController:function(){var e=[],t=this.signal={aborted:!1,addEventListener:function(t,n){e.push(n)}};this.abort=function(){t.aborted=!0,e.forEach(function(e){return e()})}},Oa=t.unstable_scheduleCallback,ka=t.unstable_NormalPriority,Aa={$$typeof:O,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function ja(){return{controller:new Da,data:new Map,refCount:0}}function Ma(e){e.refCount--,e.refCount===0&&Oa(ka,function(){e.controller.abort()})}function Na(e,t){if(e.pendingLanes&4194048){var n=e.transitionTypes;for(n===null&&(n=e.transitionTypes=[]),e=0;e<t.length;e++){var r=t[e];n.indexOf(r)===-1&&n.push(r)}}}var Pa=null;function Fa(e){var t=e.transitionTypes;return e.transitionTypes=null,t}var Ia=null,La=0,Ra=0,za=null;function Ba(e,t){if(Ia===null){var n=Ia=[];La=0,Ra=Pf(),za={status:`pending`,value:void 0,then:function(e){n.push(e)}}}return La++,t.then(Va,Va),t}function Va(){if(--La===0&&(Pa=null,Ia!==null)){za!==null&&(za.status=`fulfilled`);var e=Ia;Ia=null,Ra=0,za=null;for(var t=0;t<e.length;t++)(0,e[t])()}}function Ha(e,t){var n=[],r={status:`pending`,value:null,reason:null,then:function(e){n.push(e)}};return e.then(function(){r.status=`fulfilled`,r.value=t;for(var e=0;e<n.length;e++)(0,n[e])(t)},function(e){for(r.status=`rejected`,r.reason=e,e=0;e<n.length;e++)(0,n[e])(void 0)}),r}var Ua=j.S;j.S=function(e,t){if(hd=Ve(),typeof t==`object`&&t&&typeof t.then==`function`&&Ba(e,t),Pa!==null)for(var n=bf;n!==null;)Na(n,Pa),n=n.next;if(n=e.types,n!==null){for(var r=bf;r!==null;)Na(r,n),r=r.next;if(Ra!==0){r=Pa,r===null&&(r=Pa=[]);for(var i=0;i<n.length;i++){var a=n[i];r.indexOf(a)===-1&&r.push(a)}}}Ua!==null&&Ua(e,t)};var Wa=P(null);function Ga(){var e=Wa.current;return e===null?q.pooledCache:e}function Ka(e,t){t===null?I(Wa,Wa.current):I(Wa,t.pool)}function qa(){var e=Ga();return e===null?null:{parent:Aa._currentValue,pool:e}}var Ja=Error(i(460)),Ya=Error(i(474)),Xa=Error(i(542)),Za={then:function(){}};function Qa(e){return e=e.status,e===`fulfilled`||e===`rejected`}function $a(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(vn,vn),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,ro(e),e===void 0&&!(`reason`in t)?Error(i(600)):e;default:if(typeof t.status==`string`)t.then(vn,vn);else{if(e=q,e!==null&&100<e.shellSuspendCounter)throw Error(i(482));e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})}switch(t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,ro(e),e}throw to=t,Ja}}function eo(e){try{var t=e._init;return t(e._payload)}catch(e){throw typeof e==`object`&&e&&typeof e.then==`function`?(to=e,Ja):e}}var to=null;function no(){if(to===null)throw Error(i(459));var e=to;return to=null,e}function ro(e){if(e===Ja||e===Xa)throw Error(i(483))}var io=null,ao=0;function oo(e){var t=ao;return ao+=1,io===null&&(io=[]),$a(io,e,t)}function so(e,t){t=t.props.ref,e.ref=t===void 0?null:t}function co(e,t){throw t.$$typeof===re?Error(i(525)):(e=Object.prototype.toString.call(t),Error(i(31,e===`[object Object]`?`object with keys {`+Object.keys(t).join(`, `)+`}`:e)))}function lo(e){function t(t,n){if(e){var r=t.deletions;r===null?(t.deletions=[n],t.flags|=16):r.push(n)}}function n(n,r){if(!e)return null;for(;r!==null;)t(n,r),r=r.sibling;return null}function r(e){for(var t=new Map;e!==null;)e.key===null?t.set(e.index,e):t.set(e.key,e),e=e.sibling;return t}function a(e,t){return e=Pi(e,t),e.index=0,e.sibling=null,e}function o(t,n,r){return t.index=r,e?(r=t.alternate,r===null?(t.flags|=134217730,n):(r=r.index,r<n?(t.flags|=2,n):r)):(t.flags|=1048576,n)}function s(t){return e&&t.alternate===null&&(t.flags|=134217730),t}function c(e,t,n,r){return t===null||t.tag!==6?(t=Ri(n,e.mode,r),t.return=e,t):(t=a(t,n),t.return=e,t)}function l(e,t,n,r){var i=n.type;return i===E?(e=d(e,t,n.props.children,r,n.key),so(e,n),e):t!==null&&(t.elementType===i||typeof i==`object`&&i&&i.$$typeof===de&&eo(i)===t.type)?(t=a(t,n.props),so(t,n),t.return=e,t):(t=Ii(n.type,n.key,n.props,null,e.mode,r),so(t,n),t.return=e,t)}function u(e,t,n,r){return t===null||t.tag!==4||t.stateNode.containerInfo!==n.containerInfo||t.stateNode.implementation!==n.implementation?(t=Bi(n,e.mode,r),t.return=e,t):(t=a(t,n.children||[]),t.return=e,t)}function d(e,t,n,r,i){return t===null||t.tag!==7?(t=Li(n,e.mode,r,i),t.return=e,t):(t=a(t,n),t.return=e,t)}function f(e,t,n){if(typeof t==`string`&&t!==``||typeof t==`number`||typeof t==`bigint`)return t=Ri(``+t,e.mode,n),t.return=e,t;if(typeof t==`object`&&t){switch(t.$$typeof){case T:return n=Ii(t.type,t.key,t.props,null,e.mode,n),so(n,t),n.return=e,n;case ie:return t=Bi(t,e.mode,n),t.return=e,t;case de:return t=eo(t),f(e,t,n)}if(ye(t)||ge(t))return t=Li(t,e.mode,n,null),t.return=e,t;if(typeof t.then==`function`)return f(e,oo(t),n);if(t.$$typeof===O)return f(e,Ta(e,t),n);co(e,t)}return null}function p(e,t,n,r){var i=t===null?null:t.key;if(typeof n==`string`&&n!==``||typeof n==`number`||typeof n==`bigint`)return i===null?c(e,t,``+n,r):null;if(typeof n==`object`&&n){switch(n.$$typeof){case T:return n.key===i?l(e,t,n,r):null;case ie:return n.key===i?u(e,t,n,r):null;case de:return n=eo(n),p(e,t,n,r)}if(ye(n)||ge(n))return i===null?d(e,t,n,r,null):null;if(typeof n.then==`function`)return p(e,t,oo(n),r);if(n.$$typeof===O)return p(e,t,Ta(e,n),r);co(e,n)}return null}function m(e,t,n,r,i){if(typeof r==`string`&&r!==``||typeof r==`number`||typeof r==`bigint`)return e=e.get(n)||null,c(t,e,``+r,i);if(typeof r==`object`&&r){switch(r.$$typeof){case T:return e=e.get(r.key===null?n:r.key)||null,l(t,e,r,i);case ie:return e=e.get(r.key===null?n:r.key)||null,u(t,e,r,i);case de:return r=eo(r),m(e,t,n,r,i)}if(ye(r)||ge(r))return e=e.get(n)||null,d(t,e,r,i,null);if(typeof r.then==`function`)return m(e,t,n,oo(r),i);if(r.$$typeof===O)return m(e,t,n,Ta(t,r),i);co(t,r)}return null}function h(i,a,s,c){for(var l=null,u=null,d=a,h=a=0,g=null;d!==null&&h<s.length;h++){d.index>h?(g=d,d=null):g=d.sibling;var _=p(i,d,s[h],c);if(_===null){d===null&&(d=g);break}e&&d&&_.alternate===null&&t(i,d),a=o(_,a,h),u===null?l=_:u.sibling=_,u=_,d=g}if(h===s.length)return n(i,d),V&&Qi(i,h),l;if(d===null){for(;h<s.length;h++)d=f(i,s[h],c),d!==null&&(a=o(d,a,h),u===null?l=d:u.sibling=d,u=d);return V&&Qi(i,h),l}for(d=r(d);h<s.length;h++)g=m(d,i,h,s[h],c),g!==null&&(e&&(_=g.alternate,_!==null&&d.delete(_.key===null?h:_.key)),a=o(g,a,h),u===null?l=g:u.sibling=g,u=g);return e&&d.forEach(function(e){return t(i,e)}),V&&Qi(i,h),l}function g(a,s,c,l){if(c==null)throw Error(i(151));for(var u=null,d=null,h=s,g=s=0,_=null,v=c.next();h!==null&&!v.done;g++,v=c.next()){h.index>g?(_=h,h=null):_=h.sibling;var y=p(a,h,v.value,l);if(y===null){h===null&&(h=_);break}e&&h&&y.alternate===null&&t(a,h),s=o(y,s,g),d===null?u=y:d.sibling=y,d=y,h=_}if(v.done)return n(a,h),V&&Qi(a,g),u;if(h===null){for(;!v.done;g++,v=c.next())v=f(a,v.value,l),v!==null&&(s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return V&&Qi(a,g),u}for(h=r(h);!v.done;g++,v=c.next())v=m(h,a,g,v.value,l),v!==null&&(e&&(_=v.alternate,_!==null&&h.delete(_.key===null?g:_.key)),s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return e&&h.forEach(function(e){return t(a,e)}),V&&Qi(a,g),u}function _(e,r,o,c){if(typeof o==`object`&&o&&o.type===E&&o.key===null&&o.props.ref===void 0&&(o=o.props.children),typeof o==`object`&&o){switch(o.$$typeof){case T:a:{for(var l=o.key;r!==null;){if(r.key===l){if(l=o.type,l===E){if(r.tag===7){n(e,r.sibling),c=a(r,o.props.children),so(c,o),c.return=e,e=c;break a}}else if(r.elementType===l||typeof l==`object`&&l&&l.$$typeof===de&&eo(l)===r.type){n(e,r.sibling),c=a(r,o.props),so(c,o),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}o.type===E?(c=Li(o.props.children,e.mode,c,o.key),so(c,o),c.return=e,e=c):(c=Ii(o.type,o.key,o.props,null,e.mode,c),so(c,o),c.return=e,e=c)}return s(e);case ie:a:{for(l=o.key;r!==null;){if(r.key===l){if(r.tag===4&&r.stateNode.containerInfo===o.containerInfo&&r.stateNode.implementation===o.implementation){n(e,r.sibling),c=a(r,o.children||[]),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}c=Bi(o,e.mode,c),c.return=e,e=c}return s(e);case de:return o=eo(o),_(e,r,o,c)}if(ye(o))return h(e,r,o,c);if(ge(o)){if(l=ge(o),typeof l!=`function`)throw Error(i(150));return o=l.call(o),g(e,r,o,c)}if(typeof o.then==`function`)return _(e,r,oo(o),c);if(o.$$typeof===O)return _(e,r,Ta(e,o),c);co(e,o)}return typeof o==`string`&&o!==``||typeof o==`number`||typeof o==`bigint`?(o=``+o,r!==null&&r.tag===6?(n(e,r.sibling),c=a(r,o),c.return=e,e=c):(n(e,r),c=Ri(o,e.mode,c),c.return=e,e=c),s(e)):n(e,r)}return function(e,t,n,r){try{ao=0;var i=_(e,t,n,r);return io=null,i}catch(t){if(t===Ja||t===Xa)throw t;var a=Mi(29,t,null,e.mode);return a.lanes=r,a.return=e,a}}}var uo=lo(!0),fo=lo(!1),po=!1;function mo(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function ho(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,callbacks:null})}function go(e){return{lane:e,tag:0,payload:null,callback:null,next:null}}function _o(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,K&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,t=ki(e),Oi(e,null,n),t}return Ti(e,r,t,n),ki(e)}function vo(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,n&4194048)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ht(e,n)}}function yo(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,callbacks:r.callbacks},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}var bo=!1;function xo(){if(bo){var e=za;if(e!==null)throw e}}function So(e,t,n,r){bo=!1;var i=e.updateQueue;po=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var c=s,l=c.next;c.next=null,o===null?a=l:o.next=l,o=c;var u=e.alternate;u!==null&&(u=u.updateQueue,s=u.lastBaseUpdate,s!==o&&(s===null?u.firstBaseUpdate=l:s.next=l,u.lastBaseUpdate=c))}if(a!==null){var d=i.baseState;o=0,u=l=c=null,s=a;do{var f=s.lane&-536870913,p=f!==s.lane;if(p?(Y&f)===f:(r&f)===f){f!==0&&f===Ra&&(bo=!0),u!==null&&(u=u.next={lane:0,tag:s.tag,payload:s.payload,callback:null,next:null});a:{var m=e,h=s;f=t;var g=n;switch(h.tag){case 1:if(m=h.payload,typeof m==`function`){d=m.call(g,d,f);break a}d=m;break a;case 3:m.flags=m.flags&-65537|128;case 0:if(m=h.payload,f=typeof m==`function`?m.call(g,d,f):m,f==null)break a;d=w({},d,f);break a;case 2:po=!0}}f=s.callback,f!==null&&(e.flags|=64,p&&(e.flags|=8192),p=i.callbacks,p===null?i.callbacks=[f]:p.push(f))}else p={lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},u===null?(l=u=p,c=d):u=u.next=p,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(1);u===null&&(c=d),i.baseState=c,i.firstBaseUpdate=l,i.lastBaseUpdate=u,a===null&&(i.shared.lanes=0),od|=o,e.lanes=o,e.memoizedState=d}}function Co(e,t){if(typeof e!=`function`)throw Error(i(191,e));e.call(t)}function wo(e,t){var n=e.callbacks;if(n!==null)for(e.callbacks=null,e=0;e<n.length;e++)Co(n[e],t)}var To=P(null),Eo=P(0);function Do(e,t){e=id,I(Eo,e),I(To,t),id=e|t.baseLanes}function Oo(){I(Eo,id),I(To,To.current)}function ko(){id=Eo.current,F(To),F(Eo)}var Ao=P(null),jo=null;function Mo(e){var t=e.alternate;I(Lo,Lo.current&1),I(Ao,e),jo===null&&(t===null||To.current!==null||t.memoizedState!==null)&&(jo=e)}function No(e){I(Lo,Lo.current),I(Ao,e),jo===null&&(jo=e)}function Po(e){e.tag===22?(I(Lo,Lo.current),I(Ao,e),jo===null&&(jo=e)):Fo()}function Fo(){I(Lo,Lo.current),I(Ao,Ao.current)}function Io(e){F(Ao),jo===e&&(jo=null),F(Lo)}var Lo=P(0);function Ro(e,t){I(Ao,Ao.current),I(Lo,t)}function zo(e){F(Lo),F(Ao),jo===e&&(jo=null)}function Bo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||om(n)||sm(n)))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==`independent`){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Vo=0,H=null,U=null,Ho=null,Uo=!1,Wo=!1,Go=!1,Ko=0,qo=0,Jo=null,Yo=0;function Xo(){throw Error(i(321))}function Zo(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Vr(e[n],t[n]))return!1;return!0}function Qo(e,t,n,r,i,a){return Vo=a,H=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,j.H=e===null||e.memoizedState===null?hc:gc,Go=!1,a=n(r,i),Go=!1,Wo&&(a=es(t,n,r,i)),$o(e),a}function $o(e){j.H=mc;var t=U!==null&&U.next!==null;if(Vo=0,Ho=U=H=null,Uo=!1,qo=0,Jo=null,t)throw Error(i(300));e===null||Nc||(e=e.dependencies,e!==null&&Sa(e)&&(Nc=!0))}function es(e,t,n,r){H=e;var a=0;do{if(Wo&&(Jo=null),qo=0,Wo=!1,25<=a)throw Error(i(301));if(a+=1,Ho=U=null,e.updateQueue!=null){var o=e.updateQueue;o.lastEffect=null,o.events=null,o.stores=null,o.memoCache!=null&&(o.memoCache.index=0)}j.H=_c,o=t(n,r)}while(Wo);return o}function ts(){var e=j.H,t=e.useState()[0];return t=typeof t.then==`function`?cs(t):t,e=e.useState()[0],(U===null?null:U.memoizedState)!==e&&(H.flags|=1024),t}function ns(){var e=Ko!==0;return Ko=0,e}function rs(e,t,n){t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~n}function is(e){if(Uo){for(e=e.memoizedState;e!==null;){var t=e.queue;t!==null&&(t.pending=null),e=e.next}Uo=!1}Vo=0,Ho=U=H=null,Wo=!1,qo=Ko=0,Jo=null}function as(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return Ho===null?H.memoizedState=Ho=e:Ho=Ho.next=e,Ho}function os(){if(U===null){var e=H.alternate;e=e===null?null:e.memoizedState}else e=U.next;var t=Ho===null?H.memoizedState:Ho.next;if(t!==null)Ho=t,U=e;else{if(e===null)throw H.alternate===null?Error(i(467)):Error(i(310));U=e,e={memoizedState:U.memoizedState,baseState:U.baseState,baseQueue:U.baseQueue,queue:U.queue,next:null},Ho===null?H.memoizedState=Ho=e:Ho=Ho.next=e}return Ho}function ss(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function cs(e){var t=qo;return qo+=1,Jo===null&&(Jo=[]),e=$a(Jo,e,t),t=H,(Ho===null?t.memoizedState:Ho.next)===null&&(t=t.alternate,j.H=t===null||t.memoizedState===null?hc:gc),e}function ls(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return cs(e);if(e.$$typeof===me)return;if(e.$$typeof===O)return wa(e)}throw Error(i(438,String(e)))}function us(e){var t=null,n=H.updateQueue;if(n!==null&&(t=n.memoCache),t==null){var r=H.alternate;r!==null&&(r=r.updateQueue,r!==null&&(r=r.memoCache,r!=null&&(t={data:r.data.map(function(e){return e.slice()}),index:0})))}if(t??={data:[],index:0},n===null&&(n=ss(),H.updateQueue=n),n.memoCache=t,n=t.data[t.index],n===void 0)for(n=t.data[t.index]=Array(e),r=0;r<e;r++)n[r]=k;return t.index++,n}function ds(e,t){return typeof t==`function`?t(e):t}function fs(e){return ps(os(),U,e)}function ps(e,t,n){var r=e.queue;if(r===null)throw Error(i(311));r.lastRenderedReducer=n;var a=e.baseQueue,o=r.pending;if(o!==null){if(a!==null){var s=a.next;a.next=o.next,o.next=s}t.baseQueue=a=o,r.pending=null}if(o=e.baseState,a===null)e.memoizedState=o;else{t=a.next;var c=s=null,l=null,u=t,d=!1;do{var f=u.lane&-536870913;if(f===u.lane?(Vo&f)===f:(Y&f)===f){var p=u.revertLane;if(p===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),f===Ra&&(d=!0);else if((Vo&p)===p){u=u.next,p===Ra&&(d=!0);continue}else f={lane:0,revertLane:u.revertLane,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=f,s=o):l=l.next=f,H.lanes|=p,od|=p;f=u.action,Go&&n(o,f),o=u.hasEagerState?u.eagerState:n(o,f)}else p={lane:f,revertLane:u.revertLane,gesture:u.gesture,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=p,s=o):l=l.next=p,H.lanes|=f,od|=f;u=u.next}while(u!==null&&u!==t);if(l===null?s=o:l.next=c,!Vr(o,e.memoizedState)&&(Nc=!0,d&&(n=za,n!==null)))throw n;e.memoizedState=o,e.baseState=s,e.baseQueue=l,r.lastRenderedState=o}return a===null&&(r.lanes=0),[e.memoizedState,r.dispatch]}function ms(e){var t=os(),n=t.queue;if(n===null)throw Error(i(311));n.lastRenderedReducer=e;var r=n.dispatch,a=n.pending,o=t.memoizedState;if(a!==null){n.pending=null;var s=a=a.next;do o=e(o,s.action),s=s.next;while(s!==a);Vr(o,t.memoizedState)||(Nc=!0),t.memoizedState=o,t.baseQueue===null&&(t.baseState=o),n.lastRenderedState=o}return[o,r]}function hs(e,t,n){var r=H,a=os(),o=V;if(o){if(n===void 0)throw Error(i(407));n=n()}else n=t();var s=!Vr((U||a).memoizedState,n);if(s&&(a.memoizedState=n,Nc=!0),a=a.queue,Bs(vs.bind(null,r,a,e),[e]),e=a.getSnapshot!==t||s||Ho!==null&&!!(Ho.memoizedState.tag&1),Fs(e?9:8,{destroy:void 0},_s.bind(null,r,a,n,t),null),e){if(r.flags|=2048,q===null)throw Error(i(349));o||Vo&127||gs(r,t,n)}return n}function gs(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=H.updateQueue,t===null?(t=ss(),H.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function _s(e,t,n,r){t.value=n,t.getSnapshot=r,ys(t)&&bs(e)}function vs(e,t,n){return n(function(){ys(t)&&bs(e)})}function ys(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Vr(e,n)}catch{return!0}}function bs(e){var t=Di(e,2);t!==null&&Pd(t,e,2)}function xs(e){var t=as();if(typeof e==`function`){var n=e;if(e=n(),Go){Ze(!0);try{n()}finally{Ze(!1)}}}return t.memoizedState=t.baseState=e,t.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:ds,lastRenderedState:e},t}function Ss(e,t,n,r){return e.baseState=n,ps(e,U,typeof r==`function`?r:ds)}function Cs(e,t,n,r,a){if(dc(e))throw Error(i(485));if(e=t.action,e!==null){var o={payload:a,action:e,next:null,isTransition:!0,status:`pending`,value:null,reason:null,listeners:[],then:function(e){o.listeners.push(e)}};j.T===null?o.isTransition=!1:n(!0),r(o),n=t.pending,n===null?(o.next=t.pending=o,ws(t,o)):(o.next=n.next,t.pending=n.next=o)}}function ws(e,t){var n=t.action,r=t.payload,i=e.state;if(t.isTransition){var a=j.T,o={};o.types=a===null?null:a.types,j.T=o;try{var s=n(i,r),c=j.S;c!==null&&c(o,s),Ts(e,t,s)}catch(n){Ds(e,t,n)}finally{a!==null&&o.types!==null&&(a.types=o.types),j.T=a}}else try{a=n(i,r),Ts(e,t,a)}catch(n){Ds(e,t,n)}}function Ts(e,t,n){typeof n==`object`&&n&&typeof n.then==`function`?n.then(function(n){Es(e,t,n)},function(n){return Ds(e,t,n)}):Es(e,t,n)}function Es(e,t,n){t.status=`fulfilled`,t.value=n,Os(t),e.state=n,t=e.pending,t!==null&&(n=t.next,n===t?e.pending=null:(n=n.next,t.next=n,ws(e,n)))}function Ds(e,t,n){var r=e.pending;if(e.pending=null,r!==null){r=r.next;do t.status=`rejected`,t.reason=n,Os(t),t=t.next;while(t!==r)}e.action=null}function Os(e){e=e.listeners;for(var t=0;t<e.length;t++)(0,e[t])()}function ks(e,t){return t}function As(e,t){if(V){var n=q.formState;if(n!==null){a:{var r=H;if(V){if(B){b:{for(var i=B,a=aa;i.nodeType!==8;){if(!a){i=null;break b}if(i=lm(i.nextSibling),i===null){i=null;break b}}a=i.data,i=a===`F!`||a===`F`?i:null}if(i){B=lm(i.nextSibling),r=i.data===`F!`;break a}}sa(r)}r=!1}r&&(t=n[0])}}return n=as(),n.memoizedState=n.baseState=t,r={pending:null,lanes:0,dispatch:null,lastRenderedReducer:ks,lastRenderedState:t},n.queue=r,n=cc.bind(null,H,r),r.dispatch=n,r=xs(!1),a=uc.bind(null,H,!1,r.queue),r=as(),i={state:t,dispatch:null,action:e,pending:null},r.queue=i,n=Cs.bind(null,H,i,a,n),i.dispatch=n,r.memoizedState=e,[t,n,!1]}function js(e){return Ms(os(),U,e)}function Ms(e,t,n){if(t=ps(e,t,ks)[0],e=fs(ds)[0],typeof t==`object`&&t&&typeof t.then==`function`)try{var r=cs(t)}catch(e){throw e===Ja?Xa:e}else r=t;t=os();var i=t.queue,a=i.dispatch;return n!==t.memoizedState&&(H.flags|=2048,Fs(9,{destroy:void 0},Ns.bind(null,i,n),null)),[r,a,e]}function Ns(e,t){e.action=t}function Ps(e){var t=os(),n=U;if(n!==null)return Ms(t,n,e);os(),t=t.memoizedState,n=os();var r=n.queue.dispatch;return n.memoizedState=e,[t,r,!1]}function Fs(e,t,n,r){return e={tag:e,create:n,deps:r,inst:t,next:null},t=H.updateQueue,t===null&&(t=ss(),H.updateQueue=t),n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e),e}function Is(){return os().memoizedState}function Ls(e,t,n,r){var i=as();H.flags|=e,i.memoizedState=Fs(1|t,{destroy:void 0},n,r===void 0?null:r)}function Rs(e,t,n,r){var i=os();r=r===void 0?null:r;var a=i.memoizedState.inst;U!==null&&r!==null&&Zo(r,U.memoizedState.deps)?i.memoizedState=Fs(t,a,n,r):(H.flags|=e,i.memoizedState=Fs(1|t,a,n,r))}function zs(e,t){Ls(8390656,8,e,t)}function Bs(e,t){Rs(2048,8,e,t)}function Vs(e){H.flags|=4;var t=H.updateQueue;if(t===null)t=ss(),H.updateQueue=t,t.events=[e];else{var n=t.events;n===null?t.events=[e]:n.push(e)}}function Hs(e){var t=os().memoizedState;return Vs({ref:t,nextImpl:e}),function(){if(K&2)throw Error(i(440));return t.impl.apply(void 0,arguments)}}function Us(e,t){return Rs(4,2,e,t)}function Ws(e,t){return Rs(4,4,e,t)}function Gs(e,t){if(typeof t==`function`){e=e();var n=t(e);return function(){typeof n==`function`?n():t(null)}}if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Ks(e,t,n){n=n==null?null:n.concat([e]),Rs(4,4,Gs.bind(null,t,e),n)}function qs(){}function Js(e,t){var n=os();t=t===void 0?null:t;var r=n.memoizedState;return t!==null&&Zo(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Ys(e,t){var n=os();t=t===void 0?null:t;var r=n.memoizedState;if(t!==null&&Zo(t,r[1]))return r[0];if(r=e(),Go){Ze(!0);try{e()}finally{Ze(!1)}}return n.memoizedState=[r,t],r}function Xs(e,t,n){return n===void 0||Vo&1073741824&&!(Y&261930)?e.memoizedState=t:(e.memoizedState=n,e=Md(),H.lanes|=e,od|=e,n)}function Zs(e,t,n,r){return Vr(n,t)?n:To.current===null?!(Vo&106)||Vo&1073741824&&!(Y&261930)?(Nc=!0,e.memoizedState=n):(e=Md(),H.lanes|=e,od|=e,t):(e=Xs(e,n,r),Vr(e,t)||(Nc=!0),e)}function Qs(e,t,n,r,i){var a=M.p;M.p=a!==0&&8>a?a:8;var o=j.T,s={};s.types=o===null?null:o.types,j.T=s,uc(e,!1,t,n);try{var c=i(),l=j.S;l!==null&&l(s,c),typeof c==`object`&&c&&typeof c.then==`function`?lc(e,t,Ha(c,r),jd(e)):lc(e,t,r,jd(e))}catch(n){lc(e,t,{then:function(){},status:`rejected`,reason:n},jd())}finally{M.p=a,o!==null&&s.types!==null&&(o.types=s.types),j.T=o}}function $s(){}function ec(e,t,n,r){if(e.tag!==5)throw Error(i(476));var a=tc(e).queue;Qs(e,a,t,be,n===null?$s:function(){return nc(e),n(r)})}function tc(e){var t=e.memoizedState;if(t!==null)return t;t={memoizedState:be,baseState:be,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ds,lastRenderedState:be},next:null};var n={};return t.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ds,lastRenderedState:n},next:null},e.memoizedState=t,e=e.alternate,e!==null&&(e.memoizedState=t),t}function nc(e){var t=tc(e);t.next===null&&(t=e.alternate.memoizedState),lc(e,t.next.queue,{},jd())}function rc(){return wa(sh)}function ic(){return os().memoizedState}function ac(){return os().memoizedState}function oc(e){for(var t=e.return;t!==null;){switch(t.tag){case 24:case 3:var n=jd();e=go(n);var r=_o(t,e,n);r!==null&&(Pd(r,t,n),vo(r,t,n)),t={cache:ja()},e.payload=t;return}t=t.return}}function sc(e,t,n){var r=jd();n={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},dc(e)?fc(t,n):(n=Ei(e,t,n,r),n!==null&&(Pd(n,e,r),pc(n,t,r)))}function cc(e,t,n){lc(e,t,n,jd())}function lc(e,t,n,r){var i={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(dc(e))fc(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Vr(s,o))return Ti(e,t,i,0),q===null&&wi(),!1}catch{}if(n=Ei(e,t,i,r),n!==null)return Pd(n,e,r),pc(n,t,r),!0}return!1}function uc(e,t,n,r){if(r={lane:2,revertLane:Pf(),gesture:null,action:r,hasEagerState:!1,eagerState:null,next:null},dc(e)){if(t)throw Error(i(479))}else t=Ei(e,n,r,2),t!==null&&Pd(t,e,2)}function dc(e){var t=e.alternate;return e===H||t!==null&&t===H}function fc(e,t){Wo=Uo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function pc(e,t,n){if(n&4194048){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ht(e,n)}}var mc={readContext:wa,use:ls,useCallback:Xo,useContext:Xo,useEffect:Xo,useImperativeHandle:Xo,useLayoutEffect:Xo,useInsertionEffect:Xo,useMemo:Xo,useReducer:Xo,useRef:Xo,useState:Xo,useDebugValue:Xo,useDeferredValue:Xo,useTransition:Xo,useSyncExternalStore:Xo,useId:Xo,useHostTransitionStatus:Xo,useFormState:Xo,useActionState:Xo,useOptimistic:Xo,useMemoCache:Xo,useCacheRefresh:Xo,useEffectEvent:Xo},hc={readContext:wa,use:ls,useCallback:function(e,t){return as().memoizedState=[e,t===void 0?null:t],e},useContext:wa,useEffect:zs,useImperativeHandle:function(e,t,n){n=n==null?null:n.concat([e]),Ls(4194308,4,Gs.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Ls(4194308,4,e,t)},useInsertionEffect:function(e,t){Ls(4,2,e,t)},useMemo:function(e,t){var n=as();t=t===void 0?null:t;var r=e();if(Go){Ze(!0);try{e()}finally{Ze(!1)}}return n.memoizedState=[r,t],r},useReducer:function(e,t,n){var r=as();if(n!==void 0){var i=n(t);if(Go){Ze(!0);try{n(t)}finally{Ze(!1)}}}else i=t;return r.memoizedState=r.baseState=i,e={pending:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:i},r.queue=e,e=e.dispatch=sc.bind(null,H,e),[r.memoizedState,e]},useRef:function(e){var t=as();return e={current:e},t.memoizedState=e},useState:function(e){e=xs(e);var t=e.queue,n=cc.bind(null,H,t);return t.dispatch=n,[e.memoizedState,n]},useDebugValue:qs,useDeferredValue:function(e,t){return Xs(as(),e,t)},useTransition:function(){var e=xs(!1);return e=Qs.bind(null,H,e.queue,!0,!1),as().memoizedState=e,[!1,e]},useSyncExternalStore:function(e,t,n){var r=H,a=as();if(V){if(n===void 0)throw Error(i(407));n=n()}else{if(n=t(),q===null)throw Error(i(349));Y&127||gs(r,t,n)}a.memoizedState=n;var o={value:n,getSnapshot:t};return a.queue=o,zs(vs.bind(null,r,o,e),[e]),r.flags|=2048,Fs(9,{destroy:void 0},_s.bind(null,r,o,n,t),null),n},useId:function(){var e=as(),t=q.identifierPrefix;if(V){var n=Zi,r=Xi;n=(r&~(1<<32-Qe(r)-1)).toString(32)+n,t=`_`+t+`R_`+n,n=Ko++,0<n&&(t+=`H`+n.toString(32)),t+=`_`}else n=Yo++,t=`_`+t+`r_`+n.toString(32)+`_`;return e.memoizedState=t},useHostTransitionStatus:rc,useFormState:As,useActionState:As,useOptimistic:function(e){var t=as();t.memoizedState=t.baseState=e;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return t.queue=n,t=uc.bind(null,H,!0,n),n.dispatch=t,[e,t]},useMemoCache:us,useCacheRefresh:function(){return as().memoizedState=oc.bind(null,H)},useEffectEvent:function(e){var t=as(),n={impl:e};return t.memoizedState=n,function(){if(K&2)throw Error(i(440));return n.impl.apply(void 0,arguments)}}},gc={readContext:wa,use:ls,useCallback:Js,useContext:wa,useEffect:Bs,useImperativeHandle:Ks,useInsertionEffect:Us,useLayoutEffect:Ws,useMemo:Ys,useReducer:fs,useRef:Is,useState:function(){return fs(ds)},useDebugValue:qs,useDeferredValue:function(e,t){return Zs(os(),U.memoizedState,e,t)},useTransition:function(){var e=fs(ds)[0],t=os().memoizedState;return[typeof e==`boolean`?e:cs(e),t]},useSyncExternalStore:hs,useId:ic,useHostTransitionStatus:rc,useFormState:js,useActionState:js,useOptimistic:function(e,t){return Ss(os(),U,e,t)},useMemoCache:us,useCacheRefresh:ac,useEffectEvent:Hs},_c={readContext:wa,use:ls,useCallback:Js,useContext:wa,useEffect:Bs,useImperativeHandle:Ks,useInsertionEffect:Us,useLayoutEffect:Ws,useMemo:Ys,useReducer:ms,useRef:Is,useState:function(){return ms(ds)},useDebugValue:qs,useDeferredValue:function(e,t){var n=os();return U===null?Xs(n,e,t):Zs(n,U.memoizedState,e,t)},useTransition:function(){var e=ms(ds)[0],t=os().memoizedState;return[typeof e==`boolean`?e:cs(e),t]},useSyncExternalStore:hs,useId:ic,useHostTransitionStatus:rc,useFormState:Ps,useActionState:Ps,useOptimistic:function(e,t){var n=os();return U===null?(n.baseState=e,[e,n.queue.dispatch]):Ss(n,U,e,t)},useMemoCache:us,useCacheRefresh:ac,useEffectEvent:Hs};function vc(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:w({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var yc={enqueueSetState:function(e,t,n){e=e._reactInternals;var r=jd(),i=go(r);i.payload=t,n!=null&&(i.callback=n),t=_o(e,i,r),t!==null&&(Pd(t,e,r),vo(t,e,r))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=jd(),i=go(r);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=_o(e,i,r),t!==null&&(Pd(t,e,r),vo(t,e,r))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=jd(),r=go(n);r.tag=2,t!=null&&(r.callback=t),t=_o(e,r,n),t!==null&&(Pd(t,e,n),vo(t,e,n))}};function bc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate==`function`?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Hr(n,r)||!Hr(i,a):!0}function xc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps==`function`&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps==`function`&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&yc.enqueueReplaceState(t,t.state,null)}function Sc(e,t){var n=t;if(`ref`in t)for(var r in n={},t)r!==`ref`&&(n[r]=t[r]);if(e=e.defaultProps)for(var i in n===t&&(n=w({},n)),e)n[i]===void 0&&(n[i]=e[i]);return n}function Cc(e){bi(e)}function wc(e){console.error(e)}function Tc(e){bi(e)}function Ec(e,t){try{var n=e.onUncaughtError;n(t.value,{componentStack:t.stack})}catch(e){setTimeout(function(){throw e})}}function Dc(e,t,n){try{var r=e.onCaughtError;r(n.value,{componentStack:n.stack,errorBoundary:t.tag===1?t.stateNode:null})}catch(e){setTimeout(function(){throw e})}}function Oc(e,t,n){return n=go(n),n.tag=3,n.payload={element:null},n.callback=function(){Ec(e,t)},n}function kc(e){return e=go(e),e.tag=3,e}function Ac(e,t,n,r){var i=n.type.getDerivedStateFromError;if(typeof i==`function`){var a=r.value;e.payload=function(){return i(a)},e.callback=function(){Dc(t,n,r)}}var o=n.stateNode;o!==null&&typeof o.componentDidCatch==`function`&&(e.callback=function(){Dc(t,n,r),typeof i!=`function`&&(vd===null?vd=new Set([this]):vd.add(this));var e=r.stack;this.componentDidCatch(r.value,{componentStack:e===null?``:e})})}function jc(e,t,n,r,a){if(n.flags|=32768,typeof r==`object`&&r&&typeof r.then==`function`){if(t=n.alternate,t!==null&&xa(t,n,a,!0),n=Ao.current,n!==null){switch(n.tag){case 31:case 13:case 19:return jo===null?Kd():n.alternate===null&&ad===0&&(ad=3),n.flags&=-257,n.flags|=65536,n.lanes=a,r===Za?n.flags|=16384:(t=n.updateQueue,t===null?n.updateQueue=new Set([r]):t.add(r),mf(e,r,a)),!1;case 22:return n.flags|=65536,r===Za?n.flags|=16384:(t=n.updateQueue,t===null?(t={transitions:null,markerInstances:null,retryQueue:new Set([r])},n.updateQueue=t):(n=t.retryQueue,n===null?t.retryQueue=new Set([r]):n.add(r)),mf(e,r,a)),!1}throw Error(i(435,n.tag))}return mf(e,r,a),Kd(),!1}if(V)return t=Ao.current,t===null?(r!==oa&&(t=Error(i(423),{cause:r}),pa(Hi(t,n))),e=e.current.alternate,e.flags|=65536,a&=-a,e.lanes|=a,r=Hi(r,n),a=Oc(e.stateNode,r,a),yo(e,a),ad!==4&&(ad=2)):(!(t.flags&65536)&&(t.flags|=256),t.flags|=65536,t.lanes=a,r!==oa&&(e=Error(i(422),{cause:r}),pa(Hi(e,n)))),!1;var o=Error(i(520),{cause:r});if(o=Hi(o,n),dd===null?dd=[o]:dd.push(o),ad!==4&&(ad=2),t===null)return!0;r=Hi(r,n),n=t;do{switch(n.tag){case 3:return n.flags|=65536,e=a&-a,n.lanes|=e,e=Oc(n.stateNode,r,e),yo(n,e),!1;case 1:if(t=n.type,o=n.stateNode,!(n.flags&128)&&(typeof t.getDerivedStateFromError==`function`||o!==null&&typeof o.componentDidCatch==`function`&&(vd===null||!vd.has(o))))return n.flags|=65536,a&=-a,n.lanes|=a,a=kc(a),Ac(a,e,n,r),yo(n,a),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var Mc=Error(i(461)),Nc=!1;function Pc(e,t,n,r){t.child=e===null?fo(t,null,n,r):uo(t,e.child,n,r)}function Fc(e,t,n,r,i){n=n.render;var a=t.ref;if(`ref`in r){var o={};for(var s in r)s!==`ref`&&(o[s]=r[s])}else o=r;return Ca(t),r=Qo(e,t,n,o,a,i),s=ns(),e!==null&&!Nc?(rs(e,t,i),ll(e,t,i)):(V&&s&&ea(t),t.flags|=1,Pc(e,t,r,i),t.child)}function Ic(e,t,n,r,i){if(e===null){var a=n.type;return typeof a==`function`&&!Ni(a)&&a.defaultProps===void 0&&n.compare===null?(t.tag=15,t.type=a,Lc(e,t,a,r,i)):(e=Ii(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!ul(e,i)){var o=a.memoizedProps;if(n=n.compare,n=n===null?Hr:n,n(o,r)&&e.ref===t.ref)return ll(e,t,i)}return t.flags|=1,e=Pi(a,r),e.ref=t.ref,e.return=t,t.child=e}function Lc(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Hr(a,r)&&e.ref===t.ref){if(Nc=!1,t.pendingProps=r=a,ul(e,i))e.flags&131072&&(Nc=!0);else return t.lanes=e.lanes,ll(e,t,i)}}return Gc(e,t,n,r,i)}function Rc(e,t,n,r){var i=r.children,a=e===null?null:e.memoizedState;if(e===null&&t.stateNode===null&&(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),r.mode===`hidden`){if(t.flags&128){if(a=a===null?n:a.baseLanes|n,e!==null){for(r=t.child=e.child,i=0;r!==null;)i=i|r.lanes|r.childLanes,r=r.sibling;r=i&~a}else r=0,t.child=null;return Bc(e,t,a,n,r)}if(n&536870912)t.memoizedState={baseLanes:0,cachePool:null},e!==null&&Ka(t,a===null?null:a.cachePool),a===null?Oo():Do(t,a),Po(t);else return r=t.lanes=536870912,Bc(e,t,a===null?n:a.baseLanes|n,n,r)}else a===null?(e!==null&&Ka(t,null),Oo(),Fo()):(Ka(t,a.cachePool),Do(t,a),Fo(),t.memoizedState=null);return Pc(e,t,i,n),t.child}function zc(e,t){return e!==null&&e.tag===22||t.stateNode!==null||(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),t.sibling}function Bc(e,t,n,r,i){var a=Ga();return a=a===null?null:{parent:Aa._currentValue,pool:a},t.memoizedState={baseLanes:n,cachePool:a},e!==null&&Ka(t,null),Oo(),Po(t),e!==null&&xa(e,t,r,!0),t.childLanes=i,null}function Vc(e,t){return t=el({mode:t.mode,children:t.children},e.mode),t.ref=e.ref,e.child=t,t.return=e,t}function Hc(e,t,n){return uo(t,e.child,null,n),e=Vc(t,t.pendingProps),e.flags|=2,Io(t),t.memoizedState=null,e}function Uc(e,t,n){var r=t.pendingProps,a=!!(t.flags&128);if(t.flags&=-129,e===null){if(V){if(r.mode===`hidden`)return e=Vc(t,r),t.lanes=536870912,e.memoizedState={baseLanes:0,cachePool:null},zc(null,e);if(No(t),(e=B)?(e=am(e,aa),e=e!==null&&e.data===`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Yi===null?null:{id:Xi,overflow:Zi},retryLane:536870912,hydrationErrors:null},n=zi(e),n.return=t,t.child=n,ra=t,B=null)):e=null,e===null)throw sa(t);return t.lanes=536870912,null}return Vc(t,r)}var o=e.memoizedState;if(o!==null){var s=o.dehydrated;if(No(t),a){if(t.flags&256)t.flags&=-257,t=Hc(e,t,n);else if(t.memoizedState!==null)t.child=e.child,t.flags|=128,t=null;else throw Error(i(558))}else if(Nc||xa(e,t,n,!1),a=(n&e.childLanes)!==0,Nc||a){if(To.current===null){if(r=q,r!==null&&(s=gt(r,n),s!==0&&s!==o.retryLane))throw o.retryLane=s,Di(e,s),Pd(r,e,s),Mc;Kd()}t=Hc(e,t,n)}else e=o.treeContext,B=lm(s.nextSibling),ra=t,V=!0,ia=null,aa=!1,e!==null&&na(t,e),t=Vc(t,r),t.flags|=134221824;return t}return e=Pi(e.child,{mode:r.mode,children:r.children}),e.ref=t.ref,t.child=e,e.return=t,e}function Wc(e,t){var n=t.ref;if(n===null)e!==null&&e.ref!==null&&(t.flags|=4194816);else{if(typeof n!=`function`&&typeof n!=`object`)throw Error(i(284));(e===null||e.ref!==n)&&(t.flags|=4194816)}}function Gc(e,t,n,r,i){return Ca(t),n=Qo(e,t,n,r,void 0,i),r=ns(),e!==null&&!Nc?(rs(e,t,i),ll(e,t,i)):(V&&r&&ea(t),t.flags|=1,Pc(e,t,n,i),t.child)}function Kc(e,t,n,r,i,a){return Ca(t),t.updateQueue=null,n=es(t,r,n,i),$o(e),r=ns(),e!==null&&!Nc?(rs(e,t,a),ll(e,t,a)):(V&&r&&ea(t),t.flags|=1,Pc(e,t,n,a),t.child)}function qc(e,t,n,r,i){if(Ca(t),t.stateNode===null){var a=Ai,o=n.contextType;typeof o==`object`&&o&&(a=wa(o)),a=new n(r,a),t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,a.updater=yc,t.stateNode=a,a._reactInternals=t,a=t.stateNode,a.props=r,a.state=t.memoizedState,a.refs={},mo(t),o=n.contextType,a.context=typeof o==`object`&&o?wa(o):Ai,a.state=t.memoizedState,o=n.getDerivedStateFromProps,typeof o==`function`&&(vc(t,n,o,r),a.state=t.memoizedState),typeof n.getDerivedStateFromProps==`function`||typeof a.getSnapshotBeforeUpdate==`function`||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(o=a.state,typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount(),o!==a.state&&yc.enqueueReplaceState(a,a.state,null),So(t,r,a,i),xo(),a.state=t.memoizedState),typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!0}else if(e===null){a=t.stateNode;var s=t.memoizedProps,c=Sc(n,s);a.props=c;var l=a.context,u=n.contextType;o=Ai,typeof u==`object`&&u&&(o=wa(u));var d=n.getDerivedStateFromProps;u=typeof d==`function`||typeof a.getSnapshotBeforeUpdate==`function`,s=t.pendingProps!==s,u||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(s||l!==o)&&xc(t,a,r,o),po=!1;var f=t.memoizedState;a.state=f,So(t,r,a,i),xo(),l=t.memoizedState,s||f!==l||po?(typeof d==`function`&&(vc(t,n,d,r),l=t.memoizedState),(c=po||bc(t,n,c,r,f,l,o))?(u||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount==`function`&&(t.flags|=4194308)):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=o,r=c):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,ho(e,t),o=t.memoizedProps,u=Sc(n,o),a.props=u,d=t.pendingProps,f=a.context,l=n.contextType,c=Ai,typeof l==`object`&&l&&(c=wa(l)),s=n.getDerivedStateFromProps,(l=typeof s==`function`||typeof a.getSnapshotBeforeUpdate==`function`)||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(o!==d||f!==c)&&xc(t,a,r,c),po=!1,f=t.memoizedState,a.state=f,So(t,r,a,i),xo();var p=t.memoizedState;o!==d||f!==p||po||e!==null&&e.dependencies!==null&&Sa(e.dependencies)?(typeof s==`function`&&(vc(t,n,s,r),p=t.memoizedState),(u=po||bc(t,n,u,r,f,p,c)||e!==null&&e.dependencies!==null&&Sa(e.dependencies))?(l||typeof a.UNSAFE_componentWillUpdate!=`function`&&typeof a.componentWillUpdate!=`function`||(typeof a.componentWillUpdate==`function`&&a.componentWillUpdate(r,p,c),typeof a.UNSAFE_componentWillUpdate==`function`&&a.UNSAFE_componentWillUpdate(r,p,c)),typeof a.componentDidUpdate==`function`&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate==`function`&&(t.flags|=1024)):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=p),a.props=r,a.state=p,a.context=c,r=u):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return a=r,Wc(e,t),r=!!(t.flags&128),a||r?(a=t.stateNode,n=r&&typeof n.getDerivedStateFromError!=`function`?null:a.render(),t.flags|=1,e!==null&&r?(t.child=uo(t,e.child,null,i),t.child=uo(t,null,n,i)):Pc(e,t,n,i),t.memoizedState=a.state,e=t.child):e=ll(e,t,i),e}function Jc(e,t,n,r){return da(),t.flags|=256,Pc(e,t,n,r),t.child}var Yc={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function Xc(e){return{baseLanes:e,cachePool:qa()}}function Zc(e,t,n){return e=e===null?0:e.childLanes&~n,t&&(e|=ld),e}function Qc(e,t,n){var r=t.pendingProps,i=!1,a=!!(t.flags&128),o;if((o=a)||(o=e!==null&&e.memoizedState===null?!1:!!(Lo.current&2)),o&&(i=!0,t.flags&=-129),o=!!(t.flags&32),t.flags&=-33,e===null){if(V){if(i?Mo(t):Fo(),(e=B)?(e=am(e,aa),e=e!==null&&e.data!==`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Yi===null?null:{id:Xi,overflow:Zi},retryLane:536870912,hydrationErrors:null},n=zi(e),n.return=t,t.child=n,ra=t,B=null)):e=null,e===null)throw sa(t);return t.lanes=sm(e)?32:536870912,null}return a=r.children,r=r.fallback,i?(Fo(),i=t.mode,a=el({mode:`hidden`,children:a},i),r=Li(r,i,n,null),a.return=t,r.return=t,a.sibling=r,t.child=a,r=t.child,r.memoizedState=Xc(n),r.childLanes=Zc(e,o,n),t.memoizedState=Yc,zc(null,r)):(Mo(t),$c(t,a))}var s=e.memoizedState;if(s!==null){var c=s.dehydrated;if(c!==null)return nl(e,t,a,o,r,c,s,n)}return i?(Fo(),i=r.fallback,a=t.mode,s=e.child,c=s.sibling,r=Pi(s,{mode:`hidden`,children:r.children}),r.subtreeFlags=s.subtreeFlags&1206910976,c===null?(i=Li(i,a,n,null),i.flags|=2):i=Pi(c,i),i.return=t,r.return=t,r.sibling=i,t.child=r,zc(null,r),r=t.child,i=e.child.memoizedState,i===null?i=Xc(n):(a=i.cachePool,a===null?a=qa():(s=Aa._currentValue,a=a.parent===s?a:{parent:s,pool:s}),i={baseLanes:i.baseLanes|n,cachePool:a}),r.memoizedState=i,r.childLanes=Zc(e,o,n),t.memoizedState=Yc,zc(e.child,r)):(Mo(t),n=e.child,e=n.sibling,n=Pi(n,{mode:`visible`,children:r.children}),n.return=t,n.sibling=null,e!==null&&(o=t.deletions,o===null?(t.deletions=[e],t.flags|=16):o.push(e)),t.child=n,t.memoizedState=null,n)}function $c(e,t){return t=el({mode:`visible`,children:t},e.mode),t.return=e,e.child=t}function el(e,t){return e=Mi(22,e,null,t),e.lanes=0,e}function tl(e,t,n){return uo(t,e.child,null,n),e=$c(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function nl(e,t,n,r,a,o,s,c){if(n)return t.flags&256?(Mo(t),t.flags&=-257,tl(e,t,c)):t.memoizedState===null?(Fo(),o=a.fallback,s=t.mode,a=el({mode:`visible`,children:a.children},s),o=Li(o,s,c,null),o.flags|=2,a.return=t,o.return=t,a.sibling=o,t.child=a,uo(t,e.child,null,c),a=t.child,a.memoizedState=Xc(c),a.childLanes=Zc(e,r,c),t.memoizedState=Yc,zc(null,a)):(Fo(),t.child=e.child,t.flags|=128,null);if(Mo(t),sm(o)){if(r=o.nextSibling&&o.nextSibling.dataset,r)var l=r.dgst;return r=l,r!==``&&(a=Error(i(419)),a.stack=``,a.digest=r,pa({value:a,source:null,stack:null})),tl(e,t,c)}if(Nc||xa(e,t,c,!1),r=(c&e.childLanes)!==0,Nc||r){if(To.current!==null)return tl(e,t,c);if(r=q,r!==null&&(a=gt(r,c),a!==0&&a!==s.retryLane))throw s.retryLane=a,Di(e,a),Pd(r,e,a),Mc;return om(o)||Kd(),tl(e,t,c)}return om(o)?(t.flags|=192,t.child=e.child,null):(e=s.treeContext,B=lm(o.nextSibling),ra=t,V=!0,ia=null,aa=!1,e!==null&&na(t,e),t=$c(t,a.children),t.flags|=134221824,t)}function rl(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),ya(e.return,t,n)}function il(e){for(var t=null;e!==null;){var n=e.alternate;n!==null&&Bo(n)===null&&(t=e),e=e.sibling}return t}function al(e,t,n,r,i,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i,treeForkCount:a}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i,o.treeForkCount=a)}function ol(e){var t=e.child;for(e.child=null;t!==null;){var n=t.sibling;t.sibling=e.child,e.child=t,t=n}}function sl(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;r=r.children;var o=Lo.current;if(t.flags&128)return Ro(t,o),null;var s=!!(o&2);if(s?(o=o&1|2,t.flags|=128):o&=1,Ro(t,o),i===`backwards`&&e!==null?(ol(e),Pc(e,t,r,n),ol(e)):Pc(e,t,r,n),r=V?Ki:0,!s&&e!==null&&e.flags&128)a:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&rl(e,n,t);else if(e.tag===19)rl(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break a;for(;e.sibling===null;){if(e.return===null||e.return===t)break a;e=e.return}e.sibling.return=e.return,e=e.sibling}switch(i){case`backwards`:n=il(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null,ol(t)),al(t,!0,i,null,a,r);break;case`unstable_legacy-backwards`:for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Bo(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}al(t,!0,n,null,a,r);break;case`together`:al(t,!1,null,null,void 0,r);break;case`independent`:t.memoizedState=null;break;default:n=il(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),al(t,!1,i,n,a,r)}return t.child}function cl(e,t,n){var r=t.pendingProps;return _a(t,t.type,r.value),Pc(e,t,r.children,n),t.child}function ll(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),od|=t.lanes,(n&t.childLanes)===0){if(e!==null){if(xa(e,t,n,!1),(n&t.childLanes)===0)return null}else return null}if(e!==null&&t.child!==e.child)throw Error(i(153));if(t.child!==null){for(e=t.child,n=Pi(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=Pi(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function ul(e,t){return(e.lanes&t)!==0||(e=e.dependencies,!!(e!==null&&Sa(e)))}function dl(e,t,n){switch(t.tag){case 3:Te(t,t.stateNode.containerInfo),_a(t,Aa,e.memoizedState.cache),da();break;case 27:case 5:De(t);break;case 4:Te(t,t.stateNode.containerInfo);break;case 10:_a(t,t.type,t.memoizedProps.value);break;case 31:if(t.memoizedState!==null)return t.flags|=128,No(t),null;break;case 13:var r=t.memoizedState;if(r!==null){if(r.dehydrated!==null)return Mo(t),t.flags|=128,null;r=xa(e,t,n,!1);var i=t.child.childLanes;return r||(n&i)!==0?Qc(e,t,n):(Mo(t),e=ll(e,t,n),e===null?null:e.sibling)}Mo(t);break;case 19:if(t.flags&128)return sl(e,t,n);if(i=!!(e.flags&128),r=(n&t.childLanes)!==0,r||=(xa(e,t,n,!1),(n&t.childLanes)!==0),i){if(r)return sl(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),Ro(t,Lo.current),r)break;return null;case 22:return t.lanes=0,Rc(e,t,n,t.pendingProps);case 24:_a(t,Aa,e.memoizedState.cache)}return ll(e,t,n)}function fl(e,t,n){if(e!==null){if(e.memoizedProps!==t.pendingProps)Nc=!0;else{if(!ul(e,n)&&!(t.flags&128))return Nc=!1,dl(e,t,n);Nc=!!(e.flags&131072)}}else Nc=!1,V&&t.flags&1048576&&$i(t,Ki,t.index);switch(t.lanes=0,t.tag){case 16:a:{var r=t.pendingProps;if(e=eo(t.elementType),t.type=e,typeof e==`function`)Ni(e)?(r=Sc(e,r),t.tag=1,t=qc(null,t,e,r,n)):(t.tag=0,t=Gc(null,t,e,r,n));else{if(e!=null){var a=e.$$typeof;if(a===se){t.tag=11,t=Fc(null,t,e,r,n);break a}if(a===ue){t.tag=14,t=Ic(null,t,e,r,n);break a}if(a===O){t.tag=10,t.type=e,t=cl(null,t,n);break a}}throw t=ve(e)||e,Error(i(306,t,``))}}return t;case 0:return Gc(e,t,t.type,t.pendingProps,n);case 1:return r=t.type,a=Sc(r,t.pendingProps),qc(e,t,r,a,n);case 3:a:{if(Te(t,t.stateNode.containerInfo),e===null)throw Error(i(387));r=t.pendingProps;var o=t.memoizedState;a=o.element,ho(e,t),So(t,r,null,n);var s=t.memoizedState;if(r=s.cache,_a(t,Aa,r),r!==o.cache&&ba(t,[Aa],n,!0),xo(),r=s.element,o.isDehydrated){if(o={element:r,isDehydrated:!1,cache:s.cache},t.updateQueue.baseState=o,t.memoizedState=o,t.flags&256){t=Jc(e,t,r,n);break a}if(r!==a){a=Hi(Error(i(424)),t),pa(a),t=Jc(e,t,r,n);break a}switch(e=t.stateNode.containerInfo,e.nodeType){case 9:e=e.body;break;default:e=e.nodeName===`HTML`?e.ownerDocument.body:e}for(B=lm(e.firstChild),ra=t,V=!0,ia=null,aa=!0,n=fo(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling}else{if(da(),r===a){t=ll(e,t,n);break a}Pc(e,t,r,n)}t=t.child}return t;case 26:return Wc(e,t),e===null?(n=Nm(t.type,null,t.pendingProps,null))?t.memoizedState=n:V||(t.stateNode=fp(t.type,t.pendingProps,Ce.current,t)):t.memoizedState=Nm(t.type,e.memoizedProps,t.pendingProps,e.memoizedState),null;case 27:return De(t),e===null&&V&&(r=t.stateNode=hm(t.type,t.pendingProps,Ce.current),ra=t,aa=!0,a=B,Sp(t.type)?(um=a,B=lm(r.firstChild)):B=a),Pc(e,t,t.pendingProps.children,n),Wc(e,t),e===null&&(t.flags|=4194304),t.child;case 5:return e===null&&V&&((a=r=B)&&(r=rm(r,t.type,t.pendingProps,aa),r===null?a=!1:(t.stateNode=r,ra=t,B=lm(r.firstChild),aa=!1,a=!0)),a||sa(t)),De(t),a=t.type,o=t.pendingProps,s=e===null?null:e.memoizedProps,r=o.children,pp(a,o)?r=null:s!==null&&pp(a,s)&&(t.flags|=32),t.memoizedState!==null&&(a=Qo(e,t,ts,null,null,n),sh._currentValue=a),Wc(e,t),Pc(e,t,r,n),t.child;case 6:return e===null&&V&&((e=n=B)&&(n=im(n,t.pendingProps,aa),n===null?e=!1:(t.stateNode=n,ra=t,B=null,e=!0)),e||sa(t)),null;case 13:return Qc(e,t,n);case 4:return Te(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=uo(t,null,r,n):Pc(e,t,r,n),t.child;case 11:return Fc(e,t,t.type,t.pendingProps,n);case 7:return r=t.pendingProps,Wc(e,t),Pc(e,t,r,n),t.child;case 8:return Pc(e,t,t.pendingProps.children,n),t.child;case 12:return Pc(e,t,t.pendingProps.children,n),t.child;case 10:return cl(e,t,n);case 9:return a=t.type._context,r=t.pendingProps.children,Ca(t),a=wa(a),r=r(a),t.flags|=1,Pc(e,t,r,n),t.child;case 14:return Ic(e,t,t.type,t.pendingProps,n);case 15:return Lc(e,t,t.type,t.pendingProps,n);case 19:return sl(e,t,n);case 31:return Uc(e,t,n);case 22:return Rc(e,t,n,t.pendingProps);case 24:return Ca(t),r=wa(Aa),e===null?(a=Ga(),a===null&&(a=q,o=ja(),a.pooledCache=o,o.refCount++,o!==null&&(a.pooledCacheLanes|=n),a=o),t.memoizedState={parent:r,cache:a},mo(t),_a(t,Aa,a)):((e.lanes&n)!==0&&(ho(e,t),So(t,null,null,n),xo()),a=e.memoizedState,o=t.memoizedState,a.parent===r?(r=o.cache,_a(t,Aa,r),r!==a.cache&&ba(t,[Aa],n,!0)):(a={parent:r,cache:r},t.memoizedState=a,t.lanes===0&&(t.memoizedState=t.updateQueue.baseState=a),_a(t,Aa,r))),Pc(e,t,t.pendingProps.children,n),t.child;case 30:return t.stateNode===null&&(t.stateNode={autoName:null,paired:null,clones:null,ref:null}),r=t.pendingProps,r.name!=null&&r.name!==`auto`?t.flags|=e===null?18882560:18874368:V&&ea(t),e!==null&&e.memoizedProps.name!==r.name?t.flags|=4194816:Wc(e,t),Pc(e,t,r.children,n),t.child;case 29:throw t.pendingProps}throw Error(i(156,t.tag))}function pl(e){e.flags|=4}function ml(e,t,n,r,i){var a;if((a=!!(e.mode&32))&&(a=n===null?Jm(t,r):Jm(t,r)&&(r.src!==n.src||r.srcSet!==n.srcSet)),a){if(e.flags|=16777216,(i&335544128)===i){if(e.stateNode.complete)e.flags|=8192;else if(Ud())e.flags|=8192;else throw to=Za,Ya}}else e.flags&=-16777217}function hl(e,t){if(t.type!==`stylesheet`||t.state.loading&4)e.flags&=-16777217;else if(e.flags|=16777216,!Ym(t)){if(Ud())e.flags|=8192;else throw to=Za,Ya}}function gl(e,t){t!==null&&(e.flags|=4),e.flags&16384&&(t=e.tag===22?536870912:ut(),e.lanes|=t,ud|=t)}function _l(e,t){if(!V)switch(e.tailMode){case`visible`:break;case`collapsed`:for(var n=e.tail,r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null;break;default:for(t=e.tail,n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null}}function W(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&1206910976,r|=i.flags&1206910976,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function vl(e,t,n){var r=t.pendingProps;switch(ta(t),t.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return W(t),null;case 1:return W(t),null;case 3:return n=t.stateNode,r=null,e!==null&&(r=e.memoizedState.cache),t.memoizedState.cache!==r&&(t.flags|=2048),va(Aa),Ee(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(e===null||e.child===null)&&(ua(t)?pl(t):e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,fa())),W(t),null;case 26:var a=t.type,o=t.memoizedState;return e===null?(pl(t),o===null?(W(t),ml(t,a,null,r,n)):(W(t),hl(t,o))):o?o===e.memoizedState?(W(t),t.flags&=-16777217):(pl(t),W(t),hl(t,o)):(e=e.memoizedProps,e!==r&&pl(t),W(t),ml(t,a,e,r,n)),null;case 27:if(Oe(t),n=Ce.current,a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&pl(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return W(t),t.subtreeFlags&=-33554433,null}e=L.current,ua(t)?ca(t,e):(e=hm(a,r,n),t.stateNode=e,pl(t))}return W(t),t.subtreeFlags&=-33554433,null;case 5:if(Oe(t),a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&pl(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return W(t),t.subtreeFlags&=-33554433,null}if(o=L.current,ua(t))ca(t,o);else{var s=lp(Ce.current);switch(o){case 1:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case 2:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;default:switch(a){case`svg`:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case`math`:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;case`script`:o=s.createElement(`div`),o.innerHTML=`<script><\/script>`,o=o.removeChild(o.firstChild);break;case`select`:o=typeof r.is==`string`?s.createElement(`select`,{is:r.is}):s.createElement(`select`),r.multiple?o.multiple=!0:r.size&&(o.size=r.size);break;default:o=typeof r.is==`string`?s.createElement(a,{is:r.is}):s.createElement(a)}}o[St]=t,o[Ct]=r;a:for(s=t.child;s!==null;){if(s.tag===5||s.tag===6)o.appendChild(s.stateNode);else if(s.tag!==4&&s.tag!==27&&s.child!==null){s.child.return=s,s=s.child;continue}if(s===t)break a;for(;s.sibling===null;){if(s.return===null||s.return===t)break a;s=s.return}s.sibling.return=s.return,s=s.sibling}t.stateNode=o;a:switch(np(o,a,r),a){case`button`:case`input`:case`select`:case`textarea`:r=!!r.autoFocus;break a;case`img`:r=!0;break a;default:r=!1}r&&pl(t)}}return W(t),t.subtreeFlags&=-33554433,ml(t,t.type,e===null?null:e.memoizedProps,t.pendingProps,n),null;case 6:if(e&&t.stateNode!=null)e.memoizedProps!==r&&pl(t);else{if(typeof r!=`string`&&t.stateNode===null)throw Error(i(166));if(e=Ce.current,ua(t)){if(e=t.stateNode,n=t.memoizedProps,r=null,a=ra,a!==null)switch(a.tag){case 27:case 5:r=a.memoizedProps}e[St]=t,e=!!(e.nodeValue===n||r!==null&&!0===r.suppressHydrationWarning||ep(e.nodeValue,n)),e||sa(t,!0)}else e=lp(e).createTextNode(r),e[St]=t,t.stateNode=e}return W(t),null;case 31:if(n=t.memoizedState,e===null||e.memoizedState!==null){if(r=ua(t),n!==null){if(e===null){if(!r)throw Error(i(318));if(e=t.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(557));e[St]=t}else da(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;W(t),e=!1}else n=fa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=n),e=!0;if(!e)return t.flags&256?(Io(t),t):(Io(t),null);if(t.flags&128)throw Error(i(558))}return W(t),null;case 13:if(r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(a=ua(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(i(318));if(a=t.memoizedState,a=a===null?null:a.dehydrated,!a)throw Error(i(317));a[St]=t}else da(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;W(t),a=!1}else a=fa(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=a),a=!0;if(!a)return t.flags&256?(Io(t),t):(Io(t),null)}return Io(t),t.flags&128?(t.lanes=n,t):(n=r!==null,e=e!==null&&e.memoizedState!==null,n&&(r=t.child,a=null,r.alternate!==null&&r.alternate.memoizedState!==null&&r.alternate.memoizedState.cachePool!==null&&(a=r.alternate.memoizedState.cachePool.pool),o=null,r.memoizedState!==null&&r.memoizedState.cachePool!==null&&(o=r.memoizedState.cachePool.pool),o!==a&&(r.flags|=2048)),n!==e&&n&&(t.child.flags|=8192),gl(t,t.updateQueue),W(t),null);case 4:return Ee(),e===null&&Wf(t.stateNode.containerInfo),t.flags|=67108864,W(t),null;case 10:return va(t.type),W(t),null;case 19:if(zo(t),r=t.memoizedState,r===null)return W(t),null;if(a=!!(t.flags&128),o=r.rendering,o===null){if(a)_l(r,!1);else{if(ad!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=Bo(e),o!==null){for(t.flags|=128,_l(r,!1),e=o.updateQueue,t.updateQueue=e,gl(t,e),t.subtreeFlags=0,e=n,n=t.child;n!==null;)Fi(n,e),n=n.sibling;return Ro(t,Lo.current&1|2),V&&Qi(t,r.treeForkCount),t.child}e=e.sibling}r.tail!==null&&Ve()>gd&&(t.flags|=128,a=!0,_l(r,!1),t.lanes=4194304)}}else{if(!a){if(e=Bo(o),e!==null){if(t.flags|=128,a=!0,e=e.updateQueue,t.updateQueue=e,gl(t,e),_l(r,!0),r.tail===null&&r.tailMode!==`collapsed`&&r.tailMode!==`visible`&&!o.alternate&&!V)return W(t),null}else 2*Ve()-r.renderingStartTime>gd&&n!==536870912&&(t.flags|=128,a=!0,_l(r,!1),t.lanes=4194304)}r.isBackwards?(o.sibling=t.child,t.child=o):(e=r.last,e===null?t.child=o:e.sibling=o,r.last=o)}if(r.tail!==null){e=r.tail;a:{for(n=e;n!==null;){if(n.alternate!==null){n=!1;break a}n=n.sibling}n=!0}return r.rendering=e,r.tail=e.sibling,r.renderingStartTime=Ve(),e.sibling=null,o=Lo.current,o=a?o&1|2:o&1,r.tailMode===`visible`||r.tailMode===`collapsed`||!n||V?Ro(t,o):(n=o,I(Ao,t),I(Lo,n),jo===null&&(jo=t)),V&&Qi(t,r.treeForkCount),e}return W(t),null;case 22:case 23:return Io(t),ko(),r=t.memoizedState!==null,e===null?r&&(t.flags|=8192):e.memoizedState!==null!==r&&(t.flags|=8192),r?n&536870912&&!(t.flags&128)&&(W(t),t.subtreeFlags&6&&(t.flags|=8192)):W(t),n=t.updateQueue,n!==null&&gl(t,n.retryQueue),n=null,e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),r=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(r=t.memoizedState.cachePool.pool),r!==n&&(t.flags|=2048),e!==null&&F(Wa),null;case 24:return n=null,e!==null&&(n=e.memoizedState.cache),t.memoizedState.cache!==n&&(t.flags|=2048),va(Aa),W(t),null;case 25:return null;case 30:return t.flags|=33554432,W(t),null}throw Error(i(156,t.tag))}function yl(e,t){switch(ta(t),t.tag){case 1:return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return va(Aa),Ee(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 26:case 27:case 5:return Oe(t),null;case 31:if(t.memoizedState!==null){if(Io(t),t.alternate===null)throw Error(i(340));da()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 13:if(Io(t),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(i(340));da()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return zo(t),e=t.flags,e&65536?(t.flags=e&-65537|128,e=t.memoizedState,e!==null&&(e.rendering=null,e.tail=null),t.flags|=4,t):null;case 4:return Ee(),null;case 10:return va(t.type),null;case 22:case 23:return Io(t),ko(),e!==null&&F(Wa),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 24:return va(Aa),null;case 25:return null;default:return null}}function bl(e,t){switch(ta(t),t.tag){case 3:va(Aa),Ee();break;case 26:case 27:case 5:Oe(t);break;case 4:Ee();break;case 31:t.memoizedState!==null&&Io(t);break;case 13:Io(t);break;case 19:zo(t);break;case 10:va(t.type);break;case 22:case 23:Io(t),ko(),e!==null&&F(Wa);break;case 24:va(Aa)}}function xl(e,t){try{var n=t.updateQueue,r=n===null?null:n.lastEffect;if(r!==null){var i=r.next;n=i;do{if((n.tag&e)===e){r=void 0;var a=n.create,o=n.inst;r=a(),o.destroy=r}n=n.next}while(n!==i)}}catch(e){Z(t,t.return,e)}}function Sl(e,t,n){try{var r=t.updateQueue,i=r===null?null:r.lastEffect;if(i!==null){var a=i.next;r=a;do{if((r.tag&e)===e){var o=r.inst,s=o.destroy;if(s!==void 0){o.destroy=void 0,i=t;var c=n,l=s;try{l()}catch(e){Z(i,c,e)}}}r=r.next}while(r!==a)}}catch(e){Z(t,t.return,e)}}function Cl(e){var t=e.updateQueue;if(t!==null){var n=e.stateNode;try{wo(t,n)}catch(t){Z(e,e.return,t)}}}function wl(e,t,n){n.props=Sc(e.type,e.memoizedProps),n.state=e.memoizedState;try{n.componentWillUnmount()}catch(n){Z(e,t,n)}}function Tl(e,t){try{var n=e.ref;if(n!==null){switch(e.tag){case 26:case 27:case 5:var r=e.stateNode;break;case 30:var i=e.stateNode,a=_i(e.memoizedProps,i);(i.ref===null||i.ref.name!==a)&&(i.ref=Pp(a)),r=i.ref;break;case 7:if(e.stateNode===null){var o=new Fp(e);h(e.child,!1,Qp,o,void 0,void 0),e.stateNode=o}r=e.stateNode;break;default:r=e.stateNode}typeof n==`function`?e.refCleanup=n(r):n.current=r}}catch(n){Z(e,t,n)}}function El(e,t){var n=e.ref,r=e.refCleanup;if(n!==null){if(typeof r==`function`)try{r()}catch(n){Z(e,t,n)}finally{e.refCleanup=null,e=e.alternate,e!=null&&(e.refCleanup=null)}else if(typeof n==`function`)try{n(null)}catch(n){Z(e,t,n)}else n.current=null}}function Dl(e,t){if((e.tag===5||e.tag===27||e.tag===6)&&e.alternate===null&&t!==null)for(var n=0;n<t.length;n++)em(e.stateNode,t[n])}function Ol(e){for(var t=e.return;t!==null&&(jl(t)&&em(e.stateNode,t.stateNode),!Al(t));)t=t.return}function kl(e){for(var t=e.return;t!==null&&(jl(t)&&tm(e.stateNode,t.stateNode),!Al(t));)t=t.return}function Al(e){return e.tag===5||e.tag===3||e.tag===27}function jl(e){return e&&e.tag===7&&e.stateNode!==null}function Ml(e){var t=e.type,n=e.memoizedProps,r=e.stateNode;try{a:switch(t){case`button`:case`input`:case`select`:case`textarea`:n.autoFocus&&r.focus();break a;case`img`:n.src?r.src=n.src:n.srcSet&&(r.srcset=n.srcSet)}}catch(t){Z(e,e.return,t)}}function Nl(e,t,n){try{var r=e.stateNode;ip(r,e.type,n,t),r[Ct]=t}catch(t){Z(e,e.return,t)}}function Pl(e){return e.tag===5||e.tag===3||e.tag===26||e.tag===27&&Sp(e.type)||e.tag===4}function Fl(e){a:for(;;){for(;e.sibling===null;){if(e.return===null||Pl(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.tag===27&&Sp(e.type)||e.flags&2||e.child===null||e.tag===4)continue a;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Il(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?(n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n).insertBefore(i,t):(t=n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n,t.appendChild(i),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=vn)),Dl(e,r),z=!0;else if(i!==4&&(i===27&&(Dl(e,r),r=null,Sp(e.type)&&(n=e.stateNode,t=null)),e=e.child,e!==null))for(Il(e,t,n,r),e=e.sibling;e!==null;)Il(e,t,n,r),e=e.sibling}function Ll(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?n.insertBefore(i,t):n.appendChild(i),Dl(e,r),z=!0;else if(i!==4&&(i===27&&(Dl(e,r),r=null,Sp(e.type)&&(n=e.stateNode)),e=e.child,e!==null))for(Ll(e,t,n,r),e=e.sibling;e!==null;)Ll(e,t,n,r),e=e.sibling}function Rl(e){var t=e.stateNode,n=e.memoizedProps;try{for(var r=e.type,i=t.attributes;i.length;)t.removeAttributeNode(i[0]);np(t,r,n),t[St]=e,t[Ct]=n}catch(t){Z(e,e.return,t)}}var zl=!1,Bl=null;function Vl(e){(e.tag===30||e.subtreeFlags&33554432)&&(zl=!0)}var Hl=null;function Ul(){var e=Hl;return Hl=null,e}var Wl=0;function Gl(e,t,n,r,i){return Wl=0,Kl(e.child,t,n,r,i)}function Kl(e,t,n,r,i){for(var a=!1;e!==null;){if(e.tag===5){var o=e.stateNode;if(r!==null){var s=Op(o);r.push(s),s.view&&(a=!0)}else a||Op(o).view&&(a=!0);zl=!0,Tp(o,Wl===0?t:t+`_`+Wl,n),Wl++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&i||Kl(e.child,t,n,r,i)&&(a=!0));e=e.sibling}return a}function ql(e,t){for(;e!==null;)e.tag===5?Ep(e.stateNode,e.memoizedProps):(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&t||ql(e.child,t)),e=e.sibling}function Jl(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if((e.tag!==22||e.memoizedState===null)&&(Jl(e),e.tag===30&&e.flags&18874368&&e.stateNode.paired)){var t=e.memoizedProps;if(t.name==null||t.name===`auto`)throw Error(i(544));var n=t.name;t=yi(t.default,t.share),t!==`none`&&(Gl(e,n,t,null,!1)||ql(e.child,!1))}e=e.sibling}}function Yl(e,t){if(e.tag===30){var n=e.stateNode,r=e.memoizedProps,i=_i(r,n),a=yi(r.default,n.paired?r.share:r.enter);a===`none`?Jl(e):Gl(e,i,a,null,!1)?(Jl(e),n.paired||t||Nd(e,r.onEnter)):ql(e.child,!1)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Yl(e,t),e=e.sibling;else Jl(e)}function Xl(e){if(Bl!==null&&Bl.size!==0){var t=Bl;if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var n=e.memoizedProps,r=n.name;if(r!=null&&r!==`auto`){var i=t.get(r);if(i!==void 0){var a=yi(n.default,n.share);if(a!==`none`&&(Gl(e,r,a,null,!1)?(a=e.stateNode,i.paired=a,a.paired=i,Nd(e,n.onShare)):ql(e.child,!1)),t.delete(r),t.size===0)break}}}Xl(e)}e=e.sibling}}}function Zl(e){if(e.tag===30){var t=e.memoizedProps,n=_i(t,e.stateNode),r=Bl===null?void 0:Bl.get(n),i=yi(t.default,r===void 0?t.exit:t.share);i!==`none`&&(Gl(e,n,i,null,!1)?r===void 0?Nd(e,t.onExit):(i=e.stateNode,r.paired=i,i.paired=r,Bl.delete(n),Nd(e,t.onShare)):ql(e.child,!1)),Bl!==null&&Xl(e)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Zl(e),e=e.sibling;else Bl!==null&&Xl(e)}function Ql(e){for(e=e.child;e!==null;){if(e.tag===30){var t=e.memoizedProps,n=_i(t,e.stateNode);t=yi(t.default,t.update),e.flags&=-5,t!==`none`&&Gl(e,n,t,e.memoizedState=[],!1)}else e.subtreeFlags&33554432&&Ql(e);e=e.sibling}}function $l(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var t=e.stateNode;t.paired!==null&&(t.paired=null,ql(e.child,!1))}$l(e)}e=e.sibling}}function eu(e){if(e.tag===30)e.stateNode.paired=null,ql(e.child,!1),$l(e);else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)eu(e),e=e.sibling;else $l(e)}function tu(e){for(e=e.child;e!==null;)e.tag===30?ql(e.child,!1):e.subtreeFlags&33554432&&tu(e),e=e.sibling}function nu(e,t,n,r,i,a,o){for(var s=!1;t!==null;){if(t.tag===5){var c=t.stateNode;if(a!==null&&Wl<a.length){var l=a[Wl],u=Op(c);(l.view||u.view)&&(s=!0);var d;if(d=!(e.flags&4)){if(u.clip)d=!0;else{d=l.rect;var f=u.rect;d=d.y!==f.y||d.x!==f.x||d.height!==f.height||d.width!==f.width}}d&&(e.flags|=4),u.abs?u=!l.abs:(l=l.rect,u=u.rect,u=l.height!==u.height||l.width!==u.width),u&&(e.flags|=32)}else e.flags|=32;e.flags&4&&Tp(c,Wl===0?n:n+`_`+Wl,i),s&&e.flags&4||(Hl===null&&(Hl=[]),Hl.push(c,Wl===0?r:r+`_`+Wl,t.memoizedProps)),Wl++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&o?e.flags|=t.flags&32:nu(e,t.child,n,r,i,a,o)&&(s=!0));t=t.sibling}return s}function ru(e,t){for(e=e.child;e!==null;){if(e.tag===30){var n=e.memoizedProps,r=e.stateNode,i=_i(n,r),a=yi(n.default,n.update);if(t){r=r.clones;var o=r===null?null:r.map(kp)}else o=e.memoizedState,e.memoizedState=null;r=e;var s=e.child;Wl=0,i=nu(r,s,i,i,a,o,!1),e.flags&4&&i&&(t||Nd(e,n.onUpdate))}else e.subtreeFlags&33554432&&ru(e,t);e=e.sibling}}var iu=!1,G=!1,au=!1,ou=!1,su=typeof WeakSet==`function`?WeakSet:Set,cu=null,lu=!1,uu=!1,du=!1,fu=!1;function pu(e,t,n){if(e=e.containerInfo,sp=gh,e=qr(e),Jr(e)){if(`selectionStart`in e)var r={start:e.selectionStart,end:e.selectionEnd};else a:{r=(r=e.ownerDocument)&&r.defaultView||window;var i=r.getSelection&&r.getSelection();if(i&&i.rangeCount!==0){r=i.anchorNode;var a=i.anchorOffset,o=i.focusNode;i=i.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break a}var s=0,c=-1,l=-1,u=0,d=0,f=e,p=null;b:for(;;){for(var m;f!==r||a!==0&&f.nodeType!==3||(c=s+a),f!==o||i!==0&&f.nodeType!==3||(l=s+i),f.nodeType===3&&(s+=f.nodeValue.length),(m=f.firstChild)!==null;)p=f,f=m;for(;;){if(f===e)break b;if(p===r&&++u===a&&(c=s),p===o&&++d===i&&(l=s),(m=f.nextSibling)!==null)break;f=p,p=f.parentNode}f=m}r=c===-1||l===-1?null:{start:c,end:l}}else r=null}r||={start:0,end:0}}else r=null;for(cp={focusedElem:e,selectionRange:r},gh=!1,n=(n&335544064)===n,cu=t,t=n?9270:1024;cu!==null;){if(e=cu,n&&(r=e.deletions,r!==null))for(a=0;a<r.length;a++)n&&Zl(r[a]);if(e.alternate===null&&e.flags&2)n&&Vl(e),mu(n);else{if(e.tag===22){if(r=e.alternate,e.memoizedState!==null){r!==null&&r.memoizedState===null&&n&&Zl(r),mu(n);continue}if(r!==null&&r.memoizedState!==null){n&&Vl(e),mu(n);continue}}r=e.child,(e.subtreeFlags&t)!==0&&r!==null?(r.return=e,cu=r):(n&&Ql(e),mu(n))}}Bl=null}function mu(e){for(;cu!==null;){var t=cu,n=e,r=t.alternate,a=t.flags;switch(t.tag){case 0:case 11:case 15:break;case 1:if(a&1024&&r!==null){n=void 0,a=r.memoizedProps,r=r.memoizedState;var o=t.stateNode;try{var s=Sc(t.type,a);n=o.getSnapshotBeforeUpdate(s,r),o.__reactInternalSnapshotBeforeUpdate=n}catch(e){Z(t,t.return,e)}}break;case 3:if(a&1024){if(r=t.stateNode.containerInfo,n=r.nodeType,n===9)nm(r);else if(n===1)switch(r.nodeName){case`HEAD`:case`HTML`:case`BODY`:nm(r);break;default:r.textContent=``}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&r!==null&&(n=_i(r.memoizedProps,r.stateNode),a=t.memoizedProps,a=yi(a.default,a.update),a!==`none`&&Gl(r,n,a,r.memoizedState=[],!0));break;default:if(a&1024)throw Error(i(163))}if(r=t.sibling,r!==null){r.return=t.return,cu=r;break}cu=t.return}}function hu(e,t,n){var r=n.flags;switch(n.tag){case 0:case 11:case 15:Fu(e,n),r&4&&xl(5,n);break;case 1:if(Fu(e,n),r&4){if(e=n.stateNode,t===null)try{e.componentDidMount()}catch(e){Z(n,n.return,e)}else{var i=Sc(n.type,t.memoizedProps);t=t.memoizedState;try{e.componentDidUpdate(i,t,e.__reactInternalSnapshotBeforeUpdate)}catch(e){Z(n,n.return,e)}}}r&64&&Cl(n),r&512&&Tl(n,n.return);break;case 3:if(Fu(e,n),r&64&&(e=n.updateQueue,e!==null)){if(t=null,n.child!==null)switch(n.child.tag){case 27:case 5:t=n.child.stateNode;break;case 1:t=n.child.stateNode}try{wo(e,t)}catch(e){Z(n,n.return,e)}}break;case 27:t===null&&r&4&&Rl(n);case 26:case 5:Fu(e,n),t===null&&r&4&&Ml(n),r&512&&Tl(n,n.return);break;case 12:Fu(e,n);break;case 31:Fu(e,n),r&4&&wu(e,n);break;case 13:Fu(e,n),r&4&&Tu(e,n),r&64&&(e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null&&(n=_f.bind(null,n),cm(e,n))));break;case 22:if(r=n.memoizedState!==null||iu,!r){var a=t!==null&&t.memoizedState!==null||G;t=iu,i=G,iu=r,(G=a)&&!i?(r=2,n.subtreeFlags&8772&&(r|=1),Lu(e,n,r)):Fu(e,n),iu=t,G=i}break;case 30:Fu(e,n),r&512&&Tl(n,n.return);break;case 7:r&512&&Tl(n,n.return);default:Fu(e,n)}}function gu(e,t){for(e=e.child;e!==null;)_u(e,t),e=e.sibling}function _u(e,t){switch(e.tag){case 5:case 26:try{var n=e.stateNode;if(t){var r=n.style;typeof r.setProperty==`function`?r.setProperty(`display`,`none`,`important`):r.display=`none`}else{var i=e.stateNode,a=e.memoizedProps.style,o=a!=null&&a.hasOwnProperty(`display`)?a.display:null;i.style.display=o==null||typeof o==`boolean`?``:(``+o).trim()}}catch(t){Z(e,e.return,t)}vu(e,t);break;case 6:try{e.stateNode.nodeValue=t?``:e.memoizedProps,z=!0}catch(t){Z(e,e.return,t)}break;case 18:try{var s=e.stateNode;t?wp(s,!0):wp(e.stateNode,!1)}catch(t){Z(e,e.return,t)}break;case 22:case 23:e.memoizedState===null&&gu(e,t);break;default:gu(e,t)}}function vu(e,t){if(e.subtreeFlags&67108864)for(e=e.child;e!==null;){a:{var n=e,r=t;switch(n.tag){case 4:_u(n,r);break a;case 22:n.memoizedState===null&&vu(n,r);break a;default:vu(n,r)}}e=e.sibling}}function yu(e){var t=e.alternate;t!==null&&(e.alternate=null,yu(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&jt(t)),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}var bu=null,xu=!1;function Su(e,t,n){for(n=n.child;n!==null;)Cu(e,t,n),n=n.sibling}function Cu(e,t,n){if(Xe&&typeof Xe.onCommitFiberUnmount==`function`)try{Xe.onCommitFiberUnmount(Ye,n)}catch{}switch(n.tag){case 26:G||El(n,t),Su(e,t,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!G&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:G||El(n,t),kl(n);var r=bu,i=xu;Sp(n.type)&&(bu=n.stateNode,xu=!1),Su(e,t,n),gm(n.stateNode,n.type,n.memoizedProps),bu=r,xu=i;break;case 5:G||El(n,t),kl(n);case 6:if(n.tag===6&&kl(n),r=bu,i=xu,bu=null,Su(e,t,n),bu=r,xu=i,bu!==null){if(xu)try{(bu.nodeType===9?bu.body:bu.nodeName===`HTML`?bu.ownerDocument.body:bu).removeChild(n.stateNode),z=!0}catch(e){Z(n,t,e)}else try{bu.removeChild(n.stateNode),z=!0}catch(e){Z(n,t,e)}}break;case 18:bu!==null&&(xu?(e=bu,Cp(e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,n.stateNode),Hh(e)):Cp(bu,n.stateNode));break;case 4:r=bu,i=xu,bu=n.stateNode.containerInfo,xu=!0,Su(e,t,n),bu=r,xu=i;break;case 0:case 11:case 14:case 15:Sl(2,n,t),G||Sl(4,n,t),Su(e,t,n);break;case 1:G||(El(n,t),r=n.stateNode,typeof r.componentWillUnmount==`function`&&wl(n,t,r)),Su(e,t,n);break;case 21:Su(e,t,n);break;case 22:G=(r=G)||n.memoizedState!==null,Su(e,t,n),G=r;break;case 30:El(n,t),Su(e,t,n);break;case 7:G||El(n,t),Su(e,t,n);break;default:Su(e,t,n)}}function wu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null))){e=e.dehydrated;try{Hh(e)}catch(e){Z(t,t.return,e)}}}function Tu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null&&(e=e.dehydrated,e!==null))))try{Hh(e)}catch(e){Z(t,t.return,e)}}function Eu(e){switch(e.tag){case 31:case 13:case 19:var t=e.stateNode;return t===null&&(t=e.stateNode=new su),t;case 22:return e=e.stateNode,t=e._retryCache,t===null&&(t=e._retryCache=new su),t;default:throw Error(i(435,e.tag))}}function Du(e,t){var n=Eu(e);t.forEach(function(t){if(!n.has(t)){n.add(t);var r=vf.bind(null,e,t);t.then(r,r)}})}function Ou(e,t,n){var r=t.deletions;if(r!==null)for(var a=0;a<r.length;a++){var o=r[a],s=e,c=t,l=c;a:for(;l!==null;){switch(l.tag){case 27:if(Sp(l.type)){bu=l.stateNode,xu=!1;break a}break;case 5:bu=l.stateNode,xu=!1;break a;case 3:case 4:bu=l.stateNode.containerInfo,xu=!0;break a}l=l.return}if(bu===null)throw Error(i(160));Cu(s,c,o),bu=null,xu=!1,s=o.alternate,s!==null&&(s.return=null),o.return=null}if(t.subtreeFlags&13886)for(t=t.child;t!==null;)Au(t,e,n),t=t.sibling}var ku=null;function Au(e,t,n){var r=e.alternate,a=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(a&4&&(r=e.updateQueue,r=r===null?null:r.events,r!==null))for(var o=0;o<r.length;o++){var s=r[o];s.ref.impl=s.nextImpl}Ou(t,e,n),ju(e),a&4&&(Sl(3,e,e.return),xl(3,e),Sl(5,e,e.return));break;case 1:Ou(t,e,n),ju(e),a&512&&(G||r===null||El(r,r.return)),a&64&&iu&&(e=e.updateQueue,e!==null&&(t=e.callbacks,t!==null&&(n=e.shared.hiddenCallbacks,e.shared.hiddenCallbacks=n===null?t:n.concat(t))));break;case 26:if(o=ku,Ou(t,e,n),ju(e),a&512&&(G||r===null||El(r,r.return)),a&4){if(a=r===null?null:r.memoizedState,n=e.memoizedState,r===null){if(n===null){if(e.stateNode===null){if(iu)e.stateNode=fp(e.type,e.memoizedProps,t.containerInfo,e);else{a:{t=e.type,n=e.memoizedProps,a=o.ownerDocument||o;b:switch(t){case`title`:r=a.getElementsByTagName(`title`)[0],(!r||r[kt]||r[St]||r.namespaceURI===`http://www.w3.org/2000/svg`||r.hasAttribute(`itemprop`))&&(r=a.createElement(t),a.head.insertBefore(r,a.querySelector(`head > title`))),np(r,t,n),r[St]=e,It(r),t=r;break a;case`link`:if(o=Gm(`link`,`href`,a).get(t+(n.href||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`href`)===(n.href==null||n.href===``?null:n.href)&&r.getAttribute(`rel`)===(n.rel==null?null:n.rel)&&r.getAttribute(`title`)===(n.title==null?null:n.title)&&r.getAttribute(`crossorigin`)===(n.crossOrigin==null?null:n.crossOrigin)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;case`meta`:if(o=Gm(`meta`,`content`,a).get(t+(n.content||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`content`)===(n.content==null?null:``+n.content)&&r.getAttribute(`name`)===(n.name==null?null:n.name)&&r.getAttribute(`property`)===(n.property==null?null:n.property)&&r.getAttribute(`http-equiv`)===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute(`charset`)===(n.charSet==null?null:n.charSet)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;default:throw Error(i(468,t))}r[St]=e,It(r),t=r}e.stateNode=t}}else iu||Km(o,e.type,e.stateNode)}else e.stateNode=Bm(o,n,e.memoizedProps)}else a===n?n===null&&e.stateNode!==null&&Nl(e,e.memoizedProps,r.memoizedProps):(a===null?(t=r.stateNode,t===null||G||t.parentNode.removeChild(t)):a.count--,n===null?iu||Km(o,e.type,e.stateNode):Bm(o,n,e.memoizedProps))}break;case 27:Ou(t,e,n),ju(e),a&512&&(G||r===null||El(r,r.return)),r!==null&&a&4&&Nl(e,e.memoizedProps,r.memoizedProps);break;case 5:if(o=au,au=!1,Ou(t,e,n),au=o,ju(e),a&512&&(G||r===null||El(r,r.return)),e.flags&32){t=e.stateNode;try{un(t,``),z=!0}catch(t){Z(e,e.return,t)}}a&4&&e.stateNode!=null&&(t=e.memoizedProps,Nl(e,t,r===null?t:r.memoizedProps)),a&1024&&(ou=!0);break;case 6:if(Ou(t,e,n),ju(e),a&4){if(e.stateNode===null)throw Error(i(162));t=e.memoizedProps,n=e.stateNode;try{n.nodeValue=t,z=!0}catch(t){Z(e,e.return,t)}}break;case 3:if(z=!1,Wm=null,o=ku,ku=bm(t.containerInfo),Ou(t,e,n),ku=o,ju(e),a&4&&r!==null&&r.memoizedState.isDehydrated)try{Hh(t.containerInfo)}catch(t){Z(e,e.return,t)}ou&&(ou=!1,Mu(e)),z=!1;break;case 4:a=au,au=iu,r=Kt(),o=ku,ku=bm(e.stateNode.containerInfo),Ou(t,e,n),ju(e),ku=o,z&&uu&&(du=!0),z=r,au=a;break;case 12:Ou(t,e,n),ju(e);break;case 31:Ou(t,e,n),ju(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Du(e,t)));break;case 13:Ou(t,e,n),ju(e),e.child.flags&8192&&e.memoizedState!==null!=(r!==null&&r.memoizedState!==null)&&(md=Ve()),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Du(e,t)));break;case 22:o=e.memoizedState!==null,s=r!==null&&r.memoizedState!==null;var c=iu,l=G,u=au;iu=c||o,au=u||o,G=l||s,Ou(t,e,n),G=l,au=u,iu=c,ju(e),a&8192&&(t=e.stateNode,t._visibility=o?t._visibility&-2:t._visibility|1,!o||r===null||s||iu||G||(t=s||G,n=iu,r=G,iu=o||iu,G=t,Iu(e,2),iu=n,G=r),!o&&au||gu(e,o)),a&4&&(t=e.updateQueue,t!==null&&(n=t.retryQueue,n!==null&&(t.retryQueue=null,Du(e,n))));break;case 19:Ou(t,e,n),ju(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Du(e,t)));break;case 30:a&512&&(G||r===null||El(r,r.return)),a=Kt(),o=uu,s=(n&335544064)===n,c=e.memoizedProps,uu=s&&yi(c.default,c.update)!==`none`,Ou(t,e,n),ju(e),s&&r!==null&&z&&(e.flags|=4),uu=o,z=a;break;case 21:break;case 7:a&512&&(G||r===null||El(r,r.return)),r&&r.stateNode!==null&&(r.stateNode._fragmentFiber=e);default:Ou(t,e,n),ju(e)}}function ju(e){var t=e.flags;if(t&2){try{for(var n,r=e.return;r!==null;){if(Pl(r)){n=r;break}r=r.return}r=null;for(var a=e.return;a!==null;){if(jl(a)){var o=a.stateNode;r===null?r=[o]:r.push(o)}if(Al(a))break;a=a.return}var s=r;if(n==null)throw Error(i(160));switch(n.tag){case 27:var c=n.stateNode;Ll(e,Fl(e),c,s);break;case 5:var l=n.stateNode;n.flags&32&&(un(l,``),n.flags&=-33),Ll(e,Fl(e),l,s);break;case 3:case 4:var u=n.stateNode.containerInfo;Il(e,Fl(e),u,s);break;default:throw Error(i(161))}}catch(t){Z(e,e.return,t)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Mu(e){if(e.subtreeFlags&1024)for(e=e.child;e!==null;){var t=e;Mu(t),t.tag===5&&t.flags&1024&&(t=t.stateNode,gh=!0,t.reset(),gh=!1),e=e.sibling}}function Nu(e,t){if(t.subtreeFlags&9270)for(t=t.child;t!==null;)Pu(t,e),t=t.sibling;else ru(t,!1)}function Pu(e,t){var n=e.alternate;if(n===null)Yl(e,!1);else switch(e.tag){case 3:if(fu=lu=!1,Ul(),Nu(t,e),!lu&&!du){if(e=Hl,e!==null)for(var r=0;r<e.length;r+=3){n=e[r];var i=e[r+1];Ep(n,e[r+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(`+i+`)`})}e=t.containerInfo,e=e.nodeType===9?e.documentElement:e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===``&&(e.style.viewTransitionName=`none`,e.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(root)`}),e.animate({width:[0,0],height:[0,0]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition`})),fu=!0}Hl=null;break;case 5:Nu(t,e);break;case 4:r=lu,lu=!1,Nu(t,e),lu&&(du=!0),lu=r;break;case 22:e.memoizedState===null&&(n.memoizedState===null?Nu(t,e):Yl(e,!1));break;case 30:r=lu,i=Ul(),lu=!1,Nu(t,e),lu&&(e.flags|=4);var a=e.memoizedProps,o=e.stateNode;t=_i(a,o),o=_i(n.memoizedProps,o);var s=yi(a.default,a.update);s===`none`?t=!1:(a=n.memoizedState,n.memoizedState=null,n=e.child,Wl=0,t=nu(e,n,t,o,s,a,!0),Wl!==(a===null?0:a.length)&&(e.flags|=32)),e.flags&4&&t?(Nd(e,e.memoizedProps.onUpdate),Hl=i):i!==null&&(i.push.apply(i,Hl),Hl=i),lu=e.flags&32?!0:r;break;default:Nu(t,e)}}function Fu(e,t){if(t.subtreeFlags&8772)for(t=t.child;t!==null;)hu(e,t.alternate,t),t=t.sibling}function Iu(e,t){for(e=e.child;e!==null;){var n=e,r=t;switch(n.tag){case 0:case 11:case 14:case 15:Sl(4,n,n.return),Iu(n,r);break;case 1:El(n,n.return);var i=n.stateNode;typeof i.componentWillUnmount==`function`&&wl(n,n.return,i),Iu(n,r);break;case 27:r&2&&gm(n.stateNode,n.type,n.memoizedProps);case 5:El(n,n.return),n.tag!==5&&n.tag!==27||kl(n),Iu(n,r);break;case 6:kl(n);break;case 26:El(n,n.return),i=n.stateNode,n.memoizedState!==null||i===null||G||i.parentNode.removeChild(i),Iu(n,r);break;case 22:n.memoizedState===null&&Iu(n,r);break;case 30:El(n,n.return),Iu(n,r);break;case 7:El(n,n.return);default:Iu(n,r)}e=e.sibling}}function Lu(e,t,n){for(n=t.subtreeFlags&8772?n:n&-2,t=t.child;t!==null;){var r=t.alternate,i=e,a=t,o=a.flags,s=!!(n&1);switch(a.tag){case 0:case 11:case 15:Lu(i,a,n),xl(4,a);break;case 1:if(Lu(i,a,n),r=a,i=r.stateNode,typeof i.componentDidMount==`function`)try{i.componentDidMount()}catch(e){Z(r,r.return,e)}if(r=a,i=r.updateQueue,i!==null){var c=r.stateNode;try{var l=i.shared.hiddenCallbacks;if(l!==null)for(i.shared.hiddenCallbacks=null,i=0;i<l.length;i++)Co(l[i],c)}catch(e){Z(r,r.return,e)}}s&&o&64&&Cl(a),Tl(a,a.return);break;case 27:n&2&&Rl(a);case 5:a.tag!==5&&a.tag!==27||Ol(a),Lu(i,a,n),s&&r===null&&o&4&&Ml(a),Tl(a,a.return);break;case 6:Ol(a);break;case 26:c=a.stateNode,a.memoizedState!==null||c===null||iu||Km(bm(c.ownerDocument),a.type,c),Lu(i,a,n),s&&r===null&&o&4&&Ml(a),Tl(a,a.return);break;case 12:Lu(i,a,n);break;case 31:Lu(i,a,n),s&&o&4&&wu(i,a);break;case 13:Lu(i,a,n),s&&o&4&&Tu(i,a);break;case 22:a.memoizedState===null&&Lu(i,a,n),Tl(a,a.return);break;case 30:Lu(i,a,n),Tl(a,a.return);break;case 7:Tl(a,a.return);default:Lu(i,a,n)}t=t.sibling}}function Ru(e,t){var n=null;e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),e=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(e=t.memoizedState.cachePool.pool),e!==n&&(e!=null&&e.refCount++,n!=null&&Ma(n))}function zu(e,t){e=null,t.alternate!==null&&(e=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==e&&(t.refCount++,e!=null&&Ma(e))}function Bu(e,t,n,r){var i=(n&335544064)===n;if(t.subtreeFlags&(i?10262:10256))for(t=t.child;t!==null;)Vu(e,t,n,r),t=t.sibling;else i&&tu(t)}function Vu(e,t,n,r){var i=(n&335544064)===n;i&&t.alternate===null&&t.return!==null&&t.return.alternate!==null&&eu(t);var a=t.flags;switch(t.tag){case 0:case 11:case 15:Bu(e,t,n,r),a&2048&&xl(9,t);break;case 1:Bu(e,t,n,r);break;case 3:Bu(e,t,n,r),i&&fu&&(e=e.containerInfo,e=e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,e.style.viewTransitionName===`root`&&(e.style.viewTransitionName=``),e=e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===`none`&&(e.style.viewTransitionName=``)),a&2048&&(a=null,t.alternate!==null&&(a=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==a&&(t.refCount++,a!=null&&Ma(a)));break;case 12:if(a&2048){Bu(e,t,n,r),a=t.stateNode;try{var o=t.memoizedProps,s=o.id,c=o.onPostCommit;typeof c==`function`&&c(s,t.alternate===null?`mount`:`update`,a.passiveEffectDuration,-0)}catch(e){Z(t,t.return,e)}}else Bu(e,t,n,r);break;case 31:Bu(e,t,n,r);break;case 13:Bu(e,t,n,r);break;case 23:break;case 22:o=t.stateNode,s=t.alternate,t.memoizedState===null?(i&&s!==null&&s.memoizedState!==null&&eu(t),o._visibility&2?Bu(e,t,n,r):(o._visibility|=2,Hu(e,t,n,r,!!(t.subtreeFlags&10256)||!1))):(i&&s!==null&&s.memoizedState===null&&eu(s),o._visibility&2?Bu(e,t,n,r):Uu(e,t)),a&2048&&Ru(s,t);break;case 24:Bu(e,t,n,r),a&2048&&zu(t.alternate,t);break;case 30:i&&(a=t.alternate,a!==null&&(ql(a.child,!0),ql(t.child,!0))),Bu(e,t,n,r);break;default:Bu(e,t,n,r)}}function Hu(e,t,n,r,i){for(i&&=!!(t.subtreeFlags&10256)||!1,t=t.child;t!==null;){var a=e,o=t,s=n,c=r,l=o.flags;switch(o.tag){case 0:case 11:case 15:Hu(a,o,s,c,i),xl(8,o);break;case 23:break;case 22:var u=o.stateNode;o.memoizedState===null?(u._visibility|=2,Hu(a,o,s,c,i)):u._visibility&2?Hu(a,o,s,c,i):Uu(a,o),i&&l&2048&&Ru(o.alternate,o);break;case 24:Hu(a,o,s,c,i),i&&l&2048&&zu(o.alternate,o);break;default:Hu(a,o,s,c,i)}t=t.sibling}}function Uu(e,t){if(t.subtreeFlags&10256)for(t=t.child;t!==null;){var n=e,r=t,i=r.flags;switch(r.tag){case 22:Uu(n,r),i&2048&&Ru(r.alternate,r);break;case 24:Uu(n,r),i&2048&&zu(r.alternate,r);break;default:Uu(n,r)}t=t.sibling}}var Wu=8192;function Gu(e,t,n){if(e.subtreeFlags&Wu)for(e=e.child;e!==null;)Ku(e,t,n),e=e.sibling}function Ku(e,t,n){switch(e.tag){case 26:Gu(e,t,n),e.flags&Wu&&(e.memoizedState===null?(e=e.stateNode,(t&335544128)===t&&Zm(n,e)):Qm(n,ku,e.memoizedState,e.memoizedProps));break;case 5:Gu(e,t,n),e.flags&Wu&&(e=e.stateNode,(t&335544128)===t&&Zm(n,e));break;case 3:case 4:var r=ku;ku=bm(e.stateNode.containerInfo),Gu(e,t,n),ku=r;break;case 22:e.memoizedState===null&&(r=e.alternate,r!==null&&r.memoizedState!==null?(r=Wu,Wu=16777216,Gu(e,t,n),Wu=r):Gu(e,t,n));break;case 30:if((e.flags&Wu)!==0&&(r=e.memoizedProps.name,r!=null&&r!==`auto`)){var i=e.stateNode;i.paired=null,Bl===null&&(Bl=new Map),Bl.set(r,i)}Gu(e,t,n);break;default:Gu(e,t,n)}}function qu(e){var t=e.alternate;if(t!==null&&(e=t.child,e!==null)){t.child=null;do t=e.sibling,e.sibling=null,e=t;while(e!==null)}}function Ju(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];cu=r,Zu(r,e)}qu(e)}if(e.subtreeFlags&10256)for(e=e.child;e!==null;)Yu(e),e=e.sibling}function Yu(e){switch(e.tag){case 0:case 11:case 15:Ju(e),e.flags&2048&&Sl(9,e,e.return);break;case 3:Ju(e);break;case 12:Ju(e);break;case 22:var t=e.stateNode;e.memoizedState!==null&&t._visibility&2&&(e.return===null||e.return.tag!==13)?(t._visibility&=-3,Xu(e)):Ju(e);break;default:Ju(e)}}function Xu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];cu=r,Zu(r,e)}qu(e)}for(e=e.child;e!==null;){switch(t=e,t.tag){case 0:case 11:case 15:Sl(8,t,t.return),Xu(t);break;case 22:n=t.stateNode,n._visibility&2&&(n._visibility&=-3,Xu(t));break;default:Xu(t)}e=e.sibling}}function Zu(e,t){for(;cu!==null;){var n=cu;switch(n.tag){case 0:case 11:case 15:Sl(8,n,t);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var r=n.memoizedState.cachePool.pool;r!=null&&r.refCount++}break;case 24:Ma(n.memoizedState.cache)}if(r=n.child,r!==null)r.return=n,cu=r;else a:for(n=e;cu!==null;){r=cu;var i=r.sibling,a=r.return;if(yu(r),r===n){cu=null;break a}if(i!==null){i.return=a,cu=i;break a}cu=a}}}var Qu={getCacheForType:function(e){var t=wa(Aa),n=t.data.get(e);return n===void 0&&(n=e(),t.data.set(e,n)),n},cacheSignal:function(){return wa(Aa).controller.signal}},$u=typeof WeakMap==`function`?WeakMap:Map,K=0,q=null,J=null,Y=0,X=0,ed=null,td=!1,nd=!1,rd=!1,id=0,ad=0,od=0,sd=0,cd=0,ld=0,ud=0,dd=null,fd=null,pd=!1,md=0,hd=0,gd=1/0,_d=null,vd=null,yd=0,bd=null,xd=null,Sd=0,Cd=0,wd=null,Td=null,Ed=null,Dd=null,Od=null,kd=0,Ad=null;function jd(){return K&2&&Y!==0?Y&-Y:j.T===null?yt():Pf()}function Md(){if(ld===0){if(!(Y&536870912)||V){var e=rt;rt<<=1,!(rt&3932160)&&(rt=262144),ld=e}else ld=536870912}return e=Ao.current,e!==null&&(e.flags|=32),ld}function Nd(e,t){if(t!=null){var n=e.stateNode,r=n.ref;r===null&&(r=n.ref=Pp(_i(e.memoizedProps,n))),Dd===null&&(Dd=[]),Dd.push(t.bind(null,r))}}function Pd(e,t,n){(e===q&&(X===2||X===9)||e.cancelPendingCommit!==null)&&(Vd(e,0),Rd(e,Y,ld,!1)),ft(e,n),(!(K&2)||e!==q)&&(e===q&&(!(K&2)&&(sd|=n),ad===4&&Rd(e,Y,ld,!1)),Ef(e))}function Fd(e,t,n){if(K&6)throw Error(i(327));var r=!n&&!(t&127)&&(t&e.expiredLanes)===0||st(e,t),a=r?Yd(e,t):qd(e,t,!0),o=r;do{if(a===0){nd&&!r&&Rd(e,t,0,!1);break}if(n=e.current.alternate,o&&!Ld(n)){a=qd(e,t,!1),o=!1;continue}if(a===2){if(o=t,e.errorRecoveryDisabledLanes&o)var s=0;else s=e.pendingLanes&-536870913,s=s===0?s&536870912?536870912:0:s;if(s!==0){t=s;a:{var c=e;a=dd;var l=c.current.memoizedState.isDehydrated;if(l&&(Vd(c,s).flags|=256),s=qd(c,s,!1),s!==2&&s!==6){if(rd&&!l){c.errorRecoveryDisabledLanes|=o,sd|=o,a=4;break a}o=fd,fd=a,o!==null&&(fd===null?fd=o:fd.push.apply(fd,o))}a=s}if(o=!1,a!==2)continue}}if(a===1){Vd(e,0),Rd(e,t,0,!0);break}a:{switch(r=e,o=a,o){case 0:case 1:throw Error(i(345));case 4:if((t&4194048)!==t&&(t&62914560)!==t)break;case 6:Rd(r,t,ld,!td);break a;case 2:fd=null;break;case 3:case 5:break;default:throw Error(i(329))}if((t&62914560)===t&&(a=md+300-Ve(),10<a)){if(Rd(r,t,ld,!td),ot(r,0,!0)!==0)break a;Sd=t,r.timeoutHandle=gp(Id.bind(null,r,n,fd,_d,pd,t,ld,sd,ud,td,o,`Throttled`,-0,0),a);break a}Id(r,n,fd,_d,pd,t,ld,sd,ud,td,o,null,-0,0)}break}while(1);Ef(e)}function Id(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.timeoutHandle=-1;var m=t.subtreeFlags,h=(a&335544064)===a;if(d=null,(h||m&8192||(m&16785408)==16785408)&&(d={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:vn},Bl=null,Ku(t,a,d),h&&(m=d,h=e.containerInfo,h=(h.nodeType===9?h:h.ownerDocument).__reactViewTransition,h!=null&&(m.count++,m.waitingForViewTransition=!0,m=nh.bind(m),h.finished.then(m,m))),m=(a&62914560)===a?md-Ve():(a&4194048)===a?hd-Ve():0,m=eh(d,m),m!==null)){Sd=a,e.cancelPendingCommit=m(nf.bind(null,e,t,a,n,r,i,o,s,c,l,u,d,null,f,p)),Rd(e,a,o,!l);return}nf(e,t,a,n,r,i,o,s,c,l,u,d)}function Ld(e){for(var t=e;;){var n=t.tag;if((n===0||n===11||n===15)&&t.flags&16384&&(n=t.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Vr(a(),i))return!1}catch{return!1}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Rd(e,t,n,r){t=ct(e,t),t&=~cd,t&=~sd,e.suspendedLanes|=t,e.pingedLanes&=~t,r&&(e.warmLanes|=t),r=e.expirationTimes;for(var i=t;0<i;){var a=31-Qe(i),o=1<<a;r[a]=-1,i&=~o}n!==0&&mt(e,n,t)}function zd(){return K&6?!0:(Df(0,!1),!1)}function Bd(){if(J!==null){if(X===0)var e=J.return;else e=J,ga=ha=null,is(e),io=null,ao=0,e=J;for(;e!==null;)bl(e.alternate,e),e=e.return;J=null}}function Vd(e,t){var n=e.timeoutHandle;return n!==-1&&(e.timeoutHandle=-1,_p(n)),n=e.cancelPendingCommit,n!==null&&(e.cancelPendingCommit=null,n()),Sd=0,Bd(),q=e,J=n=Pi(e.current,null),Y=t,X=0,ed=null,td=!1,nd=st(e,t),rd=!1,ud=ld=cd=sd=od=ad=0,fd=dd=null,pd=!1,id=ct(e,t),wi(),n}function Hd(e,t){H=null,j.H=mc,t===Ja||t===Xa?(t=no(),X=3):t===Ya?(t=no(),X=4):X=t===Mc?8:typeof t==`object`&&t&&typeof t.then==`function`?6:1,ed=t,J===null&&(ad=1,Ec(e,Hi(t,e.current)))}function Ud(){var e=Ao.current;return e===null?!0:(Y&4194048)===Y?jo===null:(Y&62914560)===Y||Y&536870912?e===jo:!1}function Wd(){var e=j.H;return j.H=mc,e===null?mc:e}function Gd(){var e=j.A;return j.A=Qu,e}function Kd(){ad=4,td||(Y&4194048)!==Y&&Ao.current!==null||(nd=!0),!(od&134217727)&&!(sd&134217727)||q===null||Rd(q,Y,ld,!1)}function qd(e,t,n){var r=K;K|=2;var i=Wd(),a=Gd();(q!==e||Y!==t)&&(_d=null,Vd(e,t)),t=!1;var o=ad;a:do try{if(X!==0&&J!==null){var s=J,c=ed;switch(X){case 8:Bd(),o=6;break a;case 3:case 2:case 9:case 6:Ao.current===null&&(t=!0);var l=X;if(X=0,ed=null,$d(e,s,c,l),n&&nd){o=0;break a}break;default:l=X,X=0,ed=null,$d(e,s,c,l)}}Jd(),o=ad;break}catch(t){Hd(e,t)}while(1);return t&&e.shellSuspendCounter++,ga=ha=null,K=r,j.H=i,j.A=a,J===null&&(q=null,Y=0,wi()),o}function Jd(){for(;J!==null;)Zd(J)}function Yd(e,t){var n=K;K|=2;var r=Wd(),a=Gd();q!==e||Y!==t?(_d=null,gd=Ve()+500,Vd(e,t)):nd=st(e,t);a:do try{if(X!==0&&J!==null){t=J;var o=ed;b:switch(X){case 1:X=0,ed=null,$d(e,t,o,1);break;case 2:case 9:if(Qa(o)){X=0,ed=null,Qd(t);break}t=function(){X!==2&&X!==9||q!==e||(X=7),Ef(e)},o.then(t,t);break a;case 3:X=7;break a;case 4:X=5;break a;case 7:Qa(o)?(X=0,ed=null,Qd(t)):(X=0,ed=null,$d(e,t,o,7));break;case 5:var s=null;switch(J.tag){case 26:s=J.memoizedState;case 5:case 27:var c=J;if(s?Ym(s):c.stateNode.complete){X=0,ed=null;var l=c.sibling;if(l!==null)J=l;else{var u=c.return;u===null?J=null:(J=u,ef(u))}break b}}X=0,ed=null,$d(e,t,o,5);break;case 6:X=0,ed=null,$d(e,t,o,6);break;case 8:Bd(),ad=6;break a;default:throw Error(i(462))}}Xd();break}catch(t){Hd(e,t)}while(1);return ga=ha=null,j.H=r,j.A=a,K=n,J===null?(q=null,Y=0,wi(),ad):0}function Xd(){for(;J!==null&&!ze();)Zd(J)}function Zd(e){var t=fl(e.alternate,e,id);e.memoizedProps=e.pendingProps,t===null?ef(e):J=t}function Qd(e){var t=e,n=t.alternate;switch(t.tag){case 15:case 0:t=Kc(n,t,t.pendingProps,t.type,void 0,Y);break;case 11:t=Kc(n,t,t.pendingProps,t.type.render,t.ref,Y);break;case 5:is(t);var r=t;r===ra&&(V?(la(r),r.tag===5&&r.stateNode!=null&&(B=r.stateNode)):(la(r),V=!0));default:bl(n,t),t=J=Fi(t,id),t=fl(n,t,id)}e.memoizedProps=e.pendingProps,t===null?ef(e):J=t}function $d(e,t,n,r){ga=ha=null,is(t),io=null,ao=0;var i=t.return;try{if(jc(e,i,t,n,Y)){ad=1,Ec(e,Hi(n,e.current)),J=null;return}}catch(t){if(i!==null)throw J=i,t;ad=1,Ec(e,Hi(n,e.current)),J=null;return}t.flags&32768?(V||r===1?e=!0:nd||Y&536870912?e=!1:(td=e=!0,(r===2||r===9||r===3||r===6)&&(r=Ao.current,r!==null&&r.tag===13&&(r.flags|=16384))),tf(t,e)):ef(t)}function ef(e){var t=e;do{if(t.flags&32768){tf(t,td);return}e=t.return;var n=vl(t.alternate,t,id);if(n!==null){J=n;return}if(t=t.sibling,t!==null){J=t;return}J=t=e}while(t!==null);ad===0&&(ad=5)}function tf(e,t){do{var n=yl(e.alternate,e);if(n!==null){n.flags&=32767,J=n;return}if(n=e.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!t&&(e=e.sibling,e!==null)){J=e;return}J=e=n}while(e!==null);ad=6,J=null}function nf(e,t,n,r,a,o,s,c,l,u,d,f){e.cancelPendingCommit=null;do df();while(yd!==0);if(K&6)throw Error(i(327));if(t!==null){if(t===e.current)throw Error(i(177));e===q&&(J=q=null,Y=0),xd=t,bd=e,Sd=n,wd=a,Td=r,rf(e,t,n,s,c,l,f)}}function rf(e,t,n,r,i,a,o){var s=t.lanes|t.childLanes;if(Cd=s,s|=Ci,pt(e,n,s,r,i,a),Dd=null,(n&335544064)===n?(Od=Fa(e),r=10262):(Od=null,r=10256),(t.subtreeFlags&r)!==0||(t.flags&r)!==0?(e.callbackNode=null,e.callbackPriority=0,yf(Ge,function(){return ff(),null})):(e.callbackNode=null,e.callbackPriority=0),zl=!1,r=!!(t.flags&13878),t.subtreeFlags&13878||r){r=j.T,j.T=null,i=M.p,M.p=2,a=K,K|=4;try{pu(e,t,n)}finally{K=a,M.p=i,j.T=r}}yd=1,zl?Ed=Mp(o,e.containerInfo,Od,sf,cf,of,lf,ff,af,null,null):(sf(),cf(),lf())}function af(e){if(yd!==0){var t=bd.onRecoverableError;t(e,{componentStack:null})}}function of(){yd===3&&(yd=0,Pu(xd,bd),yd=4)}function sf(){if(yd===1){yd=0;var e=bd,t=xd,n=Sd,r=!!(t.flags&13878);if(t.subtreeFlags&13878||r){r=j.T,j.T=null;var i=M.p;M.p=2;var a=K;K|=4;try{uu=du=!1,Au(t,e,n),n=cp;var o=qr(e.containerInfo),s=n.focusedElem,c=n.selectionRange;if(o!==s&&s&&s.ownerDocument&&Kr(s.ownerDocument.documentElement,s)){if(c!==null&&Jr(s)){var l=c.start,u=c.end;if(u===void 0&&(u=l),`selectionStart`in s)s.selectionStart=l,s.selectionEnd=Math.min(u,s.value.length);else{var d=s.ownerDocument||document,f=d&&d.defaultView||window;if(f.getSelection){var p=f.getSelection(),m=s.textContent.length,h=Math.min(c.start,m),g=c.end===void 0?h:Math.min(c.end,m);!p.extend&&h>g&&(o=g,g=h,h=o);var _=Gr(s,h),v=Gr(s,g);if(_&&v&&(p.rangeCount!==1||p.anchorNode!==_.node||p.anchorOffset!==_.offset||p.focusNode!==v.node||p.focusOffset!==v.offset)){var y=d.createRange();y.setStart(_.node,_.offset),p.removeAllRanges(),h>g?(p.addRange(y),p.extend(v.node,v.offset)):(y.setEnd(v.node,v.offset),p.addRange(y))}}}}for(d=[],p=s;p=p.parentNode;)p.nodeType===1&&d.push({element:p,left:p.scrollLeft,top:p.scrollTop});for(typeof s.focus==`function`&&s.focus(),s=0;s<d.length;s++){var b=d[s];b.element.scrollLeft=b.left,b.element.scrollTop=b.top}}gh=!!sp,cp=sp=null}finally{K=a,M.p=i,j.T=r}}e.current=t,yd=2}}function cf(){if(yd===2){yd=0;var e=bd,t=xd,n=!!(t.flags&8772);if(t.subtreeFlags&8772||n){n=j.T,j.T=null;var r=M.p;M.p=2;var i=K;K|=4;try{hu(e,t.alternate,t)}finally{K=i,M.p=r,j.T=n}}yd=3}}function lf(){if(yd===4||yd===3){yd=0;var e=Ed;Ed=null,Be();var t=bd,n=xd,r=Sd,i=Td,a=(r&335544064)===r?10262:10256;if((n.subtreeFlags&a)!==0||(n.flags&a)!==0?yd=5:(yd=0,xd=bd=null,uf(t,t.pendingLanes)),a=t.pendingLanes,a===0&&(vd=null),vt(r),n=n.stateNode,Xe&&typeof Xe.onCommitFiberRoot==`function`)try{Xe.onCommitFiberRoot(Ye,n,void 0,(n.current.flags&128)==128)}catch{}if(i!==null){n=j.T,a=M.p,M.p=2,j.T=null;try{for(var o=t.onRecoverableError,s=0;s<i.length;s++){var c=i[s];o(c.value,{componentStack:c.stack})}}finally{j.T=n,M.p=a}}if(i=Dd,o=Od,Od=null,i!==null&&(Dd=null,o===null&&(o=[]),e!==null))for(c=0;c<i.length;c++)n=(0,i[c])(o),n!==void 0&&e.finished.finally(n);Sd&3&&df(),Ef(t),a=t.pendingLanes,r&261930&&a&42?t===Ad?kd++:(kd=0,Ad=t):(kd=0,Ad=null),Df(0,!1)}}function uf(e,t){(e.pooledCacheLanes&=t)===0&&(t=e.pooledCache,t!=null&&(e.pooledCache=null,Ma(t)))}function df(){return Ed!==null&&(Ed.skipTransition(),Ed=null),sf(),cf(),lf(),ff()}function ff(){if(yd!==5)return!1;var e=bd,t=Cd;Cd=0;var n=vt(Sd),r=j.T,a=M.p;try{M.p=32>n?32:n,j.T=null,n=wd,wd=null;var o=bd,s=Sd;if(yd=0,xd=bd=null,Sd=0,K&6)throw Error(i(331));var c=K;if(K|=4,Yu(o.current),Vu(o,o.current,s,n),K=c,Df(0,!1),Xe&&typeof Xe.onPostCommitFiberRoot==`function`)try{Xe.onPostCommitFiberRoot(Ye,o)}catch{}return!0}finally{M.p=a,j.T=r,uf(e,t)}}function pf(e,t,n){t=Hi(n,t),t=Oc(e.stateNode,t,2),e=_o(e,t,2),e!==null&&(ft(e,2),Ef(e))}function Z(e,t,n){if(e.tag===3)pf(e,e,n);else for(;t!==null;){if(t.tag===3){pf(t,e,n);break}if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError==`function`||typeof r.componentDidCatch==`function`&&(vd===null||!vd.has(r))){e=Hi(n,e),n=kc(2),r=_o(t,n,2),r!==null&&(Ac(n,r,t,e),ft(r,2),Ef(r));break}}t=t.return}}function mf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new $u;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(rd=!0,i.add(n),e=hf.bind(null,e,t,n),t.then(e,e))}function hf(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),e.pingedLanes|=e.suspendedLanes&n,e.warmLanes&=~n,q===e&&(Y&n)===n&&(ad===4||ad===3&&(Y&62914560)===Y&&300>Ve()-md?K&2?cd|=n:Vd(e,0):cd|=n,ud===Y&&(ud=0)),Ef(e)}function gf(e,t){t===0&&(t=ut()),e=Di(e,t),e!==null&&(ft(e,t),Ef(e))}function _f(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),gf(e,n)}function vf(e,t){var n=0;switch(e.tag){case 31:case 13:var r=e.stateNode,a=e.memoizedState;a!==null&&(n=a.retryLane);break;case 19:r=e.stateNode;break;case 22:r=e.stateNode._retryCache;break;default:throw Error(i(314))}r!==null&&r.delete(t),gf(e,n)}function yf(e,t){return Le(e,t)}var bf=null,xf=null,Sf=!1,Cf=!1,wf=!1,Tf=0;function Ef(e){e!==xf&&e.next===null&&(xf===null?bf=xf=e:xf=xf.next=e),Cf=!0,Sf||(Sf=!0,Nf())}function Df(e,t){if(!wf&&Cf){wf=!0;do for(var n=!1,r=bf;r!==null;){if(!t){if(e!==0){var i=r.pendingLanes;if(i===0)var a=0;else{var o=r.suspendedLanes,s=r.pingedLanes;a=(1<<31-Qe(42|e)+1)-1,a&=i&~(o&~s),a=a&201326741?a&201326741|1:a?a|2:0}a!==0&&(n=!0,Mf(r,a))}else a=Y,a=ot(r,r===q?a:0,r.cancelPendingCommit!==null||r.timeoutHandle!==-1),!(a&3)||st(r,a)||(n=!0,Mf(r,a))}r=r.next}while(n);wf=!1}}function Of(){kf()}function kf(){Cf=Sf=!1;var e=0;Tf!==0&&hp()&&(e=Tf);for(var t=Ve(),n=null,r=bf;r!==null;){var i=r.next,a=Af(r,t);a===0?(r.next=null,n===null?bf=i:n.next=i,i===null&&(xf=n)):(n=r,(e!==0||a&3)&&(Cf=!0)),r=i}yd!==0&&yd!==5||Df(e,!1),Tf!==0&&(Tf=0)}function Af(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes&-62914561;0<a;){var o=31-Qe(a),s=1<<o,c=i[o];c===-1?((s&n)===0||(s&r)!==0)&&(i[o]=lt(s,t)):c<=t&&(e.expiredLanes|=s),a&=~s}if(t=q,n=Y,n=ot(e,e===t?n:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r=e.callbackNode,n===0||e===t&&(X===2||X===9)||e.cancelPendingCommit!==null)return r!==null&&r!==null&&Re(r),e.callbackNode=null,e.callbackPriority=0;if(!(n&3)||st(e,n)){if(t=n&-n,t===e.callbackPriority)return t;switch(r!==null&&Re(r),vt(n)){case 2:case 8:n=We;break;case 32:n=Ge;break;case 268435456:n=R;break;default:n=Ge}return r=jf.bind(null,e),n=Le(n,r),e.callbackPriority=t,e.callbackNode=n,t}return r!==null&&r!==null&&Re(r),e.callbackPriority=2,e.callbackNode=null,2}function jf(e,t){if(yd!==0&&yd!==5)return e.callbackNode=null,e.callbackPriority=0,null;var n=e.callbackNode;if(df()&&e.callbackNode!==n)return null;var r=Y;return r=ot(e,e===q?r:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r===0?null:(Fd(e,r,t),Af(e,Ve()),e.callbackNode!=null&&e.callbackNode===n?jf.bind(null,e):null)}function Mf(e,t){if(df())return null;Fd(e,t,!0)}function Nf(){bp(function(){K&6?Le(Ue,Of):kf()})}function Pf(){if(Tf===0){var e=Ra;e===0&&(e=nt,nt<<=1,!(nt&261888)&&(nt=256)),Tf=e}return Tf}function Ff(e){return e==null||typeof e==`symbol`||typeof e==`boolean`?null:typeof e==`function`?e:_n(e)}function If(e,t,n,r,i){if(t===`submit`&&n&&n.stateNode===i){var a=Ff((i[Ct]||null).action),o=r.submitter;o&&(t=(t=o[Ct]||null)?Ff(t.formAction):o.getAttribute(`formAction`),t!==null&&(a=t,o=null));var s=new zn(`action`,`action`,null,r,i);e.push({event:s,listeners:[{instance:null,listener:function(){if(r.defaultPrevented){if(Tf!==0){var e=new FormData(i,o);ec(n,{pending:!0,data:e,method:i.method,action:a},null,e)}}else typeof a==`function`&&(s.preventDefault(),e=new FormData(i,o),ec(n,{pending:!0,data:e,method:i.method,action:a},a,e))},currentTarget:i}]})}}for(var Lf=0;Lf<mi.length;Lf++){var Rf=mi[Lf];hi(Rf.toLowerCase(),`on`+(Rf[0].toUpperCase()+Rf.slice(1)))}hi(oi,`onAnimationEnd`),hi(si,`onAnimationIteration`),hi(ci,`onAnimationStart`),hi(`dblclick`,`onDoubleClick`),hi(`focusin`,`onFocus`),hi(`focusout`,`onBlur`),hi(li,`onTransitionRun`),hi(ui,`onTransitionStart`),hi(di,`onTransitionCancel`),hi(fi,`onTransitionEnd`),Vt(`onMouseEnter`,[`mouseout`,`mouseover`]),Vt(`onMouseLeave`,[`mouseout`,`mouseover`]),Vt(`onPointerEnter`,[`pointerout`,`pointerover`]),Vt(`onPointerLeave`,[`pointerout`,`pointerover`]),Bt(`onChange`,`change click focusin focusout input keydown keyup selectionchange`.split(` `)),Bt(`onSelect`,`focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange`.split(` `)),Bt(`onBeforeInput`,[`compositionend`,`keypress`,`textInput`,`paste`]),Bt(`onCompositionEnd`,`compositionend focusout keydown keypress keyup mousedown`.split(` `)),Bt(`onCompositionStart`,`compositionstart focusout keydown keypress keyup mousedown`.split(` `)),Bt(`onCompositionUpdate`,`compositionupdate focusout keydown keypress keyup mousedown`.split(` `));var zf=`abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting`.split(` `),Bf=new Set(`beforetoggle cancel close invalid load scroll scrollend toggle`.split(` `).concat(zf));function Vf(e,t){t=!!(t&4);for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;a:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],c=s.instance,l=s.currentTarget;if(s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){bi(e)}i.currentTarget=null,a=c}else for(o=0;o<r.length;o++){if(s=r[o],c=s.instance,l=s.currentTarget,s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){bi(e)}i.currentTarget=null,a=c}}}}function Q(e,t){var n=t[Tt];n===void 0&&(n=t[Tt]=new Set);var r=e+`__bubble`;n.has(r)||(Gf(t,e,2,!1),n.add(r))}function Hf(e,t,n){var r=0;t&&(r|=4),Gf(n,e,r,t)}var Uf=`_reactListening`+Math.random().toString(36).slice(2);function Wf(e){if(!e[Uf]){e[Uf]=!0,Rt.forEach(function(t){t!==`selectionchange`&&(Bf.has(t)||Hf(t,!1,e),Hf(t,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Uf]||(t[Uf]=!0,Hf(`selectionchange`,!1,t))}}function Gf(e,t,n,r){switch(Ch(t)){case 2:var i=_h;break;case 8:i=vh;break;default:i=yh}n=i.bind(null,t,n,e),i=void 0,!On||t!==`touchstart`&&t!==`touchmove`&&t!==`wheel`||(i=!0),r?i===void 0?e.addEventListener(t,n,!0):e.addEventListener(t,n,{capture:!0,passive:i}):i===void 0?e.addEventListener(t,n,!1):e.addEventListener(t,n,{passive:i})}function Kf(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)a:for(;;){if(r===null)return;var s=r.tag;if(s===3||s===4){var c=r.stateNode.containerInfo;if(c===i)break;if(s===4)for(s=r.return;s!==null;){var l=s.tag;if((l===3||l===4)&&s.stateNode.containerInfo===i)return;s=s.return}for(;c!==null;){if(s=Mt(c),s===null)return;if(l=s.tag,l===5||l===6||l===26||l===27){r=a=s;continue a}c=c.parentNode}}r=r.return}Tn(function(){var r=a,i=bn(n),s=[];a:{var c=pi.get(e);if(c!==void 0){var l=zn,u=e;switch(e){case`keypress`:if(Pn(n)===0)break a;case`keydown`:case`keyup`:l=rr;break;case`focusin`:u=`focus`,l=Jn;break;case`focusout`:u=`blur`,l=Jn;break;case`beforeblur`:case`afterblur`:l=Jn;break;case`click`:if(n.button===2)break a;case`auxclick`:case`dblclick`:case`mousedown`:case`mousemove`:case`mouseup`:case`mouseout`:case`mouseover`:case`contextmenu`:l=Kn;break;case`drag`:case`dragend`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`dragstart`:case`drop`:l=qn;break;case`touchcancel`:case`touchend`:case`touchmove`:case`touchstart`:l=or;break;case oi:case si:case ci:l=Yn;break;case fi:l=sr;break;case`scroll`:case`scrollend`:l=Vn;break;case`wheel`:l=cr;break;case`copy`:case`cut`:case`paste`:l=Xn;break;case`gotpointercapture`:case`lostpointercapture`:case`pointercancel`:case`pointerdown`:case`pointermove`:case`pointerout`:case`pointerover`:case`pointerup`:l=ir;break;case`submit`:l=ar;break;case`toggle`:case`beforetoggle`:l=lr}var d=!!(t&4),f=!d&&(e===`scroll`||e===`scrollend`),p=d?c===null?null:c+`Capture`:c;d=[];for(var m=r,h;m!==null;){var g=m;if(h=g.stateNode,g=g.tag,g!==5&&g!==26&&g!==27||h===null||p===null||(g=En(m,p),g!=null&&d.push(qf(m,g,h))),f)break;m=m.return}0<d.length&&(c=new l(c,u,null,n,i),s.push({event:c,listeners:d}))}}if(!(t&7)){a:{if(l=e===`mouseover`||e===`pointerover`,c=e===`mouseout`||e===`pointerout`,l&&n!==yn&&(u=n.relatedTarget||n.fromElement)&&(Mt(u)||u[wt]))break a;(c||l)&&(u=i.window===i?i:(l=i.ownerDocument)?l.defaultView||l.parentWindow:window,c?(l=n.relatedTarget||n.toElement,c=r,l=l?Mt(l):null,l!==null&&(f=o(l),d=l.tag,l!==f||d!==5&&d!==27&&d!==6)&&(l=null)):(c=null,l=r),c!==l&&(d=Kn,g=`onMouseLeave`,p=`onMouseEnter`,m=`mouse`,(e===`pointerout`||e===`pointerover`)&&(d=ir,g=`onPointerLeave`,p=`onPointerEnter`,m=`pointer`),f=c==null?u:Pt(c),h=l==null?u:Pt(l),u=new d(g,m+`leave`,c,n,i),u.target=f,u.relatedTarget=h,g=null,Mt(i)===r&&(d=new d(p,m+`enter`,l,n,i),d.target=h,d.relatedTarget=f,g=d),f=g,d=c&&l?C(c,l,Yf):null,c!==null&&Xf(s,u,c,d,!1),l!==null&&f!==null&&Xf(s,f,l,d,!0)))}a:{if(c=r?Pt(r):window,l=c.nodeName&&c.nodeName.toLowerCase(),l===`select`||l===`input`&&c.type===`file`)var _=kr;else if(Cr(c)){if(Ar)_=zr;else{_=Lr;var v=Ir}}else l=c.nodeName,!l||l.toLowerCase()!==`input`||c.type!==`checkbox`&&c.type!==`radio`?r&&mn(r.elementType)&&(_=kr):_=Rr;if(_&&=_(e,r)){wr(s,_,n,i);break a}v&&v(e,c,r)}switch(v=r?Pt(r):window,e){case`focusin`:(Cr(v)||v.contentEditable===`true`)&&(Xr=v,Zr=r,Qr=null);break;case`focusout`:Qr=Zr=Xr=null;break;case`mousedown`:$r=!0;break;case`contextmenu`:case`mouseup`:case`dragend`:$r=!1,ei(s,n,i);break;case`selectionchange`:if(Yr)break;case`keydown`:case`keyup`:ei(s,n,i)}var y;if(dr)b:{switch(e){case`compositionstart`:var b=`onCompositionStart`;break b;case`compositionend`:b=`onCompositionEnd`;break b;case`compositionupdate`:b=`onCompositionUpdate`;break b}b=void 0}else yr?_r(e,n)&&(b=`onCompositionEnd`):e===`keydown`&&n.keyCode===229&&(b=`onCompositionStart`);b&&(mr&&n.locale!==`ko`&&(yr||b!==`onCompositionStart`?b===`onCompositionEnd`&&yr&&(y=Nn()):(An=i,jn=`value`in An?An.value:An.textContent,yr=!0)),v=Jf(r,b),0<v.length&&(b=new Zn(b,e,null,n,i),s.push({event:b,listeners:v}),y?b.data=y:(y=vr(n),y!==null&&(b.data=y)))),(y=pr?br(e,n):xr(e,n))&&(b=Jf(r,`onBeforeInput`),0<b.length&&(v=new Zn(`onBeforeInput`,`beforeinput`,null,n,i),s.push({event:v,listeners:b}),v.data=y)),If(s,e,r,n,i)}Vf(s,t)})}function qf(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Jf(e,t){for(var n=t+`Capture`,r=[];e!==null;){var i=e,a=i.stateNode;if(i=i.tag,i!==5&&i!==26&&i!==27||a===null||(i=En(e,n),i!=null&&r.unshift(qf(e,i,a)),i=En(e,t),i!=null&&r.push(qf(e,i,a))),e.tag===3)return r;e=e.return}return[]}function Yf(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5&&e.tag!==27);return e||null}function Xf(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,c=s.alternate,l=s.stateNode;if(s=s.tag,c!==null&&c===r)break;s!==5&&s!==26&&s!==27||l===null||(c=l,i?(l=En(n,a),l!=null&&o.unshift(qf(n,l,c))):i||(l=En(n,a),l!=null&&o.push(qf(n,l,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Zf=/\r\n?/g,Qf=/\u0000|\uFFFD/g;function $f(e){return(typeof e==`string`?e:``+e).replace(Zf,`
`).replace(Qf,``)}function ep(e,t){return t=$f(t),$f(e)===t}function $(e,t,n,r,a,o){switch(n){case`children`:if(typeof r==`string`)t===`body`||t===`textarea`&&r===``||un(e,r);else if(typeof r==`number`||typeof r==`bigint`)t!==`body`&&un(e,``+r);else return;break;case`className`:Jt(e,`class`,r);break;case`tabIndex`:Jt(e,`tabindex`,r);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:Jt(e,n,r);break;case`style`:pn(e,r,o);return;case`data`:if(t!==`object`){Jt(e,`data`,r);break}case`src`:case`href`:if(r===``&&(t!==`a`||n!==`href`)){e.removeAttribute(n);break}if(r==null||typeof r==`function`||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=_n(r),e.setAttribute(n,r);break;case`action`:case`formAction`:if(typeof r==`function`){e.setAttribute(n,`javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')`);break}if(typeof o==`function`&&(n===`formAction`?(t!==`input`&&$(e,t,`name`,a.name,a,null),$(e,t,`formEncType`,a.formEncType,a,null),$(e,t,`formMethod`,a.formMethod,a,null),$(e,t,`formTarget`,a.formTarget,a,null)):($(e,t,`encType`,a.encType,a,null),$(e,t,`method`,a.method,a,null),$(e,t,`target`,a.target,a,null))),r==null||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=_n(r),e.setAttribute(n,r);break;case`onClick`:r!=null&&(e.onclick=vn);return;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`multiple`:e.multiple=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`muted`:e.muted=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`ref`:break;case`autoFocus`:break;case`xlinkHref`:if(r==null||typeof r==`function`||typeof r==`boolean`||typeof r==`symbol`){e.removeAttribute(`xlink:href`);break}n=_n(r),e.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,n);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`credentialless`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:r&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,``):e.removeAttribute(n);break;case`capture`:case`download`:!0===r?e.setAttribute(n,``):!1!==r&&r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`cols`:case`rows`:case`size`:case`span`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`&&!isNaN(r)&&1<=r?e.setAttribute(n,r):e.removeAttribute(n);break;case`rowSpan`:case`start`:r==null||typeof r==`function`||typeof r==`symbol`||isNaN(r)?e.removeAttribute(n):e.setAttribute(n,r);break;case`popover`:Q(`beforetoggle`,e),Q(`toggle`,e),qt(e,`popover`,r);break;case`xlinkActuate`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:actuate`,r);break;case`xlinkArcrole`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:arcrole`,r);break;case`xlinkRole`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:role`,r);break;case`xlinkShow`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:show`,r);break;case`xlinkTitle`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:title`,r);break;case`xlinkType`:Yt(e,`http://www.w3.org/1999/xlink`,`xlink:type`,r);break;case`xmlBase`:Yt(e,`http://www.w3.org/XML/1998/namespace`,`xml:base`,r);break;case`xmlLang`:Yt(e,`http://www.w3.org/XML/1998/namespace`,`xml:lang`,r);break;case`xmlSpace`:Yt(e,`http://www.w3.org/XML/1998/namespace`,`xml:space`,r);break;case`is`:qt(e,`is`,r);break;case`innerText`:case`textContent`:return;default:if(!(2<n.length)||n[0]!==`o`&&n[0]!==`O`||n[1]!==`n`&&n[1]!==`N`)n=hn.get(n)||n,qt(e,n,r);else return}z=!0}function tp(e,t,n,r,a,o){switch(n){case`style`:pn(e,r,o);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`children`:if(typeof r==`string`)un(e,r);else if(typeof r==`number`||typeof r==`bigint`)un(e,``+r);else return;break;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`onClick`:r!=null&&(e.onclick=vn);return;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`innerHTML`:case`ref`:return;case`innerText`:case`textContent`:return;default:if(!zt.hasOwnProperty(n))a:{if(n[0]===`o`&&n[1]===`n`&&(a=n.endsWith(`Capture`),o=n.slice(2,a?n.length-7:void 0),t=e[Ct]||null,t=t==null?null:t[n],typeof t==`function`&&e.removeEventListener(o,t,a),typeof r==`function`)){typeof t!=`function`&&t!==null&&(n in e?e[n]=null:e.hasAttribute(n)&&e.removeAttribute(n)),e.addEventListener(o,r,a);break a}z=!0,n in e?e[n]=r:!0===r?e.setAttribute(n,``):qt(e,n,r)}return}z=!0}function np(e,t,n){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`img`:Q(`error`,e),Q(`load`,e);var r=!1,a=!1,o;for(o in n)if(n.hasOwnProperty(o)){var s=n[o];if(s!=null)switch(o){case`src`:r=!0;break;case`srcSet`:a=!0;break;case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,o,s,n,null)}}a&&$(e,t,`srcSet`,n.srcSet,n,null),r&&$(e,t,`src`,n.src,n,null);return;case`input`:Q(`invalid`,e);var c=o=s=a=null,l=null,u=null;for(r in n)if(n.hasOwnProperty(r)){var d=n[r];if(d!=null)switch(r){case`name`:a=d;break;case`type`:s=d;break;case`checked`:l=d;break;case`defaultChecked`:u=d;break;case`value`:o=d;break;case`defaultValue`:c=d;break;case`children`:case`dangerouslySetInnerHTML`:if(d!=null)throw Error(i(137,t));break;default:$(e,t,r,d,n,null)}}an(e,o,c,l,u,s,a,!1);return;case`select`:for(a in Q(`invalid`,e),r=s=o=null,n)if(n.hasOwnProperty(a)&&(c=n[a],c!=null))switch(a){case`value`:o=c;break;case`defaultValue`:s=c;break;case`multiple`:r=c;default:$(e,t,a,c,n,null)}t=o,n=s,e.multiple=!!r,t==null?n!=null&&sn(e,!!r,n,!0):sn(e,!!r,t,!1);return;case`textarea`:for(s in Q(`invalid`,e),o=a=r=null,n)if(n.hasOwnProperty(s)&&(c=n[s],c!=null))switch(s){case`value`:r=c;break;case`defaultValue`:a=c;break;case`children`:o=c;break;case`dangerouslySetInnerHTML`:if(c!=null)throw Error(i(91));break;default:$(e,t,s,c,n,null)}ln(e,r,a,o);return;case`option`:for(l in n)if(n.hasOwnProperty(l)&&(r=n[l],r!=null))switch(l){case`selected`:e.selected=r&&typeof r!=`function`&&typeof r!=`symbol`;break;default:$(e,t,l,r,n,null)}return;case`dialog`:Q(`beforetoggle`,e),Q(`toggle`,e),Q(`cancel`,e),Q(`close`,e);break;case`iframe`:case`object`:Q(`load`,e);break;case`video`:case`audio`:for(r=0;r<zf.length;r++)Q(zf[r],e);break;case`image`:Q(`error`,e),Q(`load`,e);break;case`details`:Q(`toggle`,e);break;case`embed`:case`source`:case`link`:Q(`error`,e),Q(`load`,e);case`area`:case`base`:case`br`:case`col`:case`hr`:case`keygen`:case`meta`:case`param`:case`track`:case`wbr`:case`menuitem`:for(u in n)if(n.hasOwnProperty(u)&&(r=n[u],r!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,u,r,n,null)}return;default:if(mn(t)){for(d in n)n.hasOwnProperty(d)&&(r=n[d],r!==void 0&&tp(e,t,d,r,n,void 0));return}}for(c in n)n.hasOwnProperty(c)&&(r=n[c],r!=null&&$(e,t,c,r,n,null))}var rp={};function ip(e,t,n,r){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`input`:var a=null,o=null,s=null,c=null,l=null,u=null,d=null;for(m in n){var f=n[m];if(n.hasOwnProperty(m)&&f!=null)switch(m){case`checked`:break;case`value`:break;case`defaultValue`:l=f;default:r.hasOwnProperty(m)||$(e,t,m,null,r,f)}}for(var p in r){var m=r[p];if(f=n[p],r.hasOwnProperty(p)&&(m!=null||f!=null))switch(p){case`type`:m!==f&&(z=!0),o=m;break;case`name`:m!==f&&(z=!0),a=m;break;case`checked`:m!==f&&(z=!0),u=m;break;case`defaultChecked`:m!==f&&(z=!0),d=m;break;case`value`:m!==f&&(z=!0),s=m;break;case`defaultValue`:m!==f&&(z=!0),c=m;break;case`children`:case`dangerouslySetInnerHTML`:if(m!=null)throw Error(i(137,t));break;default:m!==f&&$(e,t,p,m,r,f)}}rn(e,s,c,l,u,d,o,a);return;case`select`:for(o in m=s=c=p=null,n)if(l=n[o],n.hasOwnProperty(o)&&l!=null)switch(o){case`value`:break;case`multiple`:m=l;default:r.hasOwnProperty(o)||$(e,t,o,null,r,l)}for(a in r)if(o=r[a],l=n[a],r.hasOwnProperty(a)&&(o!=null||l!=null))switch(a){case`value`:o!==l&&(z=!0),p=o;break;case`defaultValue`:o!==l&&(z=!0),c=o;break;case`multiple`:o!==l&&(z=!0),s=o;default:o!==l&&$(e,t,a,o,r,l)}t=c,n=s,r=m,p==null?!!r!=!!n&&(t==null?sn(e,!!n,n?[]:``,!1):sn(e,!!n,t,!0)):sn(e,!!n,p,!1);return;case`textarea`:for(c in m=p=null,n)if(a=n[c],n.hasOwnProperty(c)&&a!=null&&!r.hasOwnProperty(c))switch(c){case`value`:break;case`children`:break;default:$(e,t,c,null,r,a)}for(s in r)if(a=r[s],o=n[s],r.hasOwnProperty(s)&&(a!=null||o!=null))switch(s){case`value`:a!==o&&(z=!0),p=a;break;case`defaultValue`:a!==o&&(z=!0),m=a;break;case`children`:break;case`dangerouslySetInnerHTML`:if(a!=null)throw Error(i(91));break;default:a!==o&&$(e,t,s,a,r,o)}cn(e,p,m);return;case`option`:for(var h in n)if(p=n[h],n.hasOwnProperty(h)&&p!=null&&!r.hasOwnProperty(h))switch(h){case`selected`:e.selected=!1;break;default:$(e,t,h,null,r,p)}for(l in r)if(p=r[l],m=n[l],r.hasOwnProperty(l)&&p!==m&&(p!=null||m!=null))switch(l){case`selected`:p!==m&&(z=!0),e.selected=p&&typeof p!=`function`&&typeof p!=`symbol`;break;default:$(e,t,l,p,r,m)}return;case`img`:case`link`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:case`menuitem`:for(var g in n)p=n[g],n.hasOwnProperty(g)&&p!=null&&!r.hasOwnProperty(g)&&$(e,t,g,null,r,p);for(u in r)if(p=r[u],m=n[u],r.hasOwnProperty(u)&&p!==m&&(p!=null||m!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:if(p!=null)throw Error(i(137,t));break;default:$(e,t,u,p,r,m)}return;default:if(mn(t)){for(var _ in n)p=n[_],n.hasOwnProperty(_)&&p!==void 0&&!r.hasOwnProperty(_)&&tp(e,t,_,void 0,r,p);for(d in r)p=r[d],m=n[d],!r.hasOwnProperty(d)||p===m||p===void 0&&m===void 0||tp(e,t,d,p,r,m);return}}for(var v in n)p=n[v],n.hasOwnProperty(v)&&p!=null&&!r.hasOwnProperty(v)&&$(e,t,v,null,r,p);for(f in r)p=r[f],m=n[f],!r.hasOwnProperty(f)||p===m||p==null&&m==null||$(e,t,f,p,r,m)}function ap(e){switch(e){case`css`:case`script`:case`font`:case`img`:case`image`:case`input`:case`link`:return!0;default:return!1}}function op(){if(typeof performance.getEntriesByType==`function`){for(var e=0,t=0,n=performance.getEntriesByType(`resource`),r=0;r<n.length;r++){var i=n[r],a=i.transferSize,o=i.initiatorType,s=i.duration;if(a&&s&&ap(o)){for(o=0,s=i.responseEnd,r+=1;r<n.length;r++){var c=n[r],l=c.startTime;if(l>s)break;var u=c.transferSize,d=c.initiatorType;u&&ap(d)&&(c=c.responseEnd,o+=u*(c<s?1:(s-l)/(c-l)))}if(--r,t+=8*(a+o)/(i.duration/1e3),e++,10<e)break}}if(0<e)return t/e/1e6}return navigator.connection&&(e=navigator.connection.downlink,typeof e==`number`)?e:5}var sp=null,cp=null;function lp(e){return e.nodeType===9?e:e.ownerDocument}function up(e){switch(e){case`http://www.w3.org/2000/svg`:return 1;case`http://www.w3.org/1998/Math/MathML`:return 2;default:return 0}}function dp(e,t){if(e===0)switch(t){case`svg`:return 1;case`math`:return 2;default:return 0}return e===1&&t===`foreignObject`?0:e}function fp(e,t,n,r){return n=lp(n).createElement(e),n[St]=r,n[Ct]=t,np(n,e,t),It(n),n}function pp(e,t){return e===`textarea`||e===`noscript`||typeof t.children==`string`||typeof t.children==`number`||typeof t.children==`bigint`||typeof t.dangerouslySetInnerHTML==`object`&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var mp=null;function hp(){var e=window.event;return e&&e.type===`popstate`?e!==mp&&(mp=e,!0):(mp=null,!1)}var gp=typeof setTimeout==`function`?setTimeout:void 0,_p=typeof clearTimeout==`function`?clearTimeout:void 0,vp=typeof Promise==`function`?Promise:void 0,yp=typeof requestAnimationFrame==`function`?requestAnimationFrame:gp,bp=typeof queueMicrotask==`function`?queueMicrotask:vp===void 0?gp:function(e){return vp.resolve(null).then(e).catch(xp)};function xp(e){setTimeout(function(){throw e})}function Sp(e){return e===`head`}function Cp(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8){if(n=i.data,n===`/$`||n===`/&`){if(r===0){e.removeChild(i),Hh(t);return}r--}else if(n===`$`||n===`$?`||n===`$~`||n===`$!`||n===`&`)r++;else if(n===`html`)_m(e.ownerDocument.documentElement);else if(n===`head`){n=e.ownerDocument.head,_m(n);for(var a=n.firstChild;a;){var o=a.nextSibling,s=a.nodeName;a[kt]||s===`SCRIPT`||s===`STYLE`||s===`LINK`&&a.rel.toLowerCase()===`stylesheet`||n.removeChild(a),a=o}}else n===`body`&&_m(e.ownerDocument.body)}n=i}while(n);Hh(t)}function wp(e,t){var n=e;e=0;do{var r=n.nextSibling;if(n.nodeType===1?t?(n._stashedDisplay=n.style.display,n.style.display=`none`):(n.style.display=n._stashedDisplay||``,n.getAttribute(`style`)===``&&n.removeAttribute(`style`)):n.nodeType===3&&(t?(n._stashedText=n.nodeValue,n.nodeValue=``):n.nodeValue=n._stashedText||``),r&&r.nodeType===8){if(n=r.data,n===`/$`){if(e===0)break;e--}else n!==`$`&&n!==`$?`&&n!==`$~`&&n!==`$!`||e++}n=r}while(n)}function Tp(e,t,n){if(t=CSS.escape(t)===t?t:`r-`+btoa(t).replace(/=/g,``),e.style.viewTransitionName=t,n!=null&&(e.style.viewTransitionClass=n),n=getComputedStyle(e),n.display===`inline`){if(t=e.getClientRects(),t.length===1)var r=1;else for(var i=r=0;i<t.length;i++){var a=t[i];0<a.width&&0<a.height&&r++}r===1&&(e=e.style,e.display=t.length===1?`inline-block`:`block`,e.marginTop=`-`+n.paddingTop,e.marginBottom=`-`+n.paddingBottom)}}function Ep(e,t){e=e.style,t=t.style;var n=t==null?null:t.hasOwnProperty(`viewTransitionName`)?t.viewTransitionName:t.hasOwnProperty(`view-transition-name`)?t[`view-transition-name`]:null;e.viewTransitionName=n==null||typeof n==`boolean`?``:(``+n).trim(),n=t==null?null:t.hasOwnProperty(`viewTransitionClass`)?t.viewTransitionClass:t.hasOwnProperty(`view-transition-class`)?t[`view-transition-class`]:null,e.viewTransitionClass=n==null||typeof n==`boolean`?``:(``+n).trim(),e.display===`inline-block`&&(t==null?e.display=e.margin=``:(n=t.display,e.display=n==null||typeof n==`boolean`?``:n,n=t.margin,n==null?(n=t.hasOwnProperty(`marginTop`)?t.marginTop:t[`margin-top`],e.marginTop=n==null||typeof n==`boolean`?``:n,t=t.hasOwnProperty(`marginBottom`)?t.marginBottom:t[`margin-bottom`],e.marginBottom=t==null||typeof t==`boolean`?``:t):e.margin=n))}function Dp(e,t,n){return n=n.ownerDocument.defaultView,{rect:e,abs:t.position===`absolute`||t.position===`fixed`,clip:t.clipPath!==`none`||t.overflow!==`visible`||t.filter!==`none`||t.mask!==`none`||t.mask!==`none`||t.borderRadius!==`0px`,view:0<=e.bottom&&0<=e.right&&e.top<=n.innerHeight&&e.left<=n.innerWidth}}function Op(e){return Dp(e.getBoundingClientRect(),getComputedStyle(e),e)}function kp(e){var t=e.getBoundingClientRect();t=new DOMRect(t.x+2e4,t.y+2e4,t.width,t.height);var n=getComputedStyle(e);return Dp(t,n,e)}function Ap(e){return e.documentElement.clientHeight}function jp(e){this.addEventListener(`load`,e),this.addEventListener(`error`,e)}function Mp(e,t,n,r,i,a,o,s,c){var l=t.nodeType===9?t:t.ownerDocument;try{var u=l.startViewTransition({update:function(){var t=l.defaultView,n=t.navigation&&t.navigation.transition,o=l.fonts.status;r();var s=[];if(o===`loaded`&&(Ap(l),l.fonts.status===`loading`&&s.push(l.fonts.ready)),o=s.length,e!==null)for(var c=e.suspenseyImages,u=0,d=0;d<c.length;d++){var f=c[d];if(!f.complete){var p=f.getBoundingClientRect();if(0<p.bottom&&0<p.right&&p.top<t.innerHeight&&p.left<t.innerWidth){if(u+=Xm(f),u>$m){s.length=o;break}f=new Promise(jp.bind(f)),s.push(f)}}}if(0<s.length)return t=Promise.race([Promise.all(s),new Promise(function(e){return setTimeout(e,500)})]).then(i,i),(n?Promise.allSettled([n.finished,t]):t).then(a,a);if(i(),n)return n.finished.then(a,a);a()},types:n});l.__reactViewTransition=u;var d=[];return u.ready.then(function(){for(var e=l.documentElement.getAnimations({subtree:!0}),t=0;t<e.length;t++){var n=e[t],r=n.effect,i=r.pseudoElement;if(i!=null&&i.startsWith(`::view-transition`)){d.push(n),n=r.getKeyframes();for(var a=i=void 0,s=!0,c=0;c<n.length;c++){var u=n[c],f=u.width;if(i===void 0)i=f;else if(i!==f){s=!1;break}if(f=u.height,a===void 0)a=f;else if(a!==f){s=!1;break}delete u.width,delete u.height,u.transform===`none`&&delete u.transform}s&&i!==void 0&&a!==void 0&&(r.setKeyframes(n),s=getComputedStyle(r.target,r.pseudoElement),s.width!==i||s.height!==a)&&(s=n[0],s.width=i,s.height=a,s=n[n.length-1],s.width=i,s.height=a,r.setKeyframes(n))}}o()},function(e){l.__reactViewTransition===u&&(l.__reactViewTransition=null);try{if(typeof e==`object`&&e)switch(e.name){case`InvalidStateError`:(e.message===`View transition was skipped because document visibility state is hidden.`||e.message===`Skipping view transition because document visibility state has become hidden.`||e.message===`Skipping view transition because viewport size changed.`||e.message===`Transition was aborted because of invalid state`)&&(e=null)}e!==null&&c(e)}finally{r(),i(),o()}}),u.finished.finally(function(){for(var e=0;e<d.length;e++)d[e].cancel();l.__reactViewTransition===u&&(l.__reactViewTransition=null),s()}),u}catch{return r(),i(),o(),null}}function Np(e,t){this._scope=document.documentElement,this._selector=`::view-transition-`+e+`(`+t+`)`}Np.prototype.animate=function(e,t){return t=typeof t==`number`?{duration:t}:w({},t),t.pseudoElement=this._selector,this._scope.animate(e,t)},Np.prototype.getAnimations=function(){for(var e=this._scope,t=this._selector,n=e.getAnimations({subtree:!0}),r=[],i=0;i<n.length;i++){var a=n[i].effect;a!==null&&a.target===e&&a.pseudoElement===t&&r.push(n[i])}return r},Np.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function Pp(e){return{name:e,group:new Np(`group`,e),imagePair:new Np(`image-pair`,e),old:new Np(`old`,e),new:new Np(`new`,e)}}function Fp(e){this._fragmentFiber=e,this._observers=this._eventListeners=null}Fp.prototype.addEventListener=function(e,t,n){var r=null,i=null;if(!(n!=null&&typeof n!=`boolean`&&(r=n.signal||null,r!==null&&r.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var a=this._eventListeners;if(Bp(a,e,t,n)===-1){var o=this,s=t;n!=null&&typeof n!=`boolean`&&!0===n.once&&(s=function(r){o.removeEventListener(e,t,n),typeof t==`function`?t.call(this,r):t.handleEvent(r)}),r!==null&&(i=o.removeEventListener.bind(o,e,t,n),r.addEventListener(`abort`,i,{once:!0}),i=r.removeEventListener.bind(r,`abort`,i)),r=Rp(n),a.push({type:e,listener:t,optionsOrUseCapture:n,attachedListener:s,cleanup:i}),h(this._fragmentFiber.child,!1,Ip,e,s,r)}this._eventListeners=a}};function Ip(e,t,n,r){return b(e).addEventListener(t,n,r),!1}Fp.prototype.removeEventListener=function(e,t,n){var r=this._eventListeners;if(r!==null&&(t=Bp(r,e,t,n),t!==-1)){var i=r[t];n=i.attachedListener;var a=i.cleanup;i=Rp(i.optionsOrUseCapture),h(this._fragmentFiber.child,!1,Lp,e,n,i),r.splice(t,1),a!==null&&a()}};function Lp(e,t,n,r){return b(e).removeEventListener(t,n,r),!1}function Rp(e){return e!=null&&typeof e!=`boolean`&&(!0===e.once||e.signal instanceof AbortSignal)?{capture:e.capture,passive:e.passive}:e}function zp(e){return e==null?`c=0`:typeof e==`boolean`?`c=`+(e?`1`:`0`):`c=`+(e.capture?`1`:`0`)}function Bp(e,t,n,r){if(e.length===0)return-1;r=zp(r);for(var i=0;i<e.length;i++){var a=e[i];if(a.type===t&&a.listener===n&&zp(a.optionsOrUseCapture)===r)return i}return-1}Fp.prototype.dispatchEvent=function(e){var t=g(this._fragmentFiber);if(t===null)return!0;t=b(t);var n=this._eventListeners;if(n!==null&&0<n.length||!e.bubbles){var r=t.nodeType===9?t.createComment(``):document.createTextNode(``);if(n)for(var i=0;i<n.length;i++){var a=n[i];r.addEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture))}if(t.appendChild(r),e=r.dispatchEvent(e),n)for(i=0;i<n.length;i++)a=n[i],r.removeEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture));return t.removeChild(r),e}return t.dispatchEvent(e)},Fp.prototype.focus=function(e){h(this._fragmentFiber.child,!0,Vp,e,void 0,void 0)};function Vp(e,t){return e.tag!==6&&(e=b(e),pm(e,t))}Fp.prototype.focusLast=function(e){var t=[];h(this._fragmentFiber.child,!0,Hp,t,void 0,void 0);for(var n=t.length-1;0<=n&&!Vp(t[n],e);n--);};function Hp(e,t){return t.push(e),!1}Fp.prototype.blur=function(){var e=g(this._fragmentFiber);e!==null&&(e=b(e),e=lp(e).activeElement,e!==null&&h(this._fragmentFiber.child,!1,Up,e,void 0,void 0))};function Up(e,t){return e.tag!==6&&(e=b(e),e===t||e.contains(t)?(t.blur(),!0):!1)}Fp.prototype.observeUsing=function(e){this._observers===null&&(this._observers=new Set),this._observers.add(e),h(this._fragmentFiber.child,!1,Wp,e,void 0,void 0)};function Wp(e,t){return e.tag!==6&&(e=b(e),t.observe(e),!1)}Fp.prototype.unobserveUsing=function(e){var t=this._observers;if(t!==null&&t.has(e)){t.delete(e),h(this._fragmentFiber.child,!1,Gp,e,void 0,void 0);for(var n=t=0;n<Kp.length;n++){var r=Kp[n];r.fragmentInstance===this&&r.observer===e?e.unobserve(r.instance):Kp[t++]=r}Kp.length=t}};function Gp(e,t){return e.tag!==6&&(e=b(e),t.unobserve(e),!1)}var Kp=[],qp=!1;function Jp(e,t,n){Kp.push({fragmentInstance:e,observer:t,instance:n}),qp||(qp=!0,mm(function(){qp=!1;var e=Kp;Kp=[];for(var t=0;t<e.length;t++){var n=e[t];n.observer.unobserve(n.instance)}}))}Fp.prototype.getClientRects=function(){var e=[];return h(this._fragmentFiber.child,!1,Yp,e,void 0,void 0),e};function Yp(e,t){if(e.tag===6){e=e.stateNode;var n=e.ownerDocument.createRange();n.selectNodeContents(e),t.push.apply(t,n.getClientRects())}else e=b(e),t.push.apply(t,e.getClientRects());return!1}Fp.prototype.getRootNode=function(e){var t=g(this._fragmentFiber);return t===null?this:b(t).getRootNode(e)},Fp.prototype.compareDocumentPosition=function(e){var t=g(this._fragmentFiber);if(t===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];h(this._fragmentFiber.child,!1,Hp,n,void 0,void 0);var r=b(t);if(n.length===0){if(n=r,_(this._fragmentFiber)){a:{for(t=this._fragmentFiber.return;t!==null;){if(t.tag===4){t=t.stateNode.containerInfo;break a}if(t.tag===3||t.tag===5||t.tag===27)break;t=t.return}t=null}t!=null&&(n=t)}t=this._fragmentFiber;var i=r=n.compareDocumentPosition(e);return n===e?i=Node.DOCUMENT_POSITION_CONTAINS:r&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=v(t)[1],n===null?i=Node.DOCUMENT_POSITION_PRECEDING:(e=b(n).compareDocumentPosition(e),i=e===0||e&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),i|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}t=b(n[0]),i=b(n[n.length-1]);var a=_(this._fragmentFiber)?t.parentElement:r;if(a==null)return Node.DOCUMENT_POSITION_DISCONNECTED;r=a.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_CONTAINED_BY,a=a.compareDocumentPosition(i)&Node.DOCUMENT_POSITION_CONTAINED_BY;var o=t.compareDocumentPosition(e),s=i.compareDocumentPosition(e),c=o&Node.DOCUMENT_POSITION_CONTAINED_BY||s&Node.DOCUMENT_POSITION_CONTAINED_BY;return s=r&&a&&o&Node.DOCUMENT_POSITION_FOLLOWING&&s&Node.DOCUMENT_POSITION_PRECEDING,t=r&&t===e||a&&i===e||c||s?Node.DOCUMENT_POSITION_CONTAINED_BY:!r&&t===e||!a&&i===e?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:o,t&Node.DOCUMENT_POSITION_DISCONNECTED||t&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Xp(t,this._fragmentFiber,n[0],n[n.length-1],e)?t:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Xp(e,t,n,r,i){var a=Mt(i);if(e&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!a)a:{for(;a!==null;){if(a.tag===7&&(a===t||a.alternate===t)){n=!0;break a}a=a.return}n=!1}return n}if(e&Node.DOCUMENT_POSITION_CONTAINS){if(a===null)return a=i.ownerDocument,i===a||i===a.documentElement||i===a.body;a:{for(a=t,t=g(t);a!==null;){if(!(a.tag!==5&&a.tag!==3&&a.tag!==27||a!==t&&a.alternate!==t)){a=!0;break a}a=a.return}a=!1}return a}return e&Node.DOCUMENT_POSITION_PRECEDING?((t=!!a)&&!(t=a===n)&&(t=C(n,a,S),t===null?t=!1:(h(t,!0,te,a,n),a=x,x=null,t=a!==null)),t):e&Node.DOCUMENT_POSITION_FOLLOWING?((t=!!a)&&!(t=a===r)&&(t=C(r,a,S),t===null?t=!1:(h(t,!0,ne,a,r),a=x,ee=x=null,t=a!==null)),t):!1}function Zp(e,t){var n=e.ownerDocument.createRange();n.selectNodeContents(e),e=n.getBoundingClientRect(),window.scrollTo(window.scrollX+e.left,t?window.scrollY+e.top:window.scrollY+e.bottom-window.innerHeight)}Fp.prototype.scrollIntoView=function(e){if(typeof e==`object`)throw Error(i(566));var t=[];h(this._fragmentFiber.child,!1,Hp,t,void 0,void 0);var n=!1!==e;if(t.length===0){var r=v(this._fragmentFiber);if(r=n?r[1]||r[0]||g(this._fragmentFiber):r[0]||r[1],r===null)return;if(r.tag===6){e=b(r),Zp(e,n);return}if(r=b(r),r.nodeType!==9){if(r.nodeType===11){n=`host`in r?r.host:null,n!==null&&n.scrollIntoView(e);return}r.scrollIntoView(e)}}for(r=n?t.length-1:0;r!==(n?-1:t.length);){var a=t[r];a.tag===6?(a=b(a),Zp(a,n)):b(a).scrollIntoView(e),r+=n?-1:1}};function Qp(e,t){return e=b(e),$p(e,t),!1}function $p(e,t){e.reactFragments??=new Set,e.reactFragments.add(t)}function em(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.addEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){for(var r=0,i=0;i<Kp.length;i++){var a=Kp[i];(a.fragmentInstance!==t||a.observer!==n||a.instance!==e)&&(Kp[r++]=a)}Kp.length=r,n.observe(e)}),$p(e,t))}function tm(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.removeEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){typeof n.rootMargin==`string`?Jp(t,n,e):n.unobserve(e)}),e.reactFragments!=null&&e.reactFragments.delete(t))}function nm(e){var t=e.firstChild;for(t&&t.nodeType===10&&(t=t.nextSibling);t;){var n=t;switch(t=t.nextSibling,n.nodeName){case`HTML`:case`HEAD`:case`BODY`:nm(n),jt(n);continue;case`SCRIPT`:case`STYLE`:continue;case`LINK`:if(n.rel.toLowerCase()===`stylesheet`)continue}e.removeChild(n)}}function rm(e,t,n,r){for(;e.nodeType===1;){var i=n;if(e.nodeName.toLowerCase()!==t.toLowerCase()){if(!r&&(e.nodeName!==`INPUT`||e.type!==`hidden`))break}else if(!r){if(t===`input`&&e.type===`hidden`){var a=i.name==null?null:``+i.name;if(i.type===`hidden`&&e.getAttribute(`name`)===a)return e}else return e}else if(!e[kt])switch(t){case`meta`:if(!e.hasAttribute(`itemprop`))break;return e;case`link`:if(a=e.getAttribute(`rel`),a===`stylesheet`&&e.hasAttribute(`data-precedence`)||a!==i.rel||e.getAttribute(`href`)!==(i.href==null||i.href===``?null:i.href)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin)||e.getAttribute(`title`)!==(i.title==null?null:i.title))break;return e;case`style`:if(e.hasAttribute(`data-precedence`))break;return e;case`script`:if(a=e.getAttribute(`src`),(a!==(i.src==null?null:i.src)||e.getAttribute(`type`)!==(i.type==null?null:i.type)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin))&&a&&e.hasAttribute(`async`)&&!e.hasAttribute(`itemprop`))break;return e;default:return e}if(e=lm(e.nextSibling),e===null)break}return null}function im(e,t,n){if(t===``)return null;for(;e.nodeType!==3;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!n||(e=lm(e.nextSibling),e===null))return null;return e}function am(e,t){for(;e.nodeType!==8;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!t||(e=lm(e.nextSibling),e===null))return null;return e}function om(e){return e.data===`$?`||e.data===`$~`}function sm(e){return e.data===`$!`||e.data===`$?`&&e.ownerDocument.readyState!==`loading`}function cm(e,t){var n=e.ownerDocument;if(e.data===`$~`)e._reactRetry=t;else if(e.data!==`$?`||n.readyState!==`loading`)t();else{var r=function(){t(),n.removeEventListener(`DOMContentLoaded`,r)};n.addEventListener(`DOMContentLoaded`,r),e._reactRetry=r}}function lm(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t===`$`||t===`$!`||t===`$?`||t===`$~`||t===`&`||t===`F!`||t===`F`)break;if(t===`/$`||t===`/&`)return null}}return e}var um=null;function dm(e){e=e.nextSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`/$`||n===`/&`){if(t===0)return lm(e.nextSibling);t--}else n!==`$`&&n!==`$!`&&n!==`$?`&&n!==`$~`&&n!==`&`||t++}e=e.nextSibling}return null}function fm(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`$`||n===`$!`||n===`$?`||n===`$~`||n===`&`){if(t===0)return e;t--}else n!==`/$`&&n!==`/&`||t++}e=e.previousSibling}return null}function pm(e,t){function n(){r=!0}if(e.ownerDocument.activeElement===e)return!0;var r=!1;try{e.ownerDocument.addEventListener(`focus`,n,!0),(e.focus||HTMLElement.prototype.focus).call(e,t)}finally{e.ownerDocument.removeEventListener(`focus`,n,!0)}return r}function mm(e){yp(function(){yp(function(t){return e(t)})})}function hm(e,t,n){switch(t=lp(n),e){case`html`:if(e=t.documentElement,!e)throw Error(i(452));return e;case`head`:if(e=t.head,!e)throw Error(i(453));return e;case`body`:if(e=t.body,!e)throw Error(i(454));return e;default:throw Error(i(451))}}function gm(e,t,n){for(var r in n){var i=n[r];n.hasOwnProperty(r)&&i!=null&&$(e,t,r,null,rp,i)}n.dangerouslySetInnerHTML!=null&&(e.textContent=``),e.onclick===vn&&(e.onclick=null),jt(e)}function _m(e){for(var t=e.attributes;t.length;)e.removeAttributeNode(t[0]);jt(e)}var vm=new Map,ym=new Set;function bm(e){if(typeof e.getRootNode==`function`){var t=e.getRootNode();if(t.nodeType===9||t.nodeType===11)return t}return e.nodeType===9?e:e.ownerDocument}var xm=M.d;M.d={f:Sm,r:Cm,D:Em,C:Dm,L:Om,m:km,X:jm,S:Am,M:Mm};function Sm(){var e=xm.f(),t=zd();return e||t}function Cm(e){var t=Nt(e);t!==null&&t.tag===5&&t.type===`form`?nc(t):xm.r(e)}var wm=typeof document>`u`?null:document;function Tm(e,t,n){var r=wm;if(r&&typeof t==`string`&&t){var i=nn(t);i=`link[rel="`+e+`"][href="`+i+`"]`,typeof n==`string`&&(i+=`[crossorigin="`+n+`"]`),ym.has(i)||(ym.add(i),e={rel:e,crossOrigin:n,href:t},r.querySelector(i)===null&&(t=r.createElement(`link`),np(t,`link`,e),It(t),r.head.appendChild(t)))}}function Em(e){xm.D(e),Tm(`dns-prefetch`,e,null)}function Dm(e,t){xm.C(e,t),Tm(`preconnect`,e,t)}function Om(e,t,n){xm.L(e,t,n);var r=wm;if(r&&e&&t){var i=`link[rel="preload"][as="`+nn(t)+`"]`;t===`image`&&n&&n.imageSrcSet?(i+=`[imagesrcset="`+nn(n.imageSrcSet)+`"]`,typeof n.imageSizes==`string`&&(i+=`[imagesizes="`+nn(n.imageSizes)+`"]`)):i+=`[href="`+nn(e)+`"]`;var a=i;switch(t){case`style`:a=Pm(e);break;case`script`:a=Rm(e)}if(!(vm.has(a)||(e=w({rel:`preload`,href:t===`image`&&n&&n.imageSrcSet?void 0:e,as:t},n),vm.set(a,e),r.querySelector(i)!==null||t===`style`&&r.querySelector(Fm(a))||t===`script`&&r.querySelector(zm(a))))){var o=r.createElement(`link`);np(o,`link`,e),t===`style`&&(o[At]=!0,o.onload=o.onerror=function(){Lt(o)}),It(o),r.head.appendChild(o)}}}function km(e,t){xm.m(e,t);var n=wm;if(n&&e){var r=t&&typeof t.as==`string`?t.as:`script`,i=`link[rel="modulepreload"][as="`+nn(r)+`"][href="`+nn(e)+`"]`,a=i;switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:a=Rm(e)}if(!vm.has(a)&&(e=w({rel:`modulepreload`,href:e},t),vm.set(a,e),n.querySelector(i)===null)){switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:if(n.querySelector(zm(a)))return}r=n.createElement(`link`),np(r,`link`,e),It(r),n.head.appendChild(r)}}}function Am(e,t,n){xm.S(e,t,n);var r=wm;if(r&&e){var i=Ft(r).hoistableStyles,a=Pm(e);t||=`default`;var o=i.get(a);if(!o){var s={loading:0,preload:null};if(o=r.querySelector(Fm(a)))s.loading=5;else{e=w({rel:`stylesheet`,href:e,"data-precedence":t},n),(n=vm.get(a))&&Hm(e,n);var c=o=r.createElement(`link`);It(c),np(c,`link`,e),c._p=new Promise(function(e,t){c.onload=e,c.onerror=t}),c.addEventListener(`load`,function(){s.loading|=1}),c.addEventListener(`error`,function(){s.loading|=2}),s.loading|=4,Vm(o,t,r)}o={type:`stylesheet`,instance:o,count:1,state:s},i.set(a,o)}}}function jm(e,t){xm.X(e,t);var n=wm;if(n&&e){var r=Ft(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=w({src:e,async:!0},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),It(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Mm(e,t){xm.M(e,t);var n=wm;if(n&&e){var r=Ft(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=w({src:e,async:!0,type:`module`},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),It(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Nm(e,t,n,r){var a=(a=Ce.current)?bm(a):null;if(!a)throw Error(i(446));switch(e){case`meta`:case`title`:return null;case`style`:return typeof n.precedence==`string`&&typeof n.href==`string`?(n=Pm(n.href),t=Ft(a).hoistableStyles,r=t.get(n),r||(r={type:`style`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};case`link`:if(n.rel===`stylesheet`&&typeof n.href==`string`&&typeof n.precedence==`string`){e=Pm(n.href);var o=Ft(a).hoistableStyles,s=o.get(e);if(s||(a=a.ownerDocument||a,s={type:`stylesheet`,instance:null,count:0,state:{loading:0,preload:null}},o.set(e,s),(o=a.querySelector(Fm(e)))?o._p||(s.instance=o,s.state.loading=5):(o=vm.get(e),o||(o={rel:`preload`,as:`style`,href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},vm.set(e,o)),Lm(a,e,o,s.state))),t&&r===null)throw Error(i(528,``));return s}if(t&&r!==null)throw Error(i(529,``));return null;case`script`:return t=n.async,n=n.src,typeof n==`string`&&t&&typeof t!=`function`&&typeof t!=`symbol`?(n=Rm(n),t=Ft(a).hoistableScripts,r=t.get(n),r||(r={type:`script`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};default:throw Error(i(444,e))}}function Pm(e){return`href="`+nn(e)+`"`}function Fm(e){return`link[rel="stylesheet"][`+e+`]`}function Im(e){return w({},e,{"data-precedence":e.precedence,precedence:null})}function Lm(e,t,n,r){if(t=e.querySelector(`link[rel="preload"][as="style"][`+t+`]`)){if(!0!==t[At]){r.loading=1;return}}else t=e.createElement(`link`),t[At]=!0,t.onload=t.onerror=Lt.bind(null,t),np(t,`link`,n),It(t),e.head.appendChild(t);r.preload=t,t.addEventListener(`load`,function(){return r.loading|=1}),t.addEventListener(`error`,function(){return r.loading|=2})}function Rm(e){return`[src="`+nn(e)+`"]`}function zm(e){return`script[async]`+e}function Bm(e,t,n){if(t.count++,t.instance===null)switch(t.type){case`style`:var r=e.querySelector(`style[data-href~="`+nn(n.href)+`"]`);if(r)return t.instance=r,It(r),r;var a=w({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return r=(e.ownerDocument||e).createElement(`style`),It(r),np(r,`style`,a),Vm(r,n.precedence,e),t.instance=r;case`stylesheet`:a=Pm(n.href);var o=e.querySelector(Fm(a));if(o)return t.state.loading|=4,t.instance=o,It(o),o;r=Im(n),(a=vm.get(a))&&Hm(r,a),o=(e.ownerDocument||e).createElement(`link`),It(o);var s=o;return s._p=new Promise(function(e,t){s.onload=e,s.onerror=t}),np(o,`link`,r),t.state.loading|=4,Vm(o,n.precedence,e),t.instance=o;case`script`:return o=Rm(n.src),(a=e.querySelector(zm(o)))?(t.instance=a,It(a),a):(r=n,(a=vm.get(o))&&(r=w({},n),Um(r,a)),e=e.ownerDocument||e,a=e.createElement(`script`),It(a),np(a,`link`,r),e.head.appendChild(a),t.instance=a);case`void`:return null;default:throw Error(i(443,t.type))}else t.type===`stylesheet`&&!(t.state.loading&4)&&(r=t.instance,t.state.loading|=4,Vm(r,n.precedence,e));return t.instance}function Vm(e,t,n){for(var r=n.querySelectorAll(`link[rel="stylesheet"][data-precedence],style[data-precedence]`),i=r.length?r[r.length-1]:null,a=i,o=0;o<r.length;o++){var s=r[o];if(s.dataset.precedence===t)a=s;else if(a!==i)break}a?a.parentNode.insertBefore(e,a.nextSibling):(t=n.nodeType===9?n.head:n,t.insertBefore(e,t.firstChild))}function Hm(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.title??=t.title}function Um(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.integrity??=t.integrity}var Wm=null;function Gm(e,t,n){if(Wm===null){var r=new Map,i=Wm=new Map;i.set(n,r)}else i=Wm,r=i.get(n),r||(r=new Map,i.set(n,r));if(r.has(e))return r;for(r.set(e,null),n=n.getElementsByTagName(e),i=0;i<n.length;i++){var a=n[i];if(!(a[kt]||a[St]||e===`link`&&a.getAttribute(`rel`)===`stylesheet`)&&a.namespaceURI!==`http://www.w3.org/2000/svg`){var o=a.getAttribute(t)||``;o=e+o;var s=r.get(o);s?s.push(a):r.set(o,[a])}}return r}function Km(e,t,n){e=e.ownerDocument||e,e.head.insertBefore(n,t===`title`?e.querySelector(`head > title`):null)}function qm(e,t,n){if(n===1||t.itemProp!=null)return!1;switch(e){case`meta`:case`title`:return!0;case`style`:if(typeof t.precedence!=`string`||typeof t.href!=`string`||t.href===``)break;return!0;case`link`:if(typeof t.rel!=`string`||typeof t.href!=`string`||t.href===``||t.onLoad||t.onError)break;switch(t.rel){case`stylesheet`:return e=t.disabled,typeof t.precedence==`string`&&e==null;default:return!0}case`script`:if(t.async&&typeof t.async!=`function`&&typeof t.async!=`symbol`&&!t.onLoad&&!t.onError&&t.src&&typeof t.src==`string`)return!0}return!1}function Jm(e,t){return e===`img`&&t.src!=null&&t.src!==``&&t.onLoad==null&&t.loading!==`lazy`}function Ym(e){return!(e.type===`stylesheet`&&!(e.state.loading&3))}function Xm(e){return(e.width||100)*(e.height||100)*(typeof devicePixelRatio==`number`?devicePixelRatio:1)*.25}function Zm(e,t){typeof t.decode==`function`&&(e.imgCount++,t.complete||(e.imgBytes+=Xm(t),e.suspenseyImages.push(t)),e=rh.bind(e),t.decode().then(e,e))}function Qm(e,t,n,r){if(n.type===`stylesheet`&&(typeof r.media!=`string`||!1!==matchMedia(r.media).matches)&&!(n.state.loading&4)){if(n.instance===null){var i=Pm(r.href),a=t.querySelector(Fm(i));if(a){t=a._p,typeof t==`object`&&t&&typeof t.then==`function`&&(e.count++,e=nh.bind(e),t.then(e,e)),n.state.loading|=4,n.instance=a,It(a);return}a=t.ownerDocument||t,r=Im(r),(i=vm.get(i))&&Hm(r,i),a=a.createElement(`link`),It(a);var o=a;o._p=new Promise(function(e,t){o.onload=e,o.onerror=t}),np(a,`link`,r),n.instance=a}e.stylesheets===null&&(e.stylesheets=new Map),e.stylesheets.set(n,t),(t=n.state.preload)&&!(n.state.loading&3)&&(e.count++,n=nh.bind(e),t.addEventListener(`load`,n),t.addEventListener(`error`,n))}}var $m=0;function eh(e,t){return e.stylesheets&&e.count===0&&ah(e,e.stylesheets),0<e.count||0<e.imgCount?function(n){var r=setTimeout(function(){if(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}},6e4+t);0<e.imgBytes&&$m===0&&($m=62500*op());var i=setTimeout(function(){if(e.waitingForImages=!1,e.count===0&&(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend)){var t=e.unsuspend;e.unsuspend=null,t()}},(e.imgBytes>$m?50:800)+t);return e.unsuspend=n,function(){e.unsuspend=null,clearTimeout(r),clearTimeout(i)}}:null}function th(e){if(e.count===0&&(e.imgCount===0||!e.waitingForImages)){if(e.stylesheets)ah(e,e.stylesheets);else if(e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}}}function nh(){this.count--,th(this)}function rh(){this.imgCount--,th(this)}var ih=null;function ah(e,t){e.stylesheets=null,e.unsuspend!==null&&(e.count++,ih=new Map,t.forEach(oh,e),ih=null,nh.call(e))}function oh(e,t){if(!(t.state.loading&4)){var n=ih.get(e);if(n)var r=n.get(null);else{n=new Map,ih.set(e,n);for(var i=e.querySelectorAll(`link[data-precedence],style[data-precedence]`),a=0;a<i.length;a++){var o=i[a];(o.nodeName===`LINK`||o.getAttribute(`media`)!==`not all`)&&(n.set(o.dataset.precedence,o),r=o)}r&&n.set(null,r)}i=t.instance,o=i.getAttribute(`data-precedence`),a=n.get(o)||r,a===r&&n.set(null,i),n.set(o,i),this.count++,r=nh.bind(this),i.addEventListener(`load`,r),i.addEventListener(`error`,r),a?a.parentNode.insertBefore(i,a.nextSibling):(e=e.nodeType===9?e.head:e,e.insertBefore(i,e.firstChild)),t.state.loading|=4}}var sh={$$typeof:O,Provider:null,Consumer:null,_currentValue:be,_currentValue2:be,_threadCount:0};function ch(e,t,n,r,i,a,o,s,c){this.tag=1,this.containerInfo=e,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=dt(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=dt(0),this.hiddenUpdates=dt(null),this.identifierPrefix=r,this.onUncaughtError=i,this.onCaughtError=a,this.onRecoverableError=o,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=c,this.transitionTypes=null,this.incompleteTransitions=new Map}function lh(e,t,n,r,i,a,o,s,c,l,u,d){return e=new ch(e,t,n,o,c,l,u,d,s),t=1,!0===a&&(t|=24),a=Mi(3,null,null,t),e.current=a,a.stateNode=e,t=ja(),t.refCount++,e.pooledCache=t,t.refCount++,a.memoizedState={element:r,isDehydrated:n,cache:t},mo(a),e}function uh(e){return e?(e=Ai,e):Ai}function dh(e,t,n,r,i,a){i=uh(i),r.context===null?r.context=i:r.pendingContext=i,r=go(t),r.payload={element:n},a=a===void 0?null:a,a!==null&&(r.callback=a),n=_o(e,r,t),n!==null&&(Pd(n,e,t),vo(n,e,t))}function fh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ph(e,t){fh(e,t),(e=e.alternate)&&fh(e,t)}function mh(e){if(e.tag===13||e.tag===31){var t=Di(e,67108864);t!==null&&Pd(t,e,67108864),ph(e,67108864)}}function hh(e){if(e.tag===13||e.tag===31){var t=jd();t=_t(t);var n=Di(e,t);n!==null&&Pd(n,e,t),ph(e,t)}}var gh=!0;function _h(e,t,n,r){var i=j.T;j.T=null;var a=M.p;try{M.p=2,yh(e,t,n,r)}finally{M.p=a,j.T=i}}function vh(e,t,n,r){var i=j.T;j.T=null;var a=M.p;try{M.p=8,yh(e,t,n,r)}finally{M.p=a,j.T=i}}function yh(e,t,n,r){if(gh){var i=bh(r);if(i===null)Kf(e,t,r,xh,n),Mh(e,r);else if(Ph(i,e,t,n,r))r.stopPropagation();else if(Mh(e,r),t&4&&-1<jh.indexOf(e)){for(;i!==null;){var a=Nt(i);if(a!==null)switch(a.tag){case 3:if(a=a.stateNode,a.current.memoizedState.isDehydrated){var o=at(a.pendingLanes);if(o!==0){var s=a;for(s.pendingLanes|=2,s.entangledLanes|=2;o;){var c=1<<31-Qe(o);s.entanglements[1]|=c,o&=~c}Ef(a),!(K&6)&&(gd=Ve()+500,Df(0,!1))}}break;case 31:case 13:s=Di(a,2),s!==null&&Pd(s,a,2),zd(),ph(a,2)}if(a=bh(r),a===null&&Kf(e,t,r,xh,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Kf(e,t,r,null,n)}}function bh(e){return e=bn(e),Sh(e)}var xh=null;function Sh(e){if(xh=null,e=Mt(e),e!==null){var t=o(e);if(t===null)e=null;else{var n=t.tag;if(n===13){if(e=s(t),e!==null)return e;e=null}else if(n===31){if(e=c(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null)}}return xh=e,null}function Ch(e){switch(e){case`beforetoggle`:case`cancel`:case`click`:case`close`:case`contextmenu`:case`copy`:case`cut`:case`auxclick`:case`dblclick`:case`dragend`:case`dragstart`:case`drop`:case`focusin`:case`focusout`:case`input`:case`invalid`:case`keydown`:case`keypress`:case`keyup`:case`mousedown`:case`mouseup`:case`paste`:case`pause`:case`play`:case`pointercancel`:case`pointerdown`:case`pointerup`:case`ratechange`:case`reset`:case`seeked`:case`submit`:case`toggle`:case`touchcancel`:case`touchend`:case`touchstart`:case`volumechange`:case`change`:case`selectionchange`:case`textInput`:case`compositionstart`:case`compositionend`:case`compositionupdate`:case`beforeblur`:case`afterblur`:case`beforeinput`:case`blur`:case`fullscreenchange`:case`fullscreenerror`:case`focus`:case`hashchange`:case`popstate`:case`select`:case`selectstart`:return 2;case`drag`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`mousemove`:case`mouseout`:case`mouseover`:case`pointermove`:case`pointerout`:case`pointerover`:case`resize`:case`scroll`:case`touchmove`:case`wheel`:case`mouseenter`:case`mouseleave`:case`pointerenter`:case`pointerleave`:return 8;case`message`:switch(He()){case Ue:return 2;case We:return 8;case Ge:case Ke:return 32;case R:return 268435456;default:return 32}default:return 32}}var wh=!1,Th=null,Eh=null,Dh=null,Oh=new Map,kh=new Map,Ah=[],jh=`mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset`.split(` `);function Mh(e,t){switch(e){case`focusin`:case`focusout`:Th=null;break;case`dragenter`:case`dragleave`:Eh=null;break;case`mouseover`:case`mouseout`:Dh=null;break;case`pointerover`:case`pointerout`:Oh.delete(t.pointerId);break;case`gotpointercapture`:case`lostpointercapture`:kh.delete(t.pointerId)}}function Nh(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Nt(t),t!==null&&mh(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Ph(e,t,n,r,i){switch(t){case`focusin`:return Th=Nh(Th,e,t,n,r,i),!0;case`dragenter`:return Eh=Nh(Eh,e,t,n,r,i),!0;case`mouseover`:return Dh=Nh(Dh,e,t,n,r,i),!0;case`pointerover`:var a=i.pointerId;return Oh.set(a,Nh(Oh.get(a)||null,e,t,n,r,i)),!0;case`gotpointercapture`:return a=i.pointerId,kh.set(a,Nh(kh.get(a)||null,e,t,n,r,i)),!0}return!1}function Fh(e){var t=Mt(e.target);if(t!==null){var n=o(t);if(n!==null){if(t=n.tag,t===13){if(t=s(n),t!==null){e.blockedOn=t,bt(e.priority,function(){hh(n)});return}}else if(t===31){if(t=c(n),t!==null){e.blockedOn=t,bt(e.priority,function(){hh(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ih(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=bh(e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);yn=r,n.target.dispatchEvent(r),yn=null}else return t=Nt(n),t!==null&&mh(t),e.blockedOn=n,!1;t.shift()}return!0}function Lh(e,t,n){Ih(e)&&n.delete(t)}function Rh(){wh=!1,Th!==null&&Ih(Th)&&(Th=null),Eh!==null&&Ih(Eh)&&(Eh=null),Dh!==null&&Ih(Dh)&&(Dh=null),Oh.forEach(Lh),kh.forEach(Lh)}function zh(e,n){e.blockedOn===n&&(e.blockedOn=null,wh||(wh=!0,t.unstable_scheduleCallback(t.unstable_NormalPriority,Rh)))}var Bh=null;function Vh(e){Bh!==e&&(Bh=e,t.unstable_scheduleCallback(t.unstable_NormalPriority,function(){Bh===e&&(Bh=null);for(var t=0;t<e.length;t+=3){var n=e[t],r=e[t+1],i=e[t+2];if(typeof r!=`function`){if(Sh(r||n)===null)continue;break}var a=Nt(n);a!==null&&(e.splice(t,3),t-=3,ec(a,{pending:!0,data:i,method:n.method,action:r},r,i))}}))}function Hh(e){function t(t){return zh(t,e)}Th!==null&&zh(Th,e),Eh!==null&&zh(Eh,e),Dh!==null&&zh(Dh,e),Oh.forEach(t),kh.forEach(t);for(var n=0;n<Ah.length;n++){var r=Ah[n];r.blockedOn===e&&(r.blockedOn=null)}for(;0<Ah.length&&(n=Ah[0],n.blockedOn===null);)Fh(n),n.blockedOn===null&&Ah.shift();if(n=(e.ownerDocument||e).$$reactFormReplay,n!=null)for(r=0;r<n.length;r+=3){var i=n[r],a=n[r+1],o=i[Ct]||null;if(typeof a==`function`)o||Vh(n);else if(o){var s=null;if(a&&a.hasAttribute(`formAction`)){if(i=a,o=a[Ct]||null)s=o.formAction;else if(Sh(i)!==null)continue}else s=o.action;typeof s==`function`?n[r+1]=s:(n.splice(r,3),r-=3),Vh(n)}}}function Uh(){function e(e){e.canIntercept&&e.info===`react-transition`&&e.intercept({handler:function(){return new Promise(function(e){return i=e})},focusReset:`manual`,scroll:`manual`})}function t(){i!==null&&(i(),i=null),r||setTimeout(n,20)}function n(){if(!r&&!navigation.transition){var e=navigation.currentEntry;e&&e.url!=null&&navigation.navigate(e.url,{state:e.getState(),info:`react-transition`,history:`replace`})}}if(typeof navigation==`object`){var r=!1,i=null;return navigation.addEventListener(`navigate`,e),navigation.addEventListener(`navigatesuccess`,t),navigation.addEventListener(`navigateerror`,t),setTimeout(n,100),function(){r=!0,navigation.removeEventListener(`navigate`,e),navigation.removeEventListener(`navigatesuccess`,t),navigation.removeEventListener(`navigateerror`,t),i!==null&&(i(),i=null)}}}function Wh(e){this._internalRoot=e}Gh.prototype.render=Wh.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(i(409));var n=t.current;dh(n,jd(),e,t,null,null)},Gh.prototype.unmount=Wh.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;dh(e.current,2,null,e,null,null),zd(),t[wt]=null}};function Gh(e){this._internalRoot=e}Gh.prototype.unstable_scheduleHydration=function(e){if(e){var t=yt();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Ah.length&&t!==0&&t<Ah[n].priority;n++);Ah.splice(n,0,e),n===0&&Fh(e)}};var Kh=n.version;if(Kh!==`19.3.0`)throw Error(i(527,Kh,`19.3.0`));M.findDOMNode=function(e){var t=e._reactInternals;if(t===void 0)throw typeof e.render==`function`?Error(i(188)):(e=Object.keys(e).join(`,`),Error(i(268,e)));return e=d(t),e=e===null?null:p(e),e=e===null?null:e.stateNode,e};var qh={bundleType:0,version:`19.3.0`,rendererPackageName:`react-dom`,currentDispatcherRef:j,reconcilerVersion:`19.3.0`};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`){var Jh=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Jh.isDisabled&&Jh.supportsFiber)try{Ye=Jh.inject(qh),Xe=Jh}catch{}}e.createRoot=function(e,t){if(!a(e))throw Error(i(299));var n=!1,r=``,o=Cc,s=wc,c=Tc;return t!=null&&(!0===t.unstable_strictMode&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onUncaughtError!==void 0&&(o=t.onUncaughtError),t.onCaughtError!==void 0&&(s=t.onCaughtError),t.onRecoverableError!==void 0&&(c=t.onRecoverableError)),t=lh(e,1,!1,null,null,n,r,null,o,s,c,Uh),e[wt]=t.current,Wf(e),new Wh(t)}})),g=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=h()})),_=c(u(),1),v=g(),y=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.fragment`);function r(e,n,r){var i=null;if(r!==void 0&&(i=``+r),n.key!==void 0&&(i=``+n.key),`key`in n)for(var a in r={},n)a!==`key`&&(r[a]=n[a]);else r=n;return n=r.ref,{$$typeof:t,type:e,key:i,ref:n===void 0?null:n,props:r}}e.Fragment=n,e.jsx=r,e.jsxs=r})),b=o(((e,t)=>{t.exports=y()})),x=b(),ee={models:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h4A1.5 1.5 0 0 1 12 6.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 10.5zm8 0A1.5 1.5 0 0 1 14.5 5h3A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-3a1.5 1.5 0 0 1-1.5-1.5zm-8 8A1.5 1.5 0 0 1 6.5 13h4a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 17.5z`}),chat:(0,x.jsx)(`path`,{d:`M5.5 7.5A2.5 2.5 0 0 1 8 5h8a2.5 2.5 0 0 1 2.5 2.5v5A2.5 2.5 0 0 1 16 15h-3.7l-3.6 3.2a.7.7 0 0 1-1.2-.52V15A2.5 2.5 0 0 1 5.5 12.5z`}),activity:(0,x.jsx)(`path`,{d:`M4.5 13h3l1.8-5.5 3.2 9 2.1-6.5h4.9`}),settings:(0,x.jsx)(`path`,{d:`M12 8.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6Zm0-3.2v2m0 10v2m7-7h-2M7 12H5m11.95-4.95-1.42 1.42M8.47 15.53l-1.42 1.42m9.9 0-1.42-1.42M8.47 8.47 7.05 7.05`}),gallery:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h11A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 17.5zm3 2h8m-8 3h8m-8 3h5`}),command:(0,x.jsx)(`path`,{d:`M8 8.5A2.5 2.5 0 1 1 5.5 6 2.5 2.5 0 0 1 8 8.5Zm0 7A2.5 2.5 0 1 1 5.5 13 2.5 2.5 0 0 1 8 15.5Zm8-7A2.5 2.5 0 1 1 13.5 6 2.5 2.5 0 0 1 16 8.5Zm0 7A2.5 2.5 0 1 1 13.5 13 2.5 2.5 0 0 1 16 15.5Z`}),help:(0,x.jsx)(`path`,{d:`M9.5 9a2.5 2.5 0 1 1 4.4 1.62c-.68.72-1.9 1.18-1.9 2.38v.25m0 3.25h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z`}),menu:(0,x.jsx)(`path`,{d:`M5 7h14M5 12h14M5 17h14`}),close:(0,x.jsx)(`path`,{d:`m7 7 10 10M17 7 7 17`}),search:(0,x.jsx)(`path`,{d:`m16.5 16.5 3 3M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14Z`}),warning:(0,x.jsx)(`path`,{d:`M12 4.8 21 19H3zm0 5.2v4m0 2.5h.01`}),key:(0,x.jsx)(`path`,{d:`M14.5 9.5A4.5 4.5 0 1 0 11 13.9l2.1 2.1H16v2h2v2h2v-2.9l-4.6-4.6a4.5 4.5 0 0 0-.9-3Z`}),schema:(0,x.jsx)(`path`,{d:`M6 5h7l5 5v9H6zm7 0v5h5M9 13h6M9 16h6M9 10h2`})};function te(e){return(0,x.jsx)(`svg`,{className:e.className??`ds-icon`,"aria-hidden":`true`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`1.8`,strokeLinecap:`round`,strokeLinejoin:`round`,children:ee[e.name]})}var ne=(0,_.forwardRef)(function({children:e,onClick:t,onDoubleClick:n,onMouseDown:r,onMouseEnter:i,onKeyDown:a,type:o=`button`,disabled:s=!1,variant:c=`secondary`,size:l=`medium`,shape:u=`default`,fullWidth:d=!1,icon:f,iconPosition:p=`left`,iconOnly:m=!1,inline:h=!1,loading:g=!1,active:v=!1,ariaLabel:y,title:b,role:ee,id:te,tabIndex:ne,className:S=``,style:C,"aria-pressed":w,"aria-checked":re,"aria-selected":T,"aria-expanded":ie,"aria-haspopup":E,"aria-controls":ae,"aria-describedby":oe,"aria-busy":D,"data-testid":O},se){let ce=(0,_.useCallback)(e=>{!s&&!g&&t&&t(e)},[s,g,t]),le=[`button`,`button--${c}`,`button--${l}`,u!=="default"&&`button--${u}`,d&&`button--full-width`,m&&`button--icon-only`,h&&`button--inline`,g&&`button--loading`,v&&`button--active`,S].filter(Boolean).join(` `),ue=f??(m?e:void 0),de=!m&&e,fe=ue&&!g;return(0,x.jsxs)(`button`,{ref:se,type:o,onClick:ce,onDoubleClick:n,onMouseDown:r,onMouseEnter:i,onKeyDown:a,disabled:s||g,className:le,"aria-label":y??(m&&typeof e==`string`?e:void 0),title:b,role:ee,id:te,tabIndex:ne,style:C,"aria-pressed":w,"aria-checked":re,"aria-selected":T,"aria-expanded":ie,"aria-haspopup":E,"aria-controls":ae,"aria-describedby":oe,"aria-busy":D,"data-testid":O,children:[g&&(0,x.jsx)(`span`,{className:`button__loading-spinner`,"aria-hidden":`true`}),fe&&p===`left`&&(0,x.jsx)(`span`,{className:`button__icon`,"aria-hidden":`true`,children:ue}),de&&(h?e:(0,x.jsx)(`span`,{className:`button__text`,children:e})),fe&&p===`right`&&(0,x.jsx)(`span`,{className:`button__icon`,"aria-hidden":`true`,children:ue})]})});function S({children:e,variant:t=`default`,size:n=`small`,className:r=``}){let i=[`badge`,`badge--${t}`,`badge--${n}`,r].filter(Boolean).join(` `);return(0,x.jsx)(`span`,{className:i,children:e})}function C(e){switch(e){case`running`:return`success`;case`preparing`:return`info`;case`idle`:return`default`;case`busy`:return`primary`;case`stopping`:return`warning`;case`terminated`:return`default`;case`error`:return`danger`}}function w(e){return e===`preparing`||e===`stopping`||e===`busy`}function re({state:e,label:t,size:n=`small`,pulse:r,className:i=``}){let a=r??w(e),o=[`status-tag`,`status-tag--${e}`,i].filter(Boolean).join(` `);return(0,x.jsxs)(S,{variant:C(e),size:n,className:o,children:[a&&(0,x.jsx)(`span`,{className:`status-tag__indicator`,"aria-hidden":`true`,"data-testid":`status-tag-indicator`}),(0,x.jsx)(`span`,{className:`status-tag__label`,children:t})]})}var T=(0,_.memo)(re);function ie({value:e,variant:t=`primary`,size:n=`md`,showLabel:r=!1,label:i,className:a=``,animated:o=!0,ariaLabel:s}){let c=e===null,l=e===null?0:Math.max(0,Math.min(100,e)),u=[`progress-bar`,`progress-bar--${t}`,`progress-bar--${n}`,c?`progress-bar--indeterminate`:``,o?`progress-bar--animated`:``,a].filter(Boolean).join(` `),d=i??(r&&!c?`${l.toFixed(0)}%`:null);return(0,x.jsxs)(`div`,{className:`progress-bar-wrapper`,children:[(0,x.jsx)(`div`,{className:u,role:`progressbar`,"aria-valuenow":c?void 0:l,"aria-valuemin":0,"aria-valuemax":100,"aria-label":s,"aria-busy":c,children:(0,x.jsx)(`div`,{className:`progress-bar__fill`,style:c?void 0:{width:`${String(l)}%`}})}),d&&(0,x.jsx)(`span`,{className:`progress-bar__label`,children:d})]})}function E({illustration:e,title:t,description:n,primaryAction:r,secondaryAction:i,children:a,className:o=``,showIllustration:s=!0}){let c=r?.onClick,l=i?.onClick,u=(0,_.useCallback)(()=>{c?.()},[c]),d=(0,_.useCallback)(()=>{l?.()},[l]),f=(0,_.useMemo)(()=>[`empty-state`,o].filter(Boolean).join(` `),[o]);return(0,x.jsxs)(`div`,{className:f,role:`status`,"aria-live":`polite`,children:[s&&e&&(0,x.jsx)(`div`,{className:`empty-state__illustration`,children:e}),(0,x.jsxs)(`div`,{className:`empty-state__content`,children:[(0,x.jsx)(`h3`,{className:`empty-state__title`,children:t}),(0,x.jsx)(`p`,{className:`empty-state__description`,children:n}),a]}),(r||i)&&(0,x.jsxs)(`div`,{className:`empty-state__actions`,children:[r&&(0,x.jsx)(ne,{variant:`primary`,onClick:u,className:`empty-state__action-btn empty-state__action-btn--primary`,children:r.label}),i&&(0,x.jsx)(x.Fragment,{children:i.href?(0,x.jsx)(`a`,{href:i.href,className:`empty-state__action-link`,onClick:d,children:i.label}):(0,x.jsx)(ne,{variant:`ghost`,onClick:d,className:`empty-state__action-btn empty-state__action-btn--secondary`,children:i.label})})]})]})}var ae=(0,_.memo)(E);function oe({tabs:e,groups:t=[],defaultTab:n,activeTab:r,onTabChange:i,className:a=``,panelClassName:o=``,ariaLabel:s,showOverflowControls:c=!0,showGroupLabels:l=!0,overflowMode:u,variant:d=`underlined`,fillContainer:f=!1,renderPanel:p=!0,moreTabsLabel:m=`More tabs`,selectTabLabel:h=`Select tab`,scrollLeftLabel:g=`Scroll left`,scrollRightLabel:v=`Scroll right`}){let y=u??(t.length>0?`menu`:`dropdown`),[b,ee]=(0,_.useState)(n||e[0]?.id||``),[te,ne]=(0,_.useState)(!1),[S,C]=(0,_.useState)(!1),[w,re]=(0,_.useState)(!1),[T,ie]=(0,_.useState)(!1),[E,ae]=(0,_.useState)(!1),[oe,D]=(0,_.useState)(-1),O=(0,_.useRef)(null),se=(0,_.useRef)(new Map),ce=(0,_.useRef)(null),le=(0,_.useRef)(null),ue=(0,_.useRef)(null),de=(0,_.useRef)([]),fe=(0,_.useRef)(null),pe=(0,_.useRef)(0),k=(0,_.useRef)(0),A=r??b,me=(0,_.useMemo)(()=>t.length>0?t.map(t=>({group:t,tabs:e.filter(e=>e.groupId===t.id)})):[{group:null,tabs:e}],[t,e]),he=(0,_.useMemo)(()=>me.flatMap(({tabs:e})=>e),[me]),ge=(0,_.useCallback)(e=>{r||ee(e),i?.(e),ae(!1)},[r,i]),_e=(0,_.useCallback)((t,n)=>{let r;switch(t.key){case`ArrowRight`:t.preventDefault(),r=n===e.length-1?0:n+1;break;case`ArrowLeft`:t.preventDefault(),r=n===0?e.length-1:n-1;break;case`Home`:t.preventDefault(),r=0;break;case`End`:t.preventDefault(),r=e.length-1;break;default:return}let i=e[r];i&&(ge(i.id),se.current.get(i.id)?.focus())},[e,ge]),ve=(0,_.useCallback)((e,t)=>{t?se.current.set(e,t):se.current.delete(e)},[]),ye=(0,_.useCallback)(()=>{let e=O.current;if(!e||!c){ne(!1),C(!1);return}let{scrollLeft:t,scrollWidth:n,clientWidth:r}=e;ne(t>0),C(t+r<n-1)},[c]),j=(0,_.useCallback)(e=>{let t=O.current;if(!t)return;let n=t.clientWidth*.75;t.scrollBy({left:e===`left`?-n:n,behavior:`smooth`}),fe.current!==null&&clearTimeout(fe.current),fe.current=setTimeout(ye,300)},[ye]),M=(0,_.useCallback)(()=>{if(!O.current||w)return;let e=se.current.get(A);if(!e)return;let t=O.current,n=e.getBoundingClientRect(),r=t.getBoundingClientRect();n.left<r.left?t.scrollLeft-=r.left-n.left+8:n.right>r.right&&(t.scrollLeft+=n.right-r.right+8)},[A,w]),be=(0,_.useCallback)(()=>{typeof window>`u`||y!==`dropdown`||re(window.innerWidth<480)},[y]),xe=(0,_.useCallback)(()=>{ie(e=>!e)},[]),N=(0,_.useCallback)(e=>{e.key===`Escape`&&T&&(e.preventDefault(),ie(!1))},[T]),P=(0,_.useCallback)(e=>{e.touches[0]&&(pe.current=e.touches[0].clientX)},[]),F=(0,_.useCallback)(e=>{e.touches[0]&&(k.current=e.touches[0].clientX)},[]),I=(0,_.useCallback)(()=>{if(!pe.current||!k.current)return;let t=pe.current-k.current;if(Math.abs(t)>50){let n=e.findIndex(e=>e.id===A),r=e[n+1],i=e[n-1];t>0&&r?ge(r.id):t<0&&i&&ge(i.id)}pe.current=0,k.current=0},[A,e,ge]),L=(0,_.useCallback)(e=>{let t=he.length;switch(e.key){case`Escape`:e.preventDefault(),ae(!1),D(-1),ue.current?.focus();break;case`ArrowDown`:e.preventDefault(),D(e=>{let n=e>=t-1?0:e+1;return de.current[n]?.focus(),n});break;case`ArrowUp`:e.preventDefault(),D(e=>{let n=e<=0?t-1:e-1;return de.current[n]?.focus(),n});break;case`Home`:e.preventDefault(),D(0),de.current[0]?.focus();break;case`End`:e.preventDefault(),D(t-1),de.current[t-1]?.focus()}},[he.length]),Se=(0,_.useCallback)(e=>{e.key===`Escape`&&E&&(e.preventDefault(),ae(!1),D(-1))},[E]);(0,_.useEffect)(()=>{if(!T)return;let e=e=>{ce.current&&!ce.current.contains(e.target)&&ie(!1)};return document.addEventListener(`mousedown`,e),()=>{document.removeEventListener(`mousedown`,e)}},[T]),(0,_.useEffect)(()=>{if(!E)return;let e=e=>{le.current&&!le.current.contains(e.target)&&(ae(!1),D(-1))},t=e=>{e.key===`Escape`&&(ae(!1),D(-1),ue.current?.focus())};return document.addEventListener(`mousedown`,e),document.addEventListener(`keydown`,t),()=>{document.removeEventListener(`mousedown`,e),document.removeEventListener(`keydown`,t)}},[E]),(0,_.useEffect)(()=>{if(E&&de.current.length>0){let e=he.findIndex(e=>e.id===A),t=e>=0?e:0;D(t),requestAnimationFrame(()=>{de.current[t]?.focus()})}},[E,he,A]),(0,_.useEffect)(()=>{if(!e.find(e=>e.id===A)){let t=e[0];t&&ge(t.id)}},[e,A,ge]),(0,_.useEffect)(()=>{be(),ye();let e=()=>{be(),ye()};return window.addEventListener(`resize`,e),()=>{window.removeEventListener(`resize`,e)}},[be,ye]),(0,_.useEffect)(()=>{M()},[A,M]),(0,_.useEffect)(()=>{ye()},[e,ye]),(0,_.useEffect)(()=>()=>{fe.current!==null&&clearTimeout(fe.current)},[]);let Ce=e.find(e=>e.id===A)?.content,we=e.find(e=>e.id===A)?.label,Te=t.length>0;de.current=[];let Ee=(e,t)=>{let n=e.id===A;return(0,x.jsxs)(`button`,{ref:t=>{ve(e.id,t)},role:`tab`,"aria-selected":n,"aria-controls":p?`tabpanel-${e.id}`:void 0,id:`tab-${e.id}`,tabIndex:n?0:-1,className:`tabs__tab${n?` tabs__tab--active`:``}`,onClick:()=>{ge(e.id)},onKeyDown:e=>{_e(e,t),(e.key===`ArrowLeft`||e.key===`ArrowRight`||e.key===`Home`||e.key===`End`)&&e.stopPropagation()},children:[e.labelPrefix,(0,x.jsx)(`span`,{className:`tabs__tab-label`,children:e.label}),e.labelExtra]},e.id)},De=()=>Te?me.map(({group:t,tabs:n},r)=>(0,x.jsxs)(`div`,{className:`tabs__group`,children:[t&&(0,x.jsxs)(x.Fragment,{children:[r>0&&(0,x.jsx)(`div`,{className:`tabs__separator`,role:`separator`,"aria-hidden":`true`}),l&&(0,x.jsx)(`span`,{className:`tabs__group-label`,children:t.label})]}),(0,x.jsx)(`div`,{className:`tabs__group-tabs`,children:n.map(t=>{let n=e.findIndex(e=>e.id===t.id);return Ee(t,n)})})]},t?.id||`default`)):e.map((e,t)=>Ee(e,t)),Oe=()=>y!==`menu`||!c?null:(0,x.jsxs)(`div`,{className:`tabs__overflow`,ref:le,children:[(0,x.jsx)(`button`,{ref:ue,type:`button`,className:`tabs__overflow-btn`,onClick:()=>{ae(!E)},onKeyDown:Se,"aria-label":m,"aria-expanded":E,"aria-haspopup":`menu`,children:(0,x.jsxs)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`,children:[(0,x.jsx)(`circle`,{cx:`12`,cy:`12`,r:`1`}),(0,x.jsx)(`circle`,{cx:`12`,cy:`5`,r:`1`}),(0,x.jsx)(`circle`,{cx:`12`,cy:`19`,r:`1`})]})}),E&&(0,x.jsx)(`div`,{className:`tabs__overflow-menu`,role:`menu`,onKeyDown:L,children:me.map(({group:e,tabs:t})=>(0,x.jsxs)(`div`,{className:`tabs__overflow-group`,children:[e&&l&&(0,x.jsx)(`div`,{className:`tabs__overflow-group-label`,children:e.label}),t.map(e=>{let t=e.id===A,n=he.findIndex(t=>t.id===e.id);return(0,x.jsxs)(`button`,{ref:e=>{e&&(de.current[n]=e)},role:`menuitem`,tabIndex:oe===n?0:-1,className:`tabs__overflow-item ${t?`tabs__overflow-item--active`:``}`,onClick:()=>{ge(e.id)},children:[e.labelPrefix,(0,x.jsx)(`span`,{children:e.label}),e.labelExtra]},e.id)})]},e?.id||`default`))})]}),ke=d===`segmented`||d===`compact`;return y===`dropdown`&&w&&!ke?(0,x.jsxs)(`div`,{className:`tabs tabs--mobile-dropdown ${a}`,children:[(0,x.jsxs)(`div`,{className:`tabs__dropdown-container`,ref:ce,children:[(0,x.jsxs)(`button`,{className:`tabs__dropdown-trigger`,onClick:xe,onKeyDown:N,"aria-expanded":T,"aria-haspopup":`listbox`,"aria-label":s||h,children:[(0,x.jsx)(`span`,{className:`tabs__dropdown-label`,children:we}),(0,x.jsx)(`svg`,{className:`tabs__dropdown-arrow ${T?`tabs__dropdown-arrow--open`:``}`,width:`16`,height:`16`,viewBox:`0 0 16 16`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M4 6L8 10L12 6`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})]}),T&&(0,x.jsx)(`div`,{className:`tabs__dropdown-menu`,role:`listbox`,children:e.map(e=>(0,x.jsx)(`button`,{role:`option`,"aria-selected":e.id===A,className:`tabs__dropdown-item ${e.id===A?`tabs__dropdown-item--active`:``}`,onClick:()=>{ge(e.id),ie(!1)},children:e.label},e.id))})]}),p&&(0,x.jsx)(`div`,{id:`tabpanel-${A}`,role:`tabpanel`,"aria-labelledby":`tab-${A}`,className:`tabs__panel ${o}`.trim(),tabIndex:0,onTouchStart:P,onTouchMove:F,onTouchEnd:I,children:Ce})]}):(0,x.jsxs)(`div`,{className:`tabs ${y===`menu`?`tabs--overflow-menu`:``} ${d===`segmented`?`tabs--variant-segmented`:d===`compact`?`tabs--variant-compact`:``} ${ke&&f?`tabs--fill-container`:``} ${a}`.replace(/\s+/g,` `).trim(),children:[(0,x.jsxs)(`div`,{className:`tabs__container`,children:[c&&te&&(0,x.jsx)(`button`,{className:`tabs__scroll-arrow tabs__scroll-arrow--left`,onClick:()=>{j(`left`)},"aria-label":g,tabIndex:-1,children:(0,x.jsx)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 20 20`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M12 15L7 10L12 5`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})}),(0,x.jsx)(`div`,{ref:O,className:`tabs__list`,role:`tablist`,"aria-label":s,onScroll:ye,onKeyDown:t=>{let n=e.findIndex(e=>e.id===A);n<0||_e(t,n)},onTouchStart:y===`dropdown`?P:void 0,onTouchMove:y===`dropdown`?F:void 0,onTouchEnd:y===`dropdown`?I:void 0,children:De()}),c&&S&&(0,x.jsx)(`button`,{className:`tabs__scroll-arrow tabs__scroll-arrow--right`,onClick:()=>{j(`right`)},"aria-label":v,tabIndex:-1,children:(0,x.jsx)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 20 20`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M8 5L13 10L8 15`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})}),Oe()]}),p&&(0,x.jsx)(`div`,{id:`tabpanel-${A}`,role:`tabpanel`,"aria-labelledby":`tab-${A}`,className:`tabs__panel ${o}`.trim(),tabIndex:0,children:Ce})]})}var D=[`a`,`button`,`input`,`select`,`textarea`,`summary`,`label`,`[contenteditable='true']`,`[role='button']`,`[role='link']`,`[tabindex]:not([tabindex='-1'])`].join(`,`);function O(e,t){if(!(e instanceof Element)||!(t instanceof Element))return!1;let n=e.closest(D);return n!==null&&n!==t&&t.contains(n)}var se={widths:{},visibility:{}};function ce(e,t){let n=e.map((e,t)=>({row:e,i:t}));return n.sort((e,n)=>{let r=t(e.row,n.row);return r===0?e.i-n.i:r}),n.map(({row:e})=>e)}function le(e,t){let{sortComparator:n}=e;if(n)return(e,r)=>n(e,r,t);let r=e.sortValueAccessor;return r?(e,n)=>{let i=r(e),a=r(n);if(i==null&&a==null)return 0;if(i==null)return 1;if(a==null)return-1;let o;return o=typeof i==`number`&&typeof a==`number`?i-a:String(i).localeCompare(String(a)),t===`asc`?o:-o}:()=>0}function ue(e,t,n){if(e.sortable)return e.id===t?n===`asc`?`ascending`:`descending`:`none`}function de({direction:e}){return e===`none`?(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--none`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`,opacity:`0.35`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`,opacity:`0.35`})]})}):e===`asc`?(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--asc`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`,opacity:`0.25`})]})}):(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--desc`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`,opacity:`0.25`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`})]})})}function fe({columns:e,rows:t,getRowKey:n,emptyState:r,loadingState:i,loading:a=!1,columnState:o,onColumnStateChange:s,className:c=``,ariaLabel:l=`Data table`,testId:u,onRowClick:d,isRowClickable:f,rowClassName:p,sortColumnId:m,sortDirection:h,onSortChange:g}){let[v,y]=(0,_.useState)(()=>o??se);(0,_.useEffect)(()=>{o&&y(o)},[o]);let b=m!==void 0||h!==void 0,[ee,te]=(0,_.useState)(void 0),[ne,S]=(0,_.useState)(void 0),C=b?m:ee,w=b?h:ne,re=(0,_.useCallback)(e=>{if(!e.sortable)return;let t,n;C===e.id?w===(e.defaultSortDirection??`asc`)?(t=e.id,n=w===`asc`?`desc`:`asc`):(t=void 0,n=void 0):(t=e.id,n=e.defaultSortDirection??`asc`),b||(te(t),S(n)),g?.(t,n)},[C,w,b,g]),T=(0,_.useMemo)(()=>e.filter(e=>e.alwaysVisible?!0:v.visibility[e.id]!==!1),[e,v.visibility]),ie=(0,_.useCallback)(e=>{let t=v.widths[e.id];return typeof t==`number`&&t>0?t:e.initialWidth},[v.widths]),E=(0,_.useRef)(null),ae=(0,_.useCallback)((e,t,n)=>{if(t.noResize)return;e.preventDefault(),e.stopPropagation(),E.current={columnId:t.id,startX:e.clientX,startWidth:n,minWidth:t.minWidth??80};let r=e=>{let t=E.current;if(!t)return;let n=e.clientX-t.startX,r=Math.max(t.minWidth,t.startWidth+n);y(e=>({...e,widths:{...e.widths,[t.columnId]:r}}))},i=()=>{E.current=null,window.removeEventListener(`mousemove`,r),window.removeEventListener(`mouseup`,i)};window.addEventListener(`mousemove`,r),window.addEventListener(`mouseup`,i)},[]),oe=(0,_.useRef)(v);(0,_.useEffect)(()=>{oe.current!==v&&(oe.current=v,s?.(v))},[v,s]);let D=(0,_.useMemo)(()=>{if(!C||!w)return t;let n=e.find(e=>e.id===C);return n?.sortable?ce(t,le(n,w)):t},[t,e,C,w]),fe=[`data-table`,c].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{className:fe,"data-testid":u,children:(0,x.jsxs)(`table`,{className:`data-table__table`,"aria-label":l,role:`table`,children:[(0,x.jsx)(`thead`,{className:`data-table__head`,children:(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--head`,children:T.map(e=>{let t=ie(e),n=e.sortable&&e.id===C;return(0,x.jsxs)(`th`,{scope:`col`,className:[`data-table__cell`,`data-table__cell--head`,e.sortable&&`data-table__cell--sortable`,n&&`data-table__cell--sort-active`,e.className,e.align&&`data-table__cell--align-${e.align}`].filter(Boolean).join(` `),style:t?{width:`${String(t)}px`}:void 0,"aria-sort":ue(e,C,w),onClick:e.sortable?()=>{re(e)}:void 0,onKeyDown:e.sortable?t=>{(t.key===`Enter`||t.key===` `)&&(t.preventDefault(),re(e))}:void 0,tabIndex:e.sortable?0:void 0,children:[(0,x.jsx)(`span`,{className:`data-table__head-label`,children:e.header}),e.sortable&&(0,x.jsx)(de,{direction:n&&w?w:`none`}),!e.noResize&&(0,x.jsx)(`div`,{role:`separator`,"aria-orientation":`vertical`,className:`data-table__resize-handle`,onMouseDown:n=>{ae(n,e,t??e.minWidth??120)}})]},e.id)})})}),(0,x.jsx)(`tbody`,{className:`data-table__body`,children:a?(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--state`,children:(0,x.jsx)(`td`,{colSpan:T.length||1,className:`data-table__state-cell`,children:i??(0,x.jsx)(`div`,{className:`data-table__loading`})})}):D.length===0?(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--state`,children:(0,x.jsx)(`td`,{colSpan:T.length||1,className:`data-table__state-cell`,children:r})}):D.map((e,t)=>{let r=n(e,t),i=!!(d&&(f?.(e,t)??!0)),a=p?.(e,t);return(0,x.jsx)(`tr`,{className:[`data-table__row`,i&&`data-table__row--clickable`,a].filter(Boolean).join(` `),onClick:i?t=>{O(t.target,t.currentTarget)||d?.(e)}:void 0,onKeyDown:i?t=>{(t.key===`Enter`||t.key===` `)&&(O(t.target,t.currentTarget)||(t.preventDefault(),d?.(e)))}:void 0,tabIndex:i?0:void 0,role:i?`button`:void 0,children:T.map(n=>(0,x.jsx)(`td`,{className:[`data-table__cell`,n.className,n.align&&`data-table__cell--align-${n.align}`].filter(Boolean).join(` `),children:n.render(e,t)},n.id))},r)})})]})})}var pe=(0,_.memo)(fe),k=(0,_.forwardRef)(function({tone:e=`secondary`,busy:t=!1,className:n=``,children:r,"aria-label":i,...a},o){return(0,x.jsx)(ne,{...a,ref:o,inline:!0,variant:e,loading:t,"aria-busy":t||a[`aria-busy`],ariaLabel:i,className:`ds-button ds-button-${e} ${n}`.trim(),children:(0,x.jsx)(`span`,{children:r})})}),A=(0,_.forwardRef)(function({label:e,icon:t,busy:n=!1,className:r=``,...i},a){return(0,x.jsx)(ne,{...i,ref:a,inline:!0,iconOnly:!0,icon:(0,x.jsx)(te,{name:t}),ariaLabel:e,title:i.title??e,variant:`secondary`,loading:n,"aria-busy":n||i[`aria-busy`],className:`ds-icon-button ${r}`.trim()})}),me={unloaded:`terminated`,loading:`preparing`,ready:`running`,draining:`stopping`,unloading:`stopping`,failed:`error`};function he(e){return(0,x.jsx)(`span`,{className:`ds-status-wrap`,"data-state":e.state,"aria-label":e.label,children:(0,x.jsx)(T,{state:me[e.state],label:e.children,pulse:!1,className:`ds-status`})})}function ge(e){let t=e.value===void 0||!Number.isFinite(e.value)?null:Math.max(0,Math.min(100,e.value));return(0,x.jsxs)(`div`,{className:`ds-progress`,children:[(0,x.jsx)(ie,{value:t,ariaLabel:e.label,animated:!1}),e.detail?(0,x.jsx)(`small`,{children:e.detail}):null]})}function _e(e){return(0,x.jsx)(`section`,{className:`ds-empty`,"data-testid":e.testId,ref:e=>{let t=e?.querySelector(`.empty-state__title`);t?.setAttribute(`role`,`heading`),t?.setAttribute(`aria-level`,`2`)},children:(0,x.jsx)(ae,{title:e.title,description:e.body,illustration:(0,x.jsx)(te,{name:`models`}),children:e.action})})}function ve(e){let t=(0,_.useId)();if(e.tabs.length===0)return(0,x.jsx)(`section`,{className:`ds-tabs`});let n=e.tabs.map(e=>({id:`${t}-${e.id}`,label:e.label,content:e.panel})),r=e.tabs.find(t=>t.id===e.active)??e.tabs[0];return(0,x.jsx)(`section`,{onKeyDownCapture:e=>{e.nativeEvent.isComposing&&e.stopPropagation()},children:(0,x.jsx)(oe,{className:`ds-tabs`,tabs:n,activeTab:`${t}-${r.id}`,onTabChange:n=>{let r=e.tabs.find(e=>`${t}-${e.id}`===n);r&&e.onChange(r.id)},ariaLabel:e.label??`Sections`,variant:`segmented`,fillContainer:!0,overflowMode:`menu`,showOverflowControls:!1,showGroupLabels:!1})})}function ye(e){return(0,x.jsx)(pe,{...e,className:`ds-common-table ${e.className??``}`.trim()})}var j=(0,_.createContext)(!1),M=m();function be({value:e,onChange:t,onBlur:n,options:r,label:i,placeholder:a=`Select...`,disabled:o=!1,size:s=`default`,fullWidth:c=!1,className:l=``,searchable:u=!1,searchPlaceholder:d,"aria-label":f,"aria-describedby":p,invalid:m=!1,noOptionsLabel:h=`No options`}){let[g,v]=(0,_.useState)(!1),[y,b]=(0,_.useState)(-1),[ee,te]=(0,_.useState)(``),[ne,S]=(0,_.useState)({top:0,left:0,width:0}),C=(0,_.useRef)(null),w=(0,_.useRef)(null),re=(0,_.useRef)(null),T=(0,_.useRef)(null),ie=(0,_.useRef)([]),E=(0,_.useId)(),ae=(0,_.useId)(),oe=(0,_.useId)(),D=i!=null&&i!==!1,O=r.find(t=>t.value===e),se=(0,_.useMemo)(()=>{let e=ee.trim().toLowerCase();return!u||e.length===0?r:r.filter(t=>[t.label,t.value,t.description??``].some(t=>t.toLowerCase().includes(e)))},[r,ee,u]),ce=(0,_.useCallback)((e,t)=>{for(let n=e;n>=0&&n<se.length;n+=t){let e=se[n];if(e&&!e.disabled)return n}return-1},[se]),le=e=>`${ae}-option-${String(e)}`,ue=y>=0?le(y):void 0,de=(0,_.useCallback)(()=>{if(!w.current)return;let e=w.current.getBoundingClientRect();S({top:e.bottom+4,left:e.left,width:e.width})},[]),fe=(0,_.useCallback)(()=>{if(o)return;de(),v(!0);let t=se.findIndex(t=>t.value===e&&!t.disabled);b(t===-1?ce(0,1):t)},[o,se,e,de,ce]),pe=(0,_.useCallback)(()=>{v(!1),b(-1),te(``),w.current?.focus()},[]),k=(0,_.useCallback)((e,n)=>{n&&(n.stopPropagation(),n.preventDefault()),!e.disabled&&(t(e.value),pe())},[t,pe]);(0,_.useEffect)(()=>{let e=e=>{if(!g)return;let t=e.target,n=C.current&&!C.current.contains(t),r=re.current&&!re.current.contains(t);n&&r&&pe()},t=e=>{e.key===`Escape`&&g&&pe()};return g&&(document.addEventListener(`mousedown`,e),document.addEventListener(`keydown`,t)),()=>{document.removeEventListener(`mousedown`,e),document.removeEventListener(`keydown`,t)}},[g,pe]),(0,_.useEffect)(()=>{if(!g)return;let e=()=>{de()};return window.addEventListener(`resize`,e),window.addEventListener(`scroll`,e,!0),()=>{window.removeEventListener(`resize`,e),window.removeEventListener(`scroll`,e,!0)}},[g,de]),(0,_.useEffect)(()=>{g&&u&&T.current?.focus()},[g,u]),(0,_.useEffect)(()=>{if(!g||!u)return;ie.current=[];let t=se.findIndex(t=>t.value===e&&!t.disabled);b(t>=0?t:ce(0,1))},[g,ee,u,e,se,ce]);let A=(0,_.useCallback)(e=>{if(!o)switch(e.key){case`Enter`:case` `:if(e.preventDefault(),g){let e=se[y];e&&k(e)}else fe();break;case`Escape`:e.preventDefault(),g&&pe();break;case`ArrowDown`:if(e.preventDefault(),!g)fe();else{let e=ce(y+1,1);e>=0&&b(e)}break;case`ArrowUp`:if(e.preventDefault(),g){let e=ce(y-1,-1);e>=0&&b(e)}break;case`Tab`:g&&pe()}},[o,g,y,se,ce,fe,pe,k]),me=(0,_.useCallback)(e=>{switch(e.key){case`Enter`:{e.preventDefault();let t=se[y];t&&k(t);break}case`Escape`:e.preventDefault(),e.stopPropagation(),pe();break;case`ArrowDown`:e.preventDefault(),b(e=>{let t=ce(e+1,1);return t>=0?t:e});break;case`ArrowUp`:e.preventDefault(),b(e=>{let t=ce(e-1,-1);return t>=0?t:e});break;case`Tab`:pe()}},[pe,y,k,se,ce]);(0,_.useEffect)(()=>{let e=ie.current[y];g&&y>=0&&e&&e.scrollIntoView({block:`nearest`,behavior:`smooth`})},[g,y]);let he=s==="default"?``:`select--${s}`,ge=c?`select--full-width`:``,_e=D?`select--labelled`:``,ve=D?oe:void 0,ye=d??f;return(0,x.jsxs)(`div`,{ref:C,className:`select ${he} ${ge} ${_e} ${g?`select--open`:``} ${o?`select--disabled`:``} ${l}`,children:[D&&(0,x.jsx)(`span`,{className:`select__label`,id:oe,children:i}),(0,x.jsxs)(`button`,{ref:w,type:`button`,className:`select__trigger`,onClick:()=>{g?pe():fe()},onKeyDown:A,onBlur:n,disabled:o,"aria-haspopup":`listbox`,"aria-expanded":g,"aria-controls":g?E:void 0,"aria-activedescendant":g?ue:void 0,"aria-label":f,"aria-labelledby":ve,"aria-describedby":p,"aria-invalid":m||void 0,children:[O?(0,x.jsxs)(`span`,{className:`select__value`,children:[O.icon&&(0,x.jsx)(`span`,{className:`select__value-icon`,children:O.icon}),(0,x.jsx)(`span`,{className:`select__value-label`,children:O.label})]}):(0,x.jsx)(`span`,{className:`select__placeholder`,children:a}),(0,x.jsx)(`span`,{className:`select__chevron ${g?`select__chevron--open`:``}`,children:(0,x.jsx)(`svg`,{width:`10`,height:`6`,viewBox:`0 0 10 6`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:(0,x.jsx)(`path`,{d:`M1 1L5 5L9 1`,stroke:`currentColor`,strokeWidth:`1.5`,strokeLinecap:`round`,strokeLinejoin:`round`})})})]}),g&&(0,M.createPortal)((0,x.jsxs)(`div`,{ref:re,className:`select__dropdown select__dropdown--portal`,style:{position:`fixed`,top:`${String(ne.top)}px`,left:`${String(ne.left)}px`,width:`${String(ne.width)}px`},children:[u&&(0,x.jsx)(`div`,{className:`select__search`,children:(0,x.jsx)(`input`,{ref:T,className:`select__search-input`,type:`search`,value:ee,onChange:e=>te(e.currentTarget.value),onKeyDown:me,placeholder:d,"aria-label":ye,"aria-labelledby":ye?void 0:ve,"aria-controls":E,"aria-activedescendant":ue,autoComplete:`off`,spellCheck:!1})}),(0,x.jsx)(`div`,{className:`select__options`,role:`listbox`,id:E,"aria-label":f,"aria-labelledby":ve,children:se.length===0?(0,x.jsx)(`div`,{className:`select__empty`,children:h}):se.map((t,n)=>(0,x.jsxs)(`div`,{id:le(n),ref:e=>{ie.current[n]=e},className:`select__option ${t.value===e?`select__option--selected`:``} ${n===y?`select__option--focused`:``} ${t.disabled?`select__option--disabled`:``}`,role:`option`,"aria-selected":t.value===e,"aria-disabled":t.disabled||void 0,"aria-label":t.description?t.label+`. `+t.description:void 0,onClick:e=>{k(t,e)},onMouseEnter:()=>{t.disabled||b(n)},children:[t.icon&&(0,x.jsx)(`span`,{className:`select__option-icon`,children:t.icon}),(0,x.jsxs)(`div`,{className:`select__option-content`,children:[(0,x.jsx)(`span`,{className:`select__option-label`,children:t.label}),t.description&&(0,x.jsx)(`span`,{className:`select__option-description`,children:t.description})]}),t.value===e&&(0,x.jsx)(`span`,{className:`select__option-check`,children:(0,x.jsx)(`svg`,{width:`14`,height:`10`,viewBox:`0 0 14 10`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:(0,x.jsx)(`path`,{d:`M1 5L5 9L13 1`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})})]},t.value))})]}),document.body)]})}var xe=new Map([{key:`app.title`,en:`mlxcel WebUI`,ko:`mlxcel 웹 UI`,test_id:`app-title`},{key:`app.subtitle`,en:`Local model control plane`,ko:`로컬 모델 제어판`,test_id:`app-subtitle`},{key:`nav.models`,en:`Models`,ko:`모델`,test_id:`nav-models`},{key:`nav.chat`,en:`Chat`,ko:`대화`,test_id:`nav-chat`},{key:`nav.activity`,en:`Activity`,ko:`활동`,test_id:`nav-activity`},{key:`nav.settings`,en:`Settings`,ko:`설정`,test_id:`nav-settings`},{key:`nav.gallery`,en:`Gallery`,ko:`갤러리`,test_id:`nav-gallery`},{key:`nav.primary`,en:`Primary navigation`,ko:`주 내비게이션`,test_id:`nav-primary`},{key:`nav.home`,en:`mlxcel home`,ko:`mlxcel 홈`,test_id:`nav-home`},{key:`toolbar.command`,en:`Command`,ko:`명령`,test_id:`toolbar-command`},{key:`toolbar.help`,en:`Keyboard help`,ko:`키보드 도움말`,test_id:`toolbar-help`},{key:`toolbar.logout`,en:`Clear WebUI session`,ko:`WebUI 세션 지우기`,test_id:`toolbar-logout`},{key:`toolbar.menu`,en:`Open navigation`,ko:`내비게이션 열기`,test_id:`toolbar-menu`},{key:`connection.ready`,en:`Shell loaded; local API not connected`,ko:`셸 로드됨; 로컬 API 미연결`,test_id:`connection-ready`},{key:`connection.offline`,en:`Server connection is offline`,ko:`서버 연결이 오프라인입니다`,test_id:`connection-offline`},{key:`connection.prompt.title`,en:`Connect to the local WebUI API`,ko:`로컬 WebUI API에 연결하세요`,test_id:`connection-prompt-title`},{key:`connection.prompt.body`,en:`The shell is loaded, but catalog, chat and activity data wait for the authenticated local server connection.`,ko:`셸은 로드되었지만 카탈로그, 대화, 활동 데이터는 인증된 로컬 서버 연결을 기다립니다.`,test_id:`connection-prompt-body`},{key:`connection.prompt.detail`,en:`Start mlxcel-server with --webui, enter the terminal session key when prompted, then refresh this view.`,ko:`mlxcel-server를 --webui로 시작하고, 요청되면 터미널 세션 키를 입력한 뒤 이 화면을 새로고침하세요.`,test_id:`connection-prompt-detail`},{key:`connection.footer.connected`,en:`{mode} · {status} · v{version} · seq {sequence}`,ko:`{mode} · {status} · v{version} · seq {sequence}`,test_id:`connection-footer-connected`},{key:`connection.snapshot.pending`,en:`pending`,ko:`대기 중`,test_id:`connection-snapshot-pending`},{key:`connection.authenticated.title`,en:`Authenticated local API session`,ko:`인증된 로컬 API 세션`,test_id:`connection-authenticated-title`},{key:`connection.authenticated.body`,en:`This route is connected to the shared provider. Browsing does not load models or start inference; load, unload and chat actions remain explicit.`,ko:`이 경로는 공유 provider에 연결되어 있습니다. 탐색만으로 모델을 로드하거나 추론을 시작하지 않으며, 로드·언로드·대화 동작은 명시적으로 실행됩니다.`,test_id:`connection-authenticated-body`},{key:`connection.authenticated.detail`,en:`Backend {mode}; build {version}; state {status}; catalog {count}; operations {operations}; snapshot {sequence}.`,ko:`백엔드 {mode}; 빌드 {version}; 상태 {status}; 카탈로그 {count}; 작업 {operations}; 스냅샷 {sequence}.`,test_id:`connection-authenticated-detail`},{key:`connection.error.title`,en:`Provider connection needs attention`,ko:`Provider 연결 확인 필요`,test_id:`connection-error-title`},{key:`connection.error.stale`,en:`The server snapshot changed; retry to take a fresh catalog and operation snapshot before continuing.`,ko:`서버 스냅샷이 바뀌었습니다. 계속하기 전에 다시 시도해 새 카탈로그와 작업 스냅샷을 가져오세요.`,test_id:`connection-error-stale`},{key:`connection.error.forbidden`,en:`The authenticated session is not allowed to access this UI endpoint.`,ko:`인증된 세션이 이 UI 엔드포인트에 접근할 수 없습니다.`,test_id:`connection-error-forbidden`},{key:`connection.error.unauthorized`,en:`The server rejected the current session key. Sign in again with the latest terminal key.`,ko:`서버가 현재 세션 키를 거부했습니다. 터미널에 표시된 최신 키로 다시 로그인하세요.`,test_id:`connection-error-unauthorized`},{key:`connection.error.generic`,en:`Retry the shared provider snapshot before issuing any model control action.`,ko:`모델 제어 동작을 실행하기 전에 공유 provider 스냅샷을 다시 가져오세요.`,test_id:`connection-error-generic`},{key:`connection.status.idle`,en:`idle`,ko:`대기`,test_id:`connection-status-idle`},{key:`connection.status.bootstrapping`,en:`bootstrapping`,ko:`부트스트랩`,test_id:`connection-status-bootstrapping`},{key:`connection.status.ready`,en:`ready`,ko:`준비`,test_id:`connection-status-ready`},{key:`connection.status.streaming`,en:`streaming`,ko:`스트리밍`,test_id:`connection-status-streaming`},{key:`connection.status.polling`,en:`polling`,ko:`폴링`,test_id:`connection-status-polling`},{key:`connection.status.offline`,en:`offline`,ko:`오프라인`,test_id:`connection-status-offline`},{key:`connection.status.stale`,en:`stale`,ko:`낡음`,test_id:`connection-status-stale`},{key:`connection.status.unauthorized`,en:`unauthorized`,ko:`인증 실패`,test_id:`connection-status-unauthorized`},{key:`connection.status.forbidden`,en:`forbidden`,ko:`거부됨`,test_id:`connection-status-forbidden`},{key:`connection.status.schema_mismatch`,en:`schema mismatch`,ko:`스키마 불일치`,test_id:`connection-status-schema-mismatch`},{key:`connection.status.error`,en:`error`,ko:`오류`,test_id:`connection-status-error`},{key:`auth.login`,en:`WebUI session login`,ko:`WebUI 세션 로그인`,test_id:`auth-login`},{key:`models.title`,en:`Model library`,ko:`모델 라이브러리`,test_id:`models-title`},{key:`models.empty.title`,en:`No local models yet`,ko:`아직 로컬 모델이 없습니다`,test_id:`models-empty-title`},{key:`models.empty.body`,en:`Browse, download, and load models explicitly. The shell never autoloads a checkpoint.`,ko:`모델을 명시적으로 탐색, 다운로드, 로드하세요. 셸은 체크포인트를 자동 로드하지 않습니다.`,test_id:`models-empty-body`},{key:`models.long_name`,en:`Qwen3 Very Long Local Checkpoint Name With Mixed English and 한국어 모델 이름`,ko:`Qwen3 매우 긴 로컬 체크포인트 이름과 한국어 모델 이름`,test_id:`models-long-name`},{key:`models.unsupported.reason`,en:`Vision input is unavailable for this backend; chat remains text-only.`,ko:`이 백엔드에서는 비전 입력을 사용할 수 없어 대화는 텍스트 전용입니다.`,test_id:`models-unsupported-reason`},{key:`models.load`,en:`Load`,ko:`로드`,test_id:`models-load`},{key:`models.unload`,en:`Unload`,ko:`언로드`,test_id:`models-unload`},{key:`models.status.ready`,en:`Ready`,ko:`준비됨`,test_id:`models-status-ready`},{key:`models.status.unloaded`,en:`Unloaded`,ko:`언로드됨`,test_id:`models-status-unloaded`},{key:`models.status.failed`,en:`Failed`,ko:`실패`,test_id:`models-status-failed`},{key:`models.status.loading`,en:`Loading`,ko:`로드 중`,test_id:`models-status-loading`},{key:`models.status.draining`,en:`Draining`,ko:`drain 중`,test_id:`models-status-draining`},{key:`models.status.unloading`,en:`Unloading`,ko:`언로드 중`,test_id:`models-status-unloading`},{key:`models.delete.confirm.title`,en:`Delete model from cache?`,ko:`캐시에서 모델을 삭제할까요?`,test_id:`dialog-delete-model-title`},{key:`models.delete.confirm.body`,en:`Delete {model} from the managed cache. Loaded or non-cache models cannot be deleted.`,ko:`관리 캐시에서 {model} 모델을 삭제합니다. 로드 중이거나 캐시 모델이 아니면 삭제할 수 없습니다.`,test_id:`dialog-delete-model-body`},{key:`models.delete.confirm.token_label`,en:`Type DELETE to confirm`,ko:`확인하려면 DELETE를 입력하세요`,test_id:`dialog-delete-model-token`},{key:`models.unload.confirm.body`,en:`Unload {model} after active requests drain; browser tabs will keep their selected model.`,ko:`활성 요청이 비워진 뒤 {model} 모델을 언로드합니다. 브라우저 탭의 선택 모델은 유지됩니다.`,test_id:`dialog-unload-model-body`},{key:`chat.title`,en:`Chat`,ko:`대화`,test_id:`chat-title`},{key:`chat.placeholder`,en:`Type a message; IME composition is preserved.`,ko:`메시지를 입력하세요. IME 조합은 보존됩니다.`,test_id:`chat-placeholder`},{key:`chat.streaming.status`,en:`Streaming tokens from the selected ready model.`,ko:`선택한 준비 모델에서 토큰을 스트리밍 중입니다.`,test_id:`chat-streaming-status`},{key:`chat.reasoning`,en:`Reasoning`,ko:`추론`,test_id:`chat-reasoning`},{key:`chat.tool_call`,en:`Tool call preview`,ko:`도구 호출 미리보기`,test_id:`chat-tool-call`},{key:`downloads.cancel.confirm.body`,en:`Cancel this download only after the server acknowledges worker shutdown.`,ko:`서버가 작업자 중지를 확인한 뒤에만 이 다운로드를 취소합니다.`,test_id:`dialog-cancel-download-body`},{key:`activity.title`,en:`Activity`,ko:`활동`,test_id:`activity-title`},{key:`activity.progress.indeterminate`,en:`Downloaded {bytes}; total size is not known yet.`,ko:`{bytes} 다운로드됨; 전체 크기는 아직 알 수 없습니다.`,test_id:`activity-progress-indeterminate`},{key:`activity.sse_reset`,en:`Event stream reset; take a fresh snapshot before resuming updates.`,ko:`이벤트 스트림이 재설정되었습니다. 업데이트를 재개하기 전에 새 스냅샷을 가져오세요.`,test_id:`activity-sse-reset`},{key:`settings.title`,en:`Settings`,ko:`설정`,test_id:`settings-title`},{key:`settings.appearance`,en:`Appearance preferences`,ko:`화면 표시 설정`,test_id:`settings-appearance`},{key:`settings.theme`,en:`Theme`,ko:`테마`,test_id:`settings-theme`},{key:`settings.material`,en:`Material`,ko:`재질`,test_id:`settings-material`},{key:`settings.glass_intensity`,en:`Glass intensity`,ko:`글래스 강도`,test_id:`settings-glass-intensity`},{key:`settings.reduce_motion`,en:`Reduce motion`,ko:`동작 줄이기`,test_id:`settings-reduce-motion`},{key:`settings.reduce_transparency`,en:`Reduce transparency`,ko:`투명도 줄이기`,test_id:`settings-reduce-transparency`},{key:`settings.high_contrast`,en:`High contrast`,ko:`고대비`,test_id:`settings-high-contrast`},{key:`settings.locale`,en:`Language`,ko:`언어`,test_id:`settings-locale`},{key:`settings.partial_success`,en:`Some settings changed; fields controlled by CLI or the loaded worker were left unchanged.`,ko:`일부 설정만 변경되었습니다. CLI 또는 로드된 워커가 제어하는 필드는 변경되지 않았습니다.`,test_id:`settings-partial-success`},{key:`settings.browser_only`,en:`Browser appearance only`,ko:`브라우저 표시 설정 전용`,test_id:`settings-browser-only`},{key:`settings.browser_only.body`,en:`These preferences stay in this browser and do not claim server or worker settings changed.`,ko:`이 설정은 이 브라우저에만 남으며 서버나 워커 설정 변경을 의미하지 않습니다.`,test_id:`settings-browser-only-body`},{key:`settings.clear_history.confirm.body`,en:`Clear only the explicit local browser history store; API keys are never persisted there.`,ko:`명시적으로 켠 로컬 브라우저 기록 저장소만 지웁니다. API 키는 그곳에 저장하지 않습니다.`,test_id:`dialog-clear-history-body`},{key:`state.unauthorized.title`,en:`Authentication required`,ko:`인증이 필요합니다`,test_id:`state-unauthorized-title`},{key:`state.unauthorized.body`,en:`Enter the session key printed by the local server terminal.`,ko:`로컬 서버 터미널에 한 번 표시된 세션 키를 입력하세요.`,test_id:`state-unauthorized-body`},{key:`state.offline.title`,en:`Server offline`,ko:`서버 오프라인`,test_id:`state-offline-title`},{key:`state.offline.body`,en:`The shell is available, but model data waits for the local API.`,ko:`셸은 사용할 수 있지만 모델 데이터는 로컬 API를 기다립니다.`,test_id:`state-offline-body`},{key:`state.schema_mismatch.title`,en:`UI schema mismatch`,ko:`UI 스키마 불일치`,test_id:`state-schema-mismatch-title`},{key:`state.schema_mismatch.body`,en:`The shell loaded, but the server reports a different UI API schema version; refresh after updating the bundle or server.`,ko:`셸은 로드되었지만 서버가 다른 UI API 스키마 버전을 보고했습니다. 번들이나 서버를 업데이트한 뒤 새로고침하세요.`,test_id:`state-schema-mismatch-body`},{key:`command.title`,en:`Command palette`,ko:`명령 팔레트`,test_id:`command-title`},{key:`select.search`,en:`Search options`,ko:`옵션 검색`,test_id:`select-search`},{key:`select.no_options`,en:`No matching options.`,ko:`일치하는 옵션이 없습니다.`,test_id:`select-no-options`},{key:`command.search`,en:`Search commands`,ko:`명령 검색`,test_id:`command-search`},{key:`command.no_results`,en:`No commands match this search.`,ko:`검색과 일치하는 명령이 없습니다.`,test_id:`command-no-results`},{key:`help.title`,en:`Keyboard shortcuts`,ko:`키보드 단축키`,test_id:`help-title`},{key:`help.body`,en:`Command opens search. Escape closes overlays. Brackets move navigation only while the sidebar has focus.`,ko:`Command는 검색을 엽니다. Escape는 오버레이를 닫습니다. 대괄호는 사이드바에 포커스가 있을 때만 내비게이션을 이동합니다.`,test_id:`help-body`},{key:`gallery.title`,en:`Design system gallery`,ko:`디자인 시스템 갤러리`,test_id:`gallery-title`},{key:`gallery.subtitle`,en:`Shared tokens, controls, states, and viewport fixtures for page implementers.`,ko:`페이지 구현자를 위한 공유 토큰, 컨트롤, 상태, 뷰포트 픽스처입니다.`,test_id:`gallery-subtitle`},{key:`gallery.long_cjk`,en:`Long English and 한국어 labels truncate with accessible full labels.`,ko:`긴 English 및 한국어 레이블은 접근 가능한 전체 레이블을 유지하며 줄임표 처리됩니다.`,test_id:`gallery-long-cjk`},{key:`model.selected.none`,en:`No model selected`,ko:`선택한 모델 없음`,test_id:`model-selected-none`},{key:`common.unavailable`,en:`Unavailable until server adapters connect`,ko:`서버 어댑터 연결 전에는 사용할 수 없습니다`,test_id:`common-unavailable`},{key:`common.cancel`,en:`Cancel`,ko:`취소`,test_id:`common-cancel`},{key:`common.delete`,en:`Delete`,ko:`삭제`,test_id:`common-delete`},{key:`common.retry`,en:`Retry`,ko:`다시 시도`,test_id:`common-retry`},{key:`common.reload`,en:`Reload`,ko:`새로고침`,test_id:`common-reload`},{key:`common.enter_key`,en:`Enter key`,ko:`키 입력`,test_id:`common-enter-key`},{key:`common.add_model`,en:`Add model`,ko:`모델 추가`,test_id:`common-add-model`},{key:`common.send`,en:`Send`,ko:`보내기`,test_id:`common-send`},{key:`routes.models.eyebrow`,en:`Local library`,ko:`로컬 라이브러리`,test_id:`routes-models-eyebrow`},{key:`routes.chat.eyebrow`,en:`Conversation`,ko:`대화`,test_id:`routes-chat-eyebrow`},{key:`routes.activity.eyebrow`,en:`Operations`,ko:`작업`,test_id:`routes-activity-eyebrow`},{key:`routes.settings.eyebrow`,en:`Browser only`,ko:`브라우저 전용`,test_id:`routes-settings-eyebrow`},{key:`adapters.pending.title`,en:`Local API is not connected yet`,ko:`로컬 API가 아직 연결되지 않았습니다`,test_id:`adapters-pending-title`},{key:`adapters.pending.body`,en:`This route shows the production shell only. Catalog, lifecycle, and runtime data will come from the shared typed client in the integration wave.`,ko:`이 경로는 프로덕션 셸만 보여줍니다. 카탈로그, 수명주기, 런타임 데이터는 통합 웨이브의 공유 typed client에서 제공됩니다.`,test_id:`adapters-pending-body`},{key:`chat.pending.body`,en:`Chat controls stay disabled until an authenticated ready model is selected by the shared state provider.`,ko:`공유 상태 provider가 인증된 준비 모델을 선택하기 전까지 대화 컨트롤은 비활성화됩니다.`,test_id:`chat-pending-body`},{key:`activity.empty.title`,en:`No active operations`,ko:`활성 작업 없음`,test_id:`activity-empty-title`},{key:`activity.empty.body`,en:`Loads, downloads, drains and SSE resets appear here after the lifecycle adapter connects.`,ko:`수명주기 어댑터가 연결된 뒤 로드, 다운로드, drain, SSE reset이 여기에 표시됩니다.`,test_id:`activity-empty-body`},{key:`gallery.tab.sections`,en:`Gallery sections`,ko:`갤러리 섹션`,test_id:`gallery-tab-sections`},{key:`gallery.tab.controls`,en:`Controls`,ko:`컨트롤`,test_id:`gallery-tab-controls`},{key:`gallery.tab.states`,en:`States`,ko:`상태`,test_id:`gallery-tab-states`},{key:`gallery.tab.data`,en:`Data display`,ko:`데이터 표시`,test_id:`gallery-tab-data`},{key:`gallery.controls.title`,en:`Buttons and fields`,ko:`버튼과 필드`,test_id:`gallery-controls-title`},{key:`gallery.controls.primary`,en:`Primary`,ko:`기본`,test_id:`gallery-controls-primary`},{key:`gallery.controls.secondary`,en:`Secondary`,ko:`보조`,test_id:`gallery-controls-secondary`},{key:`gallery.controls.danger`,en:`Danger`,ko:`위험`,test_id:`gallery-controls-danger`},{key:`gallery.controls.busy`,en:`Busy`,ko:`진행 중`,test_id:`gallery-controls-busy`},{key:`gallery.field.repo`,en:`Repository ID`,ko:`저장소 ID`,test_id:`gallery-field-repo`},{key:`gallery.field.repo_error`,en:`Use owner/name without a URL.`,ko:`URL 없이 owner/name 형식을 사용하세요.`,test_id:`gallery-field-repo-error`},{key:`gallery.field.repo_hint`,en:`Sample only; downloads require the later lifecycle adapter.`,ko:`샘플 전용입니다. 다운로드는 후속 수명주기 어댑터가 필요합니다.`,test_id:`gallery-field-repo-hint`},{key:`gallery.progress.measured`,en:`Sample download progress`,ko:`예시 다운로드 진행률`,test_id:`gallery-progress-measured`},{key:`gallery.select.native`,en:`Shared select combobox`,ko:`공통 선택 콤보박스`,test_id:`gallery-select-native`},{key:`gallery.overlays.title`,en:`Overlays`,ko:`오버레이`,test_id:`gallery-overlays-title`},{key:`gallery.tooltip`,en:`Tooltips are descriptive only`,ko:`툴팁은 설명 전용입니다`,test_id:`gallery-tooltip`},{key:`gallery.dialog.open`,en:`Open dialog`,ko:`다이얼로그 열기`,test_id:`gallery-dialog-open`},{key:`gallery.states.load_failed`,en:`Load failed`,ko:`로드 실패`,test_id:`gallery-states-load-failed`},{key:`gallery.states.load_failed_body`,en:`The worker exited before becoming ready; retry after checking logs.`,ko:`워커가 준비되기 전에 종료되었습니다. 로그 확인 후 다시 시도하세요.`,test_id:`gallery-states-load-failed-body`},{key:`gallery.data.title`,en:`Model rows`,ko:`모델 행`,test_id:`gallery-data-title`},{key:`gallery.data.name`,en:`Name`,ko:`이름`,test_id:`gallery-data-name`},{key:`gallery.data.status`,en:`Status`,ko:`상태`,test_id:`gallery-data-status`},{key:`gallery.data.rate`,en:`Rate`,ko:`속도`,test_id:`gallery-data-rate`},{key:`gallery.data.inspector`,en:`Inspector`,ko:`인스펙터`,test_id:`gallery-data-inspector`},{key:`gallery.sample.reasoning_body`,en:`Reasoning text is visually separated from answer content.`,ko:`추론 텍스트는 답변 본문과 시각적으로 분리됩니다.`,test_id:`gallery-sample-reasoning-body`},{key:`gallery.sample.tool_preview`,en:`{ "tool": "display_only" }`,ko:`{ "tool": "표시_전용" }`,test_id:`gallery-sample-tool-preview`},{key:`common.close`,en:`Close`,ko:`닫기`,test_id:`common-close`},{key:`settings.theme.system`,en:`System`,ko:`시스템`,test_id:`settings-theme-system`},{key:`settings.theme.light`,en:`Light`,ko:`라이트`,test_id:`settings-theme-light`},{key:`settings.theme.dark`,en:`Dark`,ko:`다크`,test_id:`settings-theme-dark`},{key:`settings.material.glass`,en:`Glass`,ko:`글래스`,test_id:`settings-material-glass`},{key:`settings.material.tinted`,en:`Tinted`,ko:`틴트`,test_id:`settings-material-tinted`},{key:`settings.material.opaque`,en:`Opaque`,ko:`불투명`,test_id:`settings-material-opaque`},{key:`settings.locale.en`,en:`English`,ko:`영어`,test_id:`settings-locale-en`},{key:`settings.locale.ko`,en:`Korean`,ko:`한국어`,test_id:`settings-locale-ko`},{key:`settings.high_contrast.system`,en:`Follow system`,ko:`시스템 따르기`,test_id:`settings-high-contrast-system`},{key:`settings.high_contrast.on`,en:`On`,ko:`켬`,test_id:`settings-high-contrast-on`},{key:`settings.high_contrast.off`,en:`Off`,ko:`끔`,test_id:`settings-high-contrast-off`},{key:`gallery.issue`,en:`Issue #1843`,ko:`이슈 #1843`,test_id:`gallery-issue`},{key:`gallery.hover_focus`,en:`Hover or focus`,ko:`호버 또는 포커스`,test_id:`gallery-hover-focus`},{key:`gallery.download`,en:`Download`,ko:`다운로드`,test_id:`gallery-download`},{key:`gallery.lifecycle.samples`,en:`Lifecycle sample badges`,ko:`수명주기 샘플 배지`,test_id:`gallery-lifecycle-samples`},{key:`gallery.sample.caption`,en:`Sample model table`,ko:`샘플 모델 표`,test_id:`gallery-sample-caption`},{key:`gallery.sample.label`,en:`Sample fixture`,ko:`샘플 픽스처`,test_id:`gallery-sample-label`},{key:`login.token.label`,en:`Session key`,ko:`세션 키`,test_id:`login-token-label`},{key:`login.token.help`,en:`Use the key printed once by the local server. It stays in memory only.`,ko:`로컬 서버가 한 번 출력한 키를 사용하세요. 키는 메모리에만 유지됩니다.`,test_id:`login-token-help`},{key:`login.submit`,en:`Connect`,ko:`연결`,test_id:`login-submit`},{key:`login.logout`,en:`Clear key`,ko:`키 지우기`,test_id:`login-logout`},{key:`login.error.sample`,en:`Sample error: the key was not accepted by the local API.`,ko:`샘플 오류: 로컬 API가 키를 허용하지 않았습니다.`,test_id:`login-error-sample`},{key:`login.error.wrong_key`,en:`The session key was rejected. Copy the latest key printed by the local server terminal.`,ko:`세션 키가 거부되었습니다. 로컬 서버 터미널에 표시된 최신 키를 복사하세요.`,test_id:`login-error-wrong-key`},{key:`login.error.offline`,en:`Could not reach the local WebUI API. Check that mlxcel-server is still running with --webui.`,ko:`로컬 WebUI API에 연결할 수 없습니다. mlxcel-server가 --webui로 계속 실행 중인지 확인하세요.`,test_id:`login-error-offline`},{key:`login.error.forbidden`,en:`The key authenticated, but this UI endpoint is forbidden for the current server session.`,ko:`키 인증은 되었지만 현재 서버 세션에서 이 UI 엔드포인트가 금지되어 있습니다.`,test_id:`login-error-forbidden`},{key:`login.error.schema`,en:`The server response does not match this bundled UI schema. Reload after updating the server or bundle.`,ko:`서버 응답이 번들된 UI 스키마와 일치하지 않습니다. 서버나 번들을 업데이트한 뒤 새로고침하세요.`,test_id:`login-error-schema`},{key:`login.error.generic`,en:`The local API could not complete authentication. Retry with the latest terminal key.`,ko:`로컬 API 인증을 완료할 수 없습니다. 터미널에 표시된 최신 키로 다시 시도하세요.`,test_id:`login-error-generic`},{key:`gallery.delete_token`,en:`DELETE`,ko:`DELETE`,test_id:`gallery-delete-token`},{key:`models.library.subtitle`,en:`Inspect local checkpoints. Selecting never loads a model.`,ko:`로컬 체크포인트를 확인하세요. 선택만으로 모델을 로드하지 않습니다.`,test_id:`models-library-subtitle`},{key:`models.library.search`,en:`Search local models`,ko:`로컬 모델 검색`,test_id:`models-library-search`},{key:`models.library.source`,en:`Source`,ko:`소스`,test_id:`models-library-source`},{key:`models.library.task`,en:`Task`,ko:`작업`,test_id:`models-library-task`},{key:`models.library.status`,en:`Lifecycle`,ko:`수명주기`,test_id:`models-library-status`},{key:`models.library.sort`,en:`Sort`,ko:`정렬`,test_id:`models-library-sort`},{key:`models.library.name`,en:`Model`,ko:`모델`,test_id:`models-library-name`},{key:`models.library.all`,en:`All`,ko:`전체`,test_id:`models-library-all`},{key:`models.library.add`,en:`Add Model`,ko:`모델 추가`,test_id:`models-library-add`},{key:`models.library.rescan`,en:`Rescan local roots`,ko:`로컬 경로 다시 검색`,test_id:`models-library-rescan`},{key:`models.library.refresh`,en:`Refresh server state`,ko:`서버 상태 새로고침`,test_id:`models-library-refresh`},{key:`models.library.roots`,en:`Configured roots`,ko:`설정된 경로`,test_id:`models-library-roots`},{key:`models.library.roots_help`,en:`To change roots, restart the server with your local directory. This command is an example, not an executed action.`,ko:`경로를 변경하려면 로컬 디렉터리를 지정하여 서버를 다시 시작하세요. 아래 명령은 예시이며 자동 실행되지 않습니다.`,test_id:`models-library-roots-help`},{key:`models.library.single`,en:`Single-model mode: lifecycle and cache changes are read-only. Restart without -m to manage models.`,ko:`단일 모델 모드: 수명주기 및 캐시는 읽기 전용입니다. 모델을 관리하려면 -m 없이 다시 시작하세요.`,test_id:`models-library-single`},{key:`models.library.unknown`,en:`Unknown`,ko:`알 수 없음`,test_id:`models-library-unknown`},{key:`models.library.yes`,en:`Yes`,ko:`예`,test_id:`models-library-yes`},{key:`models.library.no`,en:`No`,ko:`아니요`,test_id:`models-library-no`},{key:`models.library.inspect`,en:`Inspect {name}`,ko:`{name} 자세히 보기`,test_id:`models-library-inspect`},{key:`models.library.selected`,en:`Selected`,ko:`선택됨`,test_id:`models-library-selected`},{key:`models.library.details`,en:`Model details`,ko:`모델 상세`,test_id:`models-library-details`},{key:`models.library.support`,en:`Support / completeness`,ko:`지원 / 완전성`,test_id:`models-library-support`},{key:`models.library.supported`,en:`Supported architecture`,ko:`지원되는 아키텍처`,test_id:`models-library-supported`},{key:`models.library.unsupported`,en:`Unsupported architecture`,ko:`지원되지 않는 아키텍처`,test_id:`models-library-unsupported`},{key:`models.library.complete`,en:`Complete files`,ko:`파일 완전함`,test_id:`models-library-complete`},{key:`models.library.incomplete`,en:`Incomplete files`,ko:`불완전한 파일`,test_id:`models-library-incomplete`},{key:`models.library.architecture`,en:`Architecture`,ko:`아키텍처`,test_id:`models-library-architecture`},{key:`models.library.backend`,en:`Runnable on this backend`,ko:`현재 백엔드 실행 가능`,test_id:`models-library-backend`},{key:`models.library.tested`,en:`Checkpoint validated`,ko:`검증된 체크포인트`,test_id:`models-library-tested`},{key:`models.library.tested_help`,en:`Family support does not prove this checkpoint was tested.`,ko:`계열 지원은 해당 체크포인트의 검증을 의미하지 않습니다.`,test_id:`models-library-tested-help`},{key:`models.library.disk`,en:`Disk size`,ko:`디스크 크기`,test_id:`models-library-disk`},{key:`models.library.memory`,en:`Estimated memory (not measured)`,ko:`예상 메모리 (측정값 아님)`,test_id:`models-library-memory`},{key:`models.library.context`,en:`Effective context (tokens)`,ko:`실제 컨텍스트 (토큰)`,test_id:`models-library-context`},{key:`models.library.profile`,en:`Next-load profile`,ko:`다음 로드 프로필`,test_id:`models-library-profile`},{key:`models.library.defaults`,en:`Server defaults; see Settings for scope and overrides.`,ko:`서버 기본값입니다. 범위와 재정의는 설정에서 확인하세요.`,test_id:`models-library-defaults`},{key:`models.library.error`,en:`Action could not complete`,ko:`작업을 완료하지 못함`,test_id:`models-library-error`},{key:`models.library.last_error`,en:`Last server error`,ko:`최근 서버 오류`,test_id:`models-library-last-error`},{key:`models.library.chat`,en:`Use in Chat`,ko:`채팅에서 사용`,test_id:`models-library-chat`},{key:`models.library.chat_reason`,en:`Chat requires a ready provider with an available chat capability. Other tasks remain available through their API.`,ko:`채팅에는 준비된 제공자와 사용 가능한 채팅 기능이 필요합니다. 다른 작업은 API로 사용하세요.`,test_id:`models-library-chat-reason`},{key:`models.library.api`,en:`API task documentation`,ko:`API 작업 문서`,test_id:`models-library-api`},{key:`models.library.delete`,en:`Delete cached files`,ko:`캐시 파일 삭제`,test_id:`models-library-delete`},{key:`models.library.delete_body`,en:`Permanently delete the managed cache entry {name} ({source}). This removes files from disk, unlike Unload. Type the model name to confirm.`,ko:`관리 캐시 항목 {name} ({source})을 영구 삭제합니다. 언로드와 달리 디스크 파일을 제거합니다. 모델 이름을 입력하여 확인하세요.`,test_id:`models-library-delete-body`},{key:`models.library.confirm_name`,en:`Exact cache model ID to confirm`,ko:`확인용 정확한 캐시 모델 ID`,test_id:`models-library-confirm-name`},{key:`models.library.unload_body`,en:`Unload {name} ({source}): {count} active requests must drain before the worker exits. Files remain on disk. A timeout is a failure, not a completed unload.`,ko:`{name} ({source}) 언로드: 활성 요청 {count}개를 배출한 뒤 작업자가 종료됩니다. 디스크 파일은 유지됩니다. 시간 초과는 완료가 아닌 실패입니다.`,test_id:`models-library-unload-body`},{key:`models.library.confirm`,en:`Confirm`,ko:`확인`,test_id:`models-library-confirm`},{key:`models.library.cancel`,en:`Cancel`,ko:`취소`,test_id:`models-library-cancel`},{key:`models.library.repo`,en:`Public HuggingFace repo ID`,ko:`공개 HuggingFace 저장소 ID`,test_id:`models-library-repo`},{key:`models.library.revision`,en:`Revision (optional)`,ko:`리비전 (선택)`,test_id:`models-library-revision`},{key:`models.library.network`,en:`Only this explicit action contacts HuggingFace. Files go to the server-managed cache below. Size is unknown until the server resolves file metadata. Search never contacts external services.`,ko:`이 작업을 명시적으로 실행할 때만 HuggingFace에 연결합니다. 파일은 아래 서버 관리 캐시에 저장됩니다. 크기는 서버가 파일 메타데이터를 확인할 때까지 알 수 없습니다. 검색은 외부 서비스에 연결하지 않습니다.`,test_id:`models-library-network`},{key:`models.library.public`,en:`This is a public, ungated repository`,ko:`공개된 비게이트 저장소입니다`,test_id:`models-library-public`},{key:`models.library.private`,en:`Private or gated downloads are unavailable in the WebUI. Use an authenticated HuggingFace CLI outside the server, then rescan a configured local root. Do not paste credentials here.`,ko:`비공개 또는 게이트 다운로드는 WebUI에서 지원하지 않습니다. 서버 외부에서 인증된 HuggingFace CLI를 사용한 뒤 설정된 로컬 경로를 다시 검색하세요. 여기에 인증 정보를 입력하지 마세요.`,test_id:`models-library-private`},{key:`models.library.repo_invalid`,en:`Enter owner/repository, not a URL or a local path.`,ko:`URL이나 로컬 경로가 아닌 소유자/저장소를 입력하세요.`,test_id:`models-library-repo-invalid`},{key:`models.library.operations`,en:`Library operations`,ko:`라이브러리 작업`,test_id:`models-library-operations`},{key:`models.library.pending`,en:`Request outcome is being reconciled. Do not resubmit; refresh to observe the server operation.`,ko:`요청 결과를 확인하고 있습니다. 재제출하지 말고 새로고침하여 서버 작업을 확인하세요.`,test_id:`models-library-pending`},{key:`models.library.retry_download`,en:`Retry download`,ko:`다운로드 재시도`,test_id:`models-library-retry-download`},{key:`models.library.cancel_download`,en:`Cancel download`,ko:`다운로드 취소`,test_id:`models-library-cancel-download`},{key:`models.library.cancel_body`,en:`Cancel the download of {name}? Unpublished partial files are not a usable model.`,ko:`{name} 다운로드를 취소할까요? 공개되지 않은 부분 파일은 사용할 수 있는 모델이 아닙니다.`,test_id:`models-library-cancel-body`},{key:`models.library.progress`,en:`Download progress`,ko:`다운로드 진행률`,test_id:`models-library-progress`},{key:`models.library.worker`,en:`Worker exit observed`,ko:`작업자 종료 확인`,test_id:`models-library-worker`},{key:`models.library.stale`,en:`State changed or is not current. Refresh, inspect the latest revision, and explicitly confirm again.`,ko:`상태가 변경되었거나 최신이 아닙니다. 새로고침하고 최신 리비전을 확인한 뒤 다시 명시적으로 확인하세요.`,test_id:`models-library-stale`},{key:`models.library.capacity`,en:`Capacity / conflicting operation`,ko:`용량 / 작업 충돌`,test_id:`models-library-capacity`},{key:`models.library.capacity_body`,en:`Do not repeatedly load. Review active models below. If capacity is full, explicitly choose an idle ready model to unload before this load. Busy models are never offered as eviction targets. Eviction is not rolled back if the new load fails.`,ko:`반복해서 로드하지 마세요. 아래 활성 모델을 확인하세요. 용량이 가득 차면 이번 로드 전에 언로드할 유휴 준비 모델을 명시적으로 선택하세요. 사용 중인 모델은 교체 대상으로 제공되지 않습니다. 새 로드 실패 시 교체는 되돌려지지 않습니다.`,test_id:`models-library-capacity-body`},{key:`models.library.eviction`,en:`Explicit eviction target`,ko:`명시적 교체 대상`,test_id:`models-library-eviction`},{key:`models.library.choose`,en:`Choose a model`,ko:`모델 선택`,test_id:`models-library-choose`},{key:`models.library.evict_load`,en:`Unload target and load`,ko:`대상 언로드 후 로드`,test_id:`models-library-evict-load`},{key:`models.library.active`,en:`Active requests`,ko:`활성 요청`,test_id:`models-library-active`},{key:`models.library.page`,en:`Page {page} of {pages} · {count} models`,ko:`{pages}페이지 중 {page} · 모델 {count}개`,test_id:`models-library-page`},{key:`models.library.previous`,en:`Previous page`,ko:`이전 페이지`,test_id:`models-library-previous`},{key:`models.library.next`,en:`Next page`,ko:`다음 페이지`,test_id:`models-library-next`},{key:`models.library.filtered`,en:`No matching local models`,ko:`일치하는 로컬 모델 없음`,test_id:`models-library-filtered`},{key:`models.library.filtered_body`,en:`Change the local search or filters.`,ko:`로컬 검색어나 필터를 변경하세요.`,test_id:`models-library-filtered-body`},{key:`models.library.waiting`,en:`Waiting for an authoritative catalog snapshot`,ko:`서버의 카탈로그 스냅샷을 기다리는 중`,test_id:`models-library-waiting`},{key:`models.library.sort_name`,en:`Name (A–Z)`,ko:`이름 (오름차순)`,test_id:`models-library-sort-name`},{key:`models.library.sort_status`,en:`Lifecycle`,ko:`수명주기`,test_id:`models-library-sort-status`},{key:`models.library.sort_source`,en:`Source`,ko:`소스`,test_id:`models-library-sort-source`},{key:`models.library.sort_task`,en:`Task`,ko:`작업`,test_id:`models-library-sort-task`},{key:`models.library.unavailable`,en:`Not available for this server or model`,ko:`현재 서버 또는 모델에서 사용할 수 없음`,test_id:`models-library-unavailable`},{key:`models.library.permission`,en:`Root could not be read. Check server directory permissions, then rescan.`,ko:`경로를 읽지 못했습니다. 서버 디렉터리 권한을 확인한 뒤 다시 검색하세요.`,test_id:`models-library-permission`},{key:`models.library.copy`,en:`Copy command`,ko:`명령 복사`,test_id:`models-library-copy`},{key:`models.library.copied`,en:`Copied`,ko:`복사됨`,test_id:`models-library-copied`},{key:`models.library.copy_failed`,en:`Select and copy the command below.`,ko:`아래 명령을 선택하여 복사하세요.`,test_id:`models-library-copy-failed`},{key:`models.library.accepted`,en:`Accepted; waiting for observed server state.`,ko:`접수됨. 서버에서 관측된 상태를 기다리는 중입니다.`,test_id:`models-library-accepted`},{key:`models.library.operation_state`,en:`Operation state`,ko:`작업 상태`,test_id:`models-library-operation-state`}].map(e=>[e.key,e]));function N(e,t,n={}){let r=xe.get(t);return r?r[e].replace(/\{(\w+)\}/g,(e,t)=>n[t]??`{${t}}`):t}function P(e){return xe.get(e)?.test_id??e.replaceAll(`.`,`-`)}function F(e){let t=(0,_.useContext)(j),n=(0,_.useId)(),r=`${n}-hint`,i=`${n}-error`,a=[e.hint?r:null,e.error?i:null].filter(Boolean).join(` `)||void 0,o=e.disabled||e.busy,s=(0,x.jsxs)(x.Fragment,{children:[e.hint?(0,x.jsx)(`small`,{id:r,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:i,"data-tone":`error`,children:e.error}):null]});return t?(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:n,"data-disabled":o||void 0,children:[(0,x.jsx)(`span`,{children:e.label}),(0,x.jsx)(`select`,{id:n,value:e.value,disabled:o,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":a,"data-testid":e.testId,onChange:t=>e.onChange(t.currentTarget.value),children:e.options.map(e=>(0,x.jsx)(`option`,{...e,children:e.label},e.value))}),s]}):(0,x.jsxs)(`div`,{ref:t=>{let n=t?.querySelector(`.select__trigger`);n?.setAttribute(`role`,`combobox`),e.busy?n?.setAttribute(`aria-busy`,`true`):n?.removeAttribute(`aria-busy`)},onKeyDownCapture:e=>{e.nativeEvent.isComposing&&e.stopPropagation()},className:`ds-field ds-common-select`,"data-disabled":o||void 0,"aria-busy":e.busy||void 0,"data-testid":e.testId,children:[(0,x.jsx)(be,{value:e.value,options:e.options,onChange:e.onChange,label:e.label,disabled:o,invalid:!!e.error,"aria-describedby":a,fullWidth:!0,noOptionsLabel:N(e.locale??`en`,`select.no_options`),searchPlaceholder:N(e.locale??`en`,`select.search`)}),s]})}function I(e){let t=(0,_.useId)(),n=`${t}-label`,r=`${t}-hint`,i=`${t}-error`,a=[e.hint?r:null,e.error?i:null].filter(Boolean).join(` `)||void 0;return(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:t,"data-disabled":e.disabled||e.busy||void 0,children:[(0,x.jsx)(`span`,{id:n,children:e.label}),(0,x.jsx)(`input`,{id:t,"aria-labelledby":n,value:e.value,placeholder:e.placeholder,disabled:e.disabled||e.busy,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":a,"data-testid":e.testId,onChange:t=>e.onChange?.(t.currentTarget.value)}),e.hint?(0,x.jsx)(`small`,{id:r,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:i,"data-tone":`error`,children:e.error}):null]})}function L(e){return(0,x.jsxs)(`section`,{className:`ds-banner`,"data-tone":e.tone??`error`,role:e.tone===`info`?`status`:`alert`,"data-testid":e.testId,children:[(0,x.jsx)(`strong`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),e.action]})}function Se(e){let t=(0,_.useRef)(null),n=(0,_.useRef)(null),r=(0,_.useId)(),i=e.labelledBy??r,a=(0,_.useRef)(!1),o=(0,_.useRef)(e.onClose);return o.current=e.onClose,(0,_.useEffect)(()=>{let r=t.current;r&&(e.open&&!r.open&&(n.current=document.activeElement instanceof HTMLElement?document.activeElement:null,r.showModal(),r.querySelector(`[data-autofocus], button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])`)?.focus()),!e.open&&r.open&&(a.current=!0,r.close()))},[e.open]),(0,_.useEffect)(()=>{let e=t.current;if(!e)return;let r=()=>{let t=n.current;Ce(e,t),n.current=null},i=()=>{a.current?a.current=!1:o.current(),window.setTimeout(r,0)},s=t=>{if(t.key!==`Tab`)return;let n=Array.from(e.querySelectorAll(`button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])`)).filter(e=>e.offsetParent!==null||e===document.activeElement);if(n.length===0)return;let r=n[0],i=n[n.length-1];t.shiftKey&&document.activeElement===r?(t.preventDefault(),i.focus()):!t.shiftKey&&document.activeElement===i&&(t.preventDefault(),r.focus())};return e.addEventListener(`close`,i),e.addEventListener(`keydown`,s),()=>{e.removeEventListener(`close`,i),e.removeEventListener(`keydown`,s);let t=n.current;window.setTimeout(()=>Ce(e,t),0)}},[]),(0,x.jsxs)(`dialog`,{className:`ds-dialog ${e.className??``}`.trim(),"data-position":e.position??`center`,ref:t,"aria-labelledby":i,"data-testid":e.testId,children:[(0,x.jsxs)(`header`,{children:[(0,x.jsx)(`h2`,{id:i,children:e.title}),(0,x.jsx)(A,{label:e.closeLabel??`Close`,icon:`close`,onClick:()=>{let e=n.current;a.current=!0,o.current(),t.current?.close(),window.setTimeout(()=>{let n=t.current;n&&Ce(n,e)},0)},"data-testid":`dialog-close`})]}),(0,x.jsx)(`div`,{children:(0,x.jsx)(j.Provider,{value:!0,children:e.children})})]})}function Ce(e,t){if(e.isConnected&&e.open)return;let n=document.activeElement;if(n&&n!==document.body&&n!==document.documentElement&&!e.contains(n)||document.querySelector(`dialog[open]`))return;let r=e=>!!e?.isConnected&&e!==document.body&&!e.matches(`:disabled, [aria-disabled="true"]`)&&!e.closest(`[inert], [hidden]`);if(r(t)){t.focus();return}let i=document.querySelector(`[data-dialog-focus-fallback]`)??document.querySelector(`main button:not(:disabled)`);r(i)&&i.focus()}function we(e){return(0,x.jsx)(Se,{...e})}function Te(e){return(0,x.jsx)(Se,{...e,position:`left`,className:`ds-sheet ${e.className??``}`.trim()})}function Ee(e){let t=(0,_.useId)();return(0,x.jsxs)(`span`,{className:`ds-tooltip-wrap`,children:[_.cloneElement(e.children,{"aria-describedby":t}),(0,x.jsx)(`span`,{className:`ds-tooltip`,role:`tooltip`,id:t,children:e.label})]})}function De(e){return(0,x.jsxs)(`aside`,{className:`ds-inspector`,"aria-label":e.title,children:[(0,x.jsx)(`h2`,{children:e.title}),e.children]})}function Oe(e){return(0,x.jsxs)(`table`,{className:`ds-table`,children:[(0,x.jsx)(`caption`,{children:e.caption}),e.children]})}function ke(e){return(0,x.jsx)(`ul`,{className:`ds-list`,"aria-label":e.label,children:e.children})}function Ae(e){let[t,n]=_.useState(``),r=_.useId();return _.useEffect(()=>()=>n(``),[]),(0,x.jsxs)(`form`,{className:`ds-login`,"data-testid":e.testId,autoComplete:`off`,onSubmit:r=>{if(r.preventDefault(),e.busy||t.trim().length===0)return;let i=t;n(``),e.onSubmit(i)},children:[(0,x.jsx)(`h2`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:e.tokenLabel}),(0,x.jsx)(`input`,{type:`password`,autoComplete:`off`,spellCheck:!1,autoCapitalize:`none`,autoCorrect:`off`,value:t,disabled:e.busy,"aria-invalid":e.error?`true`:void 0,"aria-describedby":r,onChange:e=>n(e.currentTarget.value)}),(0,x.jsx)(`small`,{id:r,"data-tone":e.error?`error`:`hint`,children:e.error??e.tokenHelp})]}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(k,{tone:`primary`,type:`submit`,busy:e.busy,disabled:t.trim().length===0,children:e.submitLabel}),e.onLogout&&e.logoutLabel?(0,x.jsx)(k,{type:`button`,onClick:()=>{n(``),e.onLogout?.()},children:e.logoutLabel}):null]})]})}function je(e){return(0,x.jsx)(L,{tone:`error`,title:e.title,body:e.body,action:(0,x.jsxs)(k,{onClick:e.onRecover,children:[(0,x.jsx)(te,{name:`schema`}),e.actionLabel]})})}function Me(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Malformed settings response.`);return e}function Ne(e){if(typeof e!=`string`||!/^[0-9a-f]{64}$/.test(e))throw Error(`Malformed settings fingerprint.`);return e}function Pe(e){let t=Me(e);if(!Array.isArray(t.schema)||t.schema.length>256)throw Error(`Malformed settings schema.`);let n=new Set;return{schema:t.schema.map(e=>{let t=Me(e);if(typeof t.name!=`string`||n.has(t.name)||typeof t.type!=`string`||![`bool`,`int`,`int_or_null`,`float`,`str`,`str_or_null`,`array`,`object`,`object_or_null`].includes(t.type)||typeof t.mutable!=`boolean`||typeof t.help!=`string`||!(t.allowed===null||Array.isArray(t.allowed)&&t.allowed.every(e=>typeof e==`string`))||t.reason!==void 0&&typeof t.reason!=`string`||!Object.hasOwn(t,`default`))throw Error(`Malformed setting specification.`);return n.add(t.name),t}),current:Me(t.current),fingerprint:Ne(t.fingerprint)}}function Fe(e){let t=Me(e);if(!Array.isArray(t.rejected)||!t.rejected.every(e=>{let t=Me(e);return typeof t.name==`string`&&typeof t.reason==`string`}))throw Error(`Malformed settings rejection list.`);return{applied:Me(t.applied),rejected:t.rejected,current:Me(t.current),fingerprint:Ne(t.fingerprint)}}function Ie(e){let t=Me(e),n=Me(t.default_generation_settings),r=e=>typeof e==`number`&&Number.isSafeInteger(e)&&e>0?e:null;return{nCtx:r(n.n_ctx),kvCacheMode:typeof t.kv_cache_mode==`string`&&t.kv_cache_mode.length<=64&&/^[a-z0-9+_-]+$/.test(t.kv_cache_mode)?t.kv_cache_mode:null,totalSlots:r(t.total_slots),geometry:t.geometry===void 0?null:Me(t.geometry)}}function Le(e,t){if(!e.mutable)throw Error(e.reason??`Restart required.`);let n=t;if(![`str`,`str_or_null`].includes(e.type)||e.type===`str_or_null`&&t===`null`)try{n=JSON.parse(t)}catch{throw Error(`Enter a valid ${e.type} value.`)}let r=e.type.endsWith(`_or_null`),i=e.type.replace(`_or_null`,``);if(!(r&&n===null)&&(i===`bool`?typeof n!=`boolean`:i===`int`?typeof n!=`number`||!Number.isSafeInteger(n):i===`float`?typeof n!=`number`||!Number.isFinite(n):i===`str`?typeof n!=`string`:i===`array`?!Array.isArray(n):typeof n!=`object`||!n||Array.isArray(n)))throw Error(`Expected ${e.type}.`);if(e.allowed!==null&&(typeof n!=`string`||!e.allowed.includes(n)))throw Error(`Allowed values: ${e.allowed.join(`, `)}.`);return n}function Re(e,t){return typeof t==`string`&&[`str`,`str_or_null`].includes(e.type)?t:JSON.stringify(t)??``}function ze(e){let t=Me(e);if(!Array.isArray(t.tokens)||!t.tokens.every(e=>typeof e==`number`&&Number.isInteger(e)))throw Error(`Malformed tokenization response.`);return t.tokens.length}var Be=1048576,Ve=class{decoder=new TextDecoder(`utf-8`);maxFrameBytes;onMessage;onDone;buffered=``;eventName=``;dataLines=[];id=null;retry=null;frameBytes=0;ended=!1;get done(){return this.ended}constructor(e){this.maxFrameBytes=e.maxFrameBytes??Be,this.onMessage=e.onMessage,this.onDone=e.onDone}push(e){if(!this.ended){if(this.frameBytes+=e.byteLength,this.frameBytes>this.maxFrameBytes)throw Error(`SSE frame exceeded the configured byte limit.`);this.buffered+=this.decoder.decode(e,{stream:!0}),this.drainLines(!1)}}close(){if(this.ended)return;let e=this.decoder.decode();e.length>0&&(this.buffered+=e),this.drainLines(!0),this.buffered.length>0&&(this.consumeLine(this.buffered),this.buffered=``),this.dispatch()}drainLines(e){for(;this.buffered.length>0;){let t=this.buffered.indexOf(`
`),n=this.buffered.indexOf(`\r`),r;if(r=t===-1?n:n===-1?t:Math.min(t,n),r===-1||this.buffered[r]===`\r`&&this.buffered[r+1]===void 0&&!e)break;let i=this.buffered.slice(0,r),a=this.buffered[r]===`\r`&&this.buffered[r+1]===`
`?r+2:r+1;this.buffered=this.buffered.slice(a),this.consumeLine(i)}e&&this.buffered.length===0&&this.dispatch()}consumeLine(e){if(e.length===0){this.dispatch();return}if(e.startsWith(`:`))return;let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?``:e.slice(t+1),i=r.startsWith(` `)?r.slice(1):r;n===`event`?this.eventName=i:n===`data`?this.dataLines.push(i):n===`id`?this.id=i:n===`retry`&&/^\d+$/.test(i)&&(this.retry=Number(i))}dispatch(){if(this.dataLines.length===0){this.eventName=``,this.retry=null,this.frameBytes=0;return}let e=this.dataLines.join(`
`),t=this.eventName.length===0?`message`:this.eventName;if(this.eventName=``,this.dataLines=[],this.frameBytes=0,e===`[DONE]`){this.ended=!0,this.onDone?.();return}this.onMessage({event:t,data:e,id:this.id,retry:this.retry}),this.retry=null}};function He(e){let t=e??Ue();if(t===``)return``;if(!t.startsWith(`/`)||t.startsWith(`//`))throw Error(`WebUI API base must be a same-origin absolute path.`);if(t.includes(`://`)||t.includes(`?`)||t.includes(`#`)||t.includes(`\\`))throw Error(`WebUI API base must not contain an origin, query, hash, or backslash.`);let n=t.split(`/`).filter(e=>e.length>0);if(n.some(e=>e===`.`||e===`..`))throw Error(`WebUI API base must not contain dot path segments.`);return`/${n.map(encodeURIComponent).join(`/`)}`}function Ue(){if(typeof document>`u`)return``;let e=document.querySelector(`meta[name="mlxcel-ui-api-base"]`)?.content;if(e!==void 0&&e.length>0)return e;let t=document.querySelector(`base`)?.getAttribute(`href`);if(t==null||t.length===0)return``;let n=new URL(t,window.location.origin);if(n.origin!==window.location.origin||n.search.length>0||n.hash.length>0)throw Error(`WebUI document base must stay on the current origin.`);return n.pathname.replace(/\/webui\/?$/,``)}function We(e,t,n){if(!t.startsWith(`/ui-api/v1/`)&&t!==`/v1/chat/completions`&&t!==`/v1/responses`&&t!==`/settings`&&t!==`/props`&&t!==`/tokenize`)throw Error(`WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.`);let r=new URLSearchParams;for(let[e,t]of Object.entries(n??{}))t!=null&&r.set(e,String(t));return`${e}${t}${r.size===0?``:`?${r.toString()}`}`}function Ge(e){if(e.length===0)throw Error(`Opaque path segment must not be empty.`);return encodeURIComponent(e)}var Ke=`{
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
          "limits"
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
            "maxLength": 128
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
            "maxItems": 256
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
      }
    }
  }
}
`,R=class extends Error{path;constructor(e,t){super(`${e}: ${t}`),this.path=e,this.name=`ValidationError`}},qe=Xe(Ke);function Je(e,t,n=`$`){let r=ct(qe,[`components`,`schemas`],`#/components/schemas`)[e];if(r===void 0)throw new R(n,`unknown schema ${e}`);Ze(r,t,n)}function Ye(e){return JSON.parse(e)}function Xe(e){return st(Ye(e),`$contract`)}function Ze(e,t,n){let r=st(e,`${n}#schema`);if(typeof r.$ref==`string`){Ze(at(r.$ref),t,n);return}if(r.anyOf!==void 0){if(!Array.isArray(r.anyOf)||!r.anyOf.some(e=>ot(e,t,n)))throw new R(n,`did not match any allowed schema`);return}if(r.oneOf!==void 0){if(!Array.isArray(r.oneOf))throw new R(n,`schema oneOf must be an array`);let e=r.oneOf.filter(e=>ot(e,t,n)).length;if(e!==1)throw new R(n,`matched ${e} oneOf schemas`);return}if(r.const!==void 0&&t!==r.const)throw new R(n,`expected constant ${String(r.const)}`);if(Array.isArray(r.enum)&&!r.enum.some(e=>e===t))throw new R(n,`unexpected enum value ${String(t)}`);let i=r.type;if(Array.isArray(i)){if(!i.some(e=>it(e,t)))throw new R(n,`expected one of ${i.join(`, `)}`)}else if(typeof i==`string`&&!it(i,t))throw new R(n,`expected ${i}`);typeof t==`string`&&tt(r,t,n),typeof t==`number`&&rt(r,t,n),Array.isArray(t)&&$e(r,t,n),lt(t)&&Qe(r,t,n)}function Qe(e,t,n){let r=ut(e.required,`${n}#schema.required`,!1),i=e.properties===void 0?{}:st(e.properties,`${n}#schema.properties`);for(let e of r)if(!(e in t))throw new R(`${n}.${e}`,`missing required property`);if(e.maxProperties!==void 0&&Object.keys(t).length>dt(e.maxProperties,`${n}#schema.maxProperties`))throw new R(n,`too many object properties`);for(let[r,a]of Object.entries(t)){let t=i[r];if(t!==void 0)Ze(t,a,`${n}.${r}`);else if(e.additionalProperties===!1)throw new R(`${n}.${r}`,`unexpected property`);else lt(e.additionalProperties)&&Ze(e.additionalProperties,a,`${n}.${r}`)}}function $e(e,t,n){if(e.minItems!==void 0&&t.length<dt(e.minItems,`${n}#schema.minItems`))throw new R(n,`array shorter than minimum`);if(e.maxItems!==void 0&&t.length>dt(e.maxItems,`${n}#schema.maxItems`))throw new R(n,`array longer than maximum`);e.items!==void 0&&t.forEach((t,r)=>Ze(e.items,t,`${n}[${r}]`))}function et(e){let t=0;for(let n=0;n<e.length;t++)n+=(e.codePointAt(n)??0)>65535?2:1;return t}function tt(e,t,n){let r=et(t);if(e.minLength!==void 0&&r<dt(e.minLength,`${n}#schema.minLength`))throw new R(n,`string shorter than minimum`);if(e.maxLength!==void 0&&r>dt(e.maxLength,`${n}#schema.maxLength`))throw new R(n,`string longer than maximum`);if(typeof e.pattern==`string`&&!new RegExp(e.pattern,`u`).test(t))throw new R(n,`string does not match pattern`);if(e.format===`date-time`&&!nt(t))throw new R(n,`invalid RFC3339 date-time`)}function nt(e){return/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(e)?!Number.isNaN(Date.parse(e)):!1}function rt(e,t,n){if(!Number.isFinite(t))throw new R(n,`expected finite number`);if(e.type===`integer`&&!Number.isInteger(t))throw new R(n,`expected integer`);if(e.minimum!==void 0&&t<dt(e.minimum,`${n}#schema.minimum`))throw new R(n,`number below minimum`);if(e.maximum!==void 0&&t>dt(e.maximum,`${n}#schema.maximum`))throw new R(n,`number above maximum`)}function it(e,t){return e===`null`?t===null:e===`array`?Array.isArray(t):e===`object`?lt(t):e===`integer`?typeof t==`number`&&Number.isInteger(t):e===`number`?typeof t==`number`&&Number.isFinite(t):typeof t===e}function at(e){if(!e.startsWith(`#/`))throw new R(`$ref`,`unsupported external ref ${e}`);return e.slice(2).split(`/`).map(e=>e.replace(/~1/g,`/`).replace(/~0/g,`~`)).reduce((t,n)=>st(t,e)[n],qe)}function ot(e,t,n){try{return Ze(e,t,n),!0}catch(e){if(e instanceof R)return!1;throw e}}function st(e,t){if(!lt(e))throw new R(t,`expected object`);return e}function ct(e,t,n){return t.reduce((e,t)=>st(e,n)[t],e)}function lt(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function ut(e,t,n){if(e===void 0&&!n)return[];if(!Array.isArray(e)||!e.every(e=>typeof e==`string`))throw new R(t,`expected string array`);return e}function dt(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw new R(t,`expected numeric schema bound`);return e}function ft(e,t,n){return Je(e,t,n),t}function pt(e,t=`$`){return ft(`ErrorEnvelope`,e,t)}function mt(e,t=`$`){return ft(`BootstrapResponse`,e,t)}function ht(e,t=`$`){return ft(`CatalogEntry`,e,t)}function gt(e,t=`$`){return ft(`CatalogListResponse`,e,t)}function _t(e,t=`$`){return ft(`OperationsListResponse`,e,t)}function vt(e,t=`$`){return ft(`Operation`,e,t)}function yt(e,t=`$`){return ft(`OperationAccepted`,e,t)}function bt(e,t=`$`){return ft(`RuntimeSnapshot`,e,t)}function xt(e,t=`$`){return ft(`UiEvent`,e,t)}var St=class extends Error{status;envelope;constructor(e,t,n){super(n??t?.error.message??`WebUI request failed with HTTP ${e}`),this.status=e,this.envelope=t,this.name=`WebUiHttpError`}},Ct=2097152,wt=class{apiBase;fetchImpl;bearerToken=null;onUnauthorized;controllers=new Set;constructor(e={}){this.apiBase=He(e.apiBase),this.fetchImpl=e.fetchImpl??fetch.bind(globalThis),this.onUnauthorized=e.onUnauthorized}setBearerToken(e){this.bearerToken=e}abortAll(){for(let e of this.controllers)e.abort();this.controllers.clear()}async bootstrap(e){return this.request(`/ui-api/v1/bootstrap`,mt,{method:`GET`,signal:e})}async catalog(e={},t){return this.request(`/ui-api/v1/catalog`,gt,{method:`GET`,query:{...e},signal:t})}async catalogEntry(e,t){return this.request(`/ui-api/v1/catalog/${Ge(e)}`,ht,{method:`GET`,signal:t})}async refreshCatalog(e,t){return this.request(`/ui-api/v1/catalog/refresh`,yt,{method:`POST`,body:{idempotency_key:e},signal:t})}async modelAction(e,t){return this.request(`/ui-api/v1/model-actions`,yt,{method:`POST`,body:e,signal:t})}async download(e,t){return this.request(`/ui-api/v1/downloads`,yt,{method:`POST`,body:e,signal:t})}async removeModel(e,t){return this.request(`/ui-api/v1/model-removals`,yt,{method:`POST`,body:e,signal:t})}async operationsPage(e={},t){return this.request(`/ui-api/v1/operations`,_t,{method:`GET`,query:e,signal:t})}async operations(e){let t=[],n,r=null;for(;;){let i=await this.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null)r=i.server_instance_id;else if(i.server_instance_id!==r)throw Error(`Operations pagination crossed a server restart; refresh the authoritative snapshot.`);if(t.push(...i.items),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operation(e,t){return this.request(`/ui-api/v1/operations/${Ge(e)}`,vt,{method:`GET`,signal:t})}async cancelOperation(e,t){return this.request(`/ui-api/v1/operations/${Ge(e)}/cancel`,yt,{method:`POST`,body:{},signal:t})}async runtime(e,t){let n=await this.request(`/ui-api/v1/runtime`,bt,{method:`GET`,query:{model_id:e,autoload:!1},signal:t});if(n.model_id!==e)throw Error(`Runtime response model_id did not match the requested model.`);return n}async tokenCount(e,t,n){return this.request(`/tokenize`,ze,{method:`POST`,query:{model:e,autoload:!1},body:{content:t,add_special:!1,parse_special:!0,with_pieces:!1},signal:n})}async settings(e,t){return this.request(`/settings`,Pe,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async patchSettings(e,t,n){return this.request(`/settings`,Fe,{method:`PATCH`,query:{model:e,autoload:!1},body:{op:`merge`,values:t},signal:n})}async modelProps(e,t){return this.request(`/props`,Ie,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async events(e,t,n){let r=n?.serverInstanceId!==void 0&&n.serverInstanceId!==null&&n.afterSequence!==null&&n.afterSequence!==void 0,i=r?{server_instance_id:n.serverInstanceId,after_sequence:n.afterSequence}:void 0,a=new Headers({Accept:`text/event-stream`});!r&&n?.lastEventId!==void 0&&n.lastEventId!==null&&a.set(`Last-Event-ID`,n.lastEventId),await this.sse(We(this.apiBase,`/ui-api/v1/events`,i),{method:`GET`,signal:t,headers:a},{onDone:e.onDone,onFrame:t=>{t.retry!==null&&e.onRetryAfter?.(t.retry),t.data.length>0&&e.onEvent(xt(Ye(t.data)))}})}async chatCompletions(e,t,n,r){await this.sse(We(this.apiBase,`/v1/chat/completions`,{autoload:!1}),{method:`POST`,body:Dt(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async responses(e,t,n,r){await this.sse(We(this.apiBase,`/v1/responses`,{autoload:!1}),{method:`POST`,body:Dt(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async request(e,t,n){let r=await this.fetchWithAuth(We(this.apiBase,e,n.query),n);try{return r.response.ok||await this.throwHttp(r.response,r.signal),t(Ye(await this.readBody(r.response,r.signal)))}finally{this.controllers.delete(r.controller),r.cleanup()}}async sse(e,t,n){let r=await this.fetchWithAuth(e,t);try{let e=r.response;if(e.ok||await this.throwHttp(e,r.signal),e.body===null)throw Error(`WebUI event stream response has no body.`);let t=new Ve({onDone:n.onDone,onMessage:n.onFrame}),i=e.body.getReader();try{for(;;){let e=await Ot(i,r.signal);if(e.done)break;t.push(e.value)}if(t.close(),!t.done&&r.signal.aborted!==!0)throw Error(`WebUI event stream ended before a DONE frame.`)}catch(e){throw await i.cancel().catch(()=>void 0),e}finally{i.releaseLock()}}finally{this.controllers.delete(r.controller),r.cleanup()}}async fetchWithAuth(e,t){let n=new AbortController,{signal:r,cleanup:i}=kt(n,t.signal);this.controllers.add(n);let a=new Headers(t.headers);a.set(`Accept`,a.get(`Accept`)??`application/json`),this.bearerToken!==null&&a.set(`Authorization`,`Bearer ${this.bearerToken}`),t.body!==void 0&&a.set(`Content-Type`,`application/json`);try{return{response:await this.fetchImpl(e,{method:t.method,headers:a,body:t.body===void 0?void 0:JSON.stringify(t.body),signal:r,credentials:`same-origin`,cache:`no-store`,redirect:`error`}),controller:n,signal:r,cleanup:i}}catch(e){throw this.controllers.delete(n),i(),e}}async throwHttp(e,t){let n=this.bearerToken;e.status===401&&(this.bearerToken=null,this.abortAll(),this.onUnauthorized?.());let r=null;try{let n=await this.readBody(e,t);n.length>0&&(r=pt(Ye(n)))}catch{r=null}throw new St(e.status,Tt(r,n),Et(r?.error.message??`WebUI request failed with HTTP ${e.status}`,n))}async readBody(e,t){if(e.body===null)return``;let n=e.body.getReader(),r=[],i=0;try{for(;;){let e=await Ot(n,t);if(e.done)break;if(i+=e.value.byteLength,i>Ct)throw Error(`WebUI JSON response exceeded the configured byte limit.`);r.push(e.value)}}catch(e){throw await n.cancel().catch(()=>void 0),e}finally{n.releaseLock()}let a=new Uint8Array(i),o=0;for(let e of r)a.set(e,o),o+=e.byteLength;return new TextDecoder().decode(a)}};function Tt(e,t){return e===null?null:{...e,error:{...e.error,message:Et(e.error.message,t),field_errors:e.error.field_errors?.map(e=>({...e,message:Et(e.message,t)}))}}}function Et(e,t){let n=e.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`).replace(/token[=:]\s*\S+/gi,`token=[redacted]`);return t!==null&&t.length>0&&(n=n.split(t).join(`[redacted]`)),n}function Dt(e,t){if(t.length===0)throw Error(`Inference model id is required for streaming inference.`);if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Streaming inference bodies must be JSON objects.`);return{...e,model:t,stream:!0}}async function Ot(e,t){if(t.aborted)throw await e.cancel().catch(()=>void 0),At();let n=null;try{return await Promise.race([e.read(),new Promise((r,i)=>{n=()=>{e.cancel().catch(()=>void 0),i(At())},t.addEventListener(`abort`,n,{once:!0})})])}finally{n!==null&&t.removeEventListener(`abort`,n)}}function kt(e,t){if(t===void 0)return{signal:e.signal,cleanup:()=>void 0};let n=()=>e.abort(t.reason);return t.aborted?n():t.addEventListener(`abort`,n,{once:!0}),{signal:e.signal,cleanup:()=>t.removeEventListener(`abort`,n)}}function At(){return typeof DOMException==`function`?new DOMException(`The WebUI request was aborted.`,`AbortError`):Error(`The WebUI request was aborted.`)}function jt(e,t,n){let r=e.filter(e=>e.runtime.server_instance_id===t.server_instance_id&&e.runtime.model_id===t.model_id&&e.receivedAt>n-3e5&&e.receivedAt<=n),i=r.at(-1);return i!==void 0&&Math.floor(i.receivedAt/2e3)===Math.floor(n/2e3)&&r.pop(),[...r,{receivedAt:n,runtime:t}].slice(-150)}function Mt(e,t){let n=[...e.values()].sort((e,t)=>t.updated_at.localeCompare(e.updated_at)),r=n.filter(e=>!Nt(e)),i=n.filter(e=>Nt(e)&&Date.parse(e.updated_at)>t-36e5).slice(0,200);return new Map([...r,...i].map(e=>[e.operation_id,e]))}function Nt(e){return[`succeeded`,`failed`,`cancelled`].includes(e.state)}var Pt=`webui.ui-api.v1`;function Ft(){return{schemaVersion:Pt,auth:{status:`signed-out`,tokenPresent:!1},connection:`idle`,bootstrap:null,catalog:[],catalogSequence:null,operations:new Map,runtimes:new Map,runtimeHistory:[],selectedModelId:null,serverInstanceId:null,lastEventId:null,lastSequence:null,lastUpdatedAt:null,lastSuccessfulAt:null,error:null,pendingReconciliations:new Map,resourceFences:{catalog:null,operationsSnapshot:null,operations:new Map,models:new Map,runtimes:new Map}}}function It(e,t){return t.type===`login-start`?{...e,auth:{status:`authenticating`,tokenPresent:!0},connection:`bootstrapping`,error:null}:t.type===`login-success`?Lt(e,t.bootstrap,t.now):t.type===`logout`?{...Ft(),lastUpdatedAt:t.now}:t.type===`select-model`?{...e,selectedModelId:t.modelId,runtimeHistory:[]}:t.type===`catalog`?Rt(e,t.response,t.now):t.type===`operations-snapshot`?zt(e,t.response,t.now):t.type===`operation`?Bt(e,t.operation,t.sequence,t.now):t.type===`runtime`?Vt(e,t.runtime,t.sequence,t.now):t.type===`event`?Ht(e,t.event,t.now):t.type===`pending`?{...e,pendingReconciliations:Yt(e.pendingReconciliations,t.item.idempotencyKey,t.item)}:t.type===`reconciled`?{...e,pendingReconciliations:Xt(e.pendingReconciliations,t.idempotencyKey)}:{...e,connection:t.connection,error:t.error??null,lastUpdatedAt:t.now}}function Lt(e,t,n){let r=t.server.server_instance_id;return e.serverInstanceId!==null&&e.serverInstanceId!==r?{...Ft(),auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n}:{...e,auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n,error:null}}function Rt(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return z(e,t.server_instance_id,n);if(e.catalogSequence!==null&&t.snapshot_sequence<e.catalogSequence)return e;let r=e.catalogSequence===t.snapshot_sequence;return Kt({...e,connection:e.connection===`bootstrapping`?`ready`:e.connection,catalog:qt(r?e.catalog:[],t),catalogSequence:t.snapshot_sequence,serverInstanceId:t.server_instance_id,lastSequence:Jt({...e.resourceFences,catalog:t.snapshot_sequence}),error:null,resourceFences:{...e.resourceFences,catalog:t.snapshot_sequence}},n)}function zt(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return z(e,t.server_instance_id,n);if(e.resourceFences.operationsSnapshot!==null&&t.snapshot_sequence<e.resourceFences.operationsSnapshot)return e;let r=e.resourceFences.operationsSnapshot===t.snapshot_sequence,i=r?new Map(e.operations):new Map,a=r?new Map(e.resourceFences.operations):new Map;for(let e of t.items)i.set(e.operation_id,e),a.set(e.operation_id,t.snapshot_sequence);let o=Mt(i,n),s={...e.resourceFences,operationsSnapshot:t.snapshot_sequence,operations:new Map([...a].filter(([e])=>o.has(e)))};return Kt({...e,operations:o,serverInstanceId:t.server_instance_id,lastSequence:Jt(s),error:null,resourceFences:s},n)}function Bt(e,t,n,r){if(n!==null&&e.resourceFences.operationsSnapshot!==null&&n<=e.resourceFences.operationsSnapshot)return e;let i=e.resourceFences.operations.get(t.operation_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=Mt(Yt(e.operations,t.operation_id,t),r),o=n===null?e.resourceFences.operations:Yt(e.resourceFences.operations,t.operation_id,n),s={...e.resourceFences,operations:new Map([...o].filter(([e])=>a.has(e)))};return Kt({...e,operations:a,lastSequence:Jt(s),error:null,resourceFences:s},r)}function Vt(e,t,n,r){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return z(e,t.server_instance_id,r);let i=e.resourceFences.runtimes.get(t.model_id);if(n!==null&&i!==void 0&&n<i)return e;let a=n===null?e.resourceFences.runtimes:Yt(e.resourceFences.runtimes,t.model_id,n),o={...e.resourceFences,runtimes:a};return Kt({...e,runtimes:Yt(e.runtimes,t.model_id,t),runtimeHistory:t.model_id===e.selectedModelId?jt(e.runtimeHistory,t,r):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastSequence:Jt(o),error:null,resourceFences:o},r)}function Ht(e,t,n){if(t.schema_version!==`webui.ui-api.v1`)return{...e,connection:`schema-mismatch`,error:{code:`schema_mismatch`,message:`Unsupported WebUI schema ${t.schema_version}`,retryable:!1},lastUpdatedAt:n};if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return z(e,t.server_instance_id,n);if(t.type===`heartbeat`)return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:e.connection===`polling`?`polling`:`streaming`,lastUpdatedAt:n};if(t.type===`server_restart`||t.type===`gap`||t.type===`reset`)return Gt(e,t,n);if(t.type===`operation`)return{...Bt(e,t.payload.operation,t.sequence,n),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`};if(t.type===`model_revision`)return Ut(e,t,n);if(t.type===`runtime`)return Wt(e,t,n);if(t.type===`snapshot`){let r={...e.resourceFences,catalog:t.payload.catalog_changed?t.sequence:e.resourceFences.catalog};return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Jt(r),connection:`streaming`,lastUpdatedAt:n,resourceFences:r}}return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`,lastUpdatedAt:n}}function Ut(e,t,n){let r=t.payload.model_id,i=e.resourceFences.models.get(r);if(i!==void 0&&t.sequence<=i)return e;let a=e.catalog.map(e=>e.identity.id===r&&t.payload.revision>=e.identity.revision?{...e,identity:{...e.identity,revision:t.payload.revision},lifecycle:t.payload.lifecycle}:e),o={...e.resourceFences,models:Yt(e.resourceFences.models,r,t.sequence)};return Kt({...e,catalog:a,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Jt(o),connection:`streaming`,resourceFences:o},n)}function Wt(e,t,n){let r=t.payload.runtime,i=e.resourceFences.runtimes.get(r.model_id);if(i!==void 0&&t.sequence<=i)return e;let a={...e.resourceFences,runtimes:Yt(e.resourceFences.runtimes,r.model_id,t.sequence)};return Kt({...e,runtimes:Yt(e.runtimes,r.model_id,r),runtimeHistory:r.model_id===e.selectedModelId?jt(e.runtimeHistory,r,n):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Jt(a),connection:`streaming`,resourceFences:a},n)}function Gt(e,t,n){let r=t.type===`server_restart`?null:e.lastSuccessfulAt;return{...Ft(),auth:e.auth,connection:`stale`,serverInstanceId:t.server_instance_id,selectedModelId:e.selectedModelId,lastEventId:t.event_id,error:{code:t.type,message:t.payload.reason,retryable:t.payload.resnapshot},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n,lastSuccessfulAt:r}}function z(e,t,n){return{...Ft(),auth:e.auth,connection:`stale`,serverInstanceId:t,selectedModelId:e.selectedModelId,error:{code:`server_restarted`,message:`The mlxcel server restarted; refresh the authoritative snapshot before continuing.`,retryable:!0},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n}}function Kt(e,t){return{...e,lastUpdatedAt:t,lastSuccessfulAt:t}}function qt(e,t){if(e.length>0){let n=new Map(e.map(e=>[e.identity.id,e]));for(let e of t.items)n.set(e.identity.id,e);return Array.from(n.values()).sort((e,t)=>e.identity.display_name.localeCompare(t.identity.display_name))}return t.items}function Jt(e){let t=[e.catalog,e.operationsSnapshot,...e.operations.values(),...e.models.values(),...e.runtimes.values()].filter(e=>e!==null);return t.length===0?null:Math.min(...t)}function Yt(e,t,n){let r=new Map(e);return r.set(t,n),r}function Xt(e,t){let n=new Map(e);return n.delete(t),n}var Zt=2e3,Qt=1e4,$t=3e4,en=6e4,tn=class{client;dispatch;getSnapshot;clock;visibility;random;timer=null;observationTimer=null;inflight=null;eventAbort=null;stopped=!0;failures=0;generation=0;unsubscribe;constructor(e){this.client=e.client,this.dispatch=e.dispatch,this.getSnapshot=e.getSnapshot,this.clock=e.clock??cn,this.visibility=e.visibility??ln,this.random=e.random??Math.random,this.unsubscribe=this.visibility.subscribe(()=>this.visibilityChanged())}start(){this.stopped&&(this.stopped=!1,this.generation+=1,this.reschedule(0))}stop(){this.generation+=1,this.stopped=!0,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null}selectionChanged(){this.cancelObservation(),this.reschedule(0)}cancelObservation(){this.generation+=1,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null}visibilityChanged(){this.cancelObservation(),!this.stopped&&(this.dispatch({type:`connection`,connection:`stale`,now:this.clock.now()}),this.reschedule(0))}dispose(){this.stop(),this.unsubscribe()}async refresh(){if(this.inflight!==null)return;let e=new AbortController;this.inflight=e;let t=this.generation,n=this.clock.now(),r=this.clock.setTimeout(()=>{e.abort(),this.generation===t&&this.dispatch({type:`connection`,connection:`stale`,error:{code:`observation_timeout`,message:`Runtime observation timed out; retrying.`,retryable:!0},now:this.clock.now()})},Qt);this.observationTimer=r;try{if(this.getSnapshot().auth.status===`signed-out`)return;let r=await this.client.bootstrap(e.signal);if(!this.isCurrent(e,t)||this.getSnapshot().auth.status===`signed-out`)return;this.dispatch({type:`login-success`,bootstrap:r,now:n});let i=await this.catalogSnapshot(e.signal);if(!this.isCurrent(e,t))return;for(let e of i)this.dispatch({type:`catalog`,response:e,now:n});let a=await this.operationSnapshot(e.signal);if(!this.isCurrent(e,t))return;let o=i.at(-1),s=a[0];if(o!==void 0&&s!==void 0&&s.server_instance_id!==o.server_instance_id){this.dispatch({type:`connection`,connection:`stale`,error:{code:`snapshot_mismatch`,message:`Operation and catalog snapshots came from different server instances.`,retryable:!0},now:n});return}let c=a.flatMap(e=>(this.dispatch({type:`operations-snapshot`,response:e,now:n}),e.items)),l=this.getSnapshot().selectedModelId,u=l!==null&&i.every(e=>e.server_instance_id===r.server.server_instance_id)&&!i.some(e=>e.items.some(e=>e.identity.id===l));if(u&&this.dispatch({type:`select-model`,modelId:null}),u||await this.refreshSelectedRuntime(e.signal,t),!this.isCurrent(e,t))return;let d=this.reconcilePending(c,n);this.failures=0,d||this.dispatch({type:`connection`,connection:`ready`,now:n}),this.eventAbort===null&&!this.visibility.hidden()&&this.startEvents()}catch(n){!e.signal.aborted&&this.generation===t&&(this.failures+=1,this.dispatch({type:`connection`,connection:rn(n),error:an(n),now:this.clock.now()}))}finally{this.clock.clearTimeout(r),this.observationTimer===r&&(this.observationTimer=null),this.inflight===e&&(this.inflight=null),!this.stopped&&this.generation===t&&this.reschedule(this.pollDelay())}}noteUnknownPost(e){this.dispatch({type:`pending`,item:e}),this.reschedule(0)}startEvents(){this.eventAbort?.abort();let e=new AbortController;this.eventAbort=e;let t=this.generation,n=nn(this.getSnapshot());this.client.events({onEvent:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.handleEvent(n)},onRetryAfter:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.reschedule(n)}},e.signal,n).then(()=>{this.generation===t&&(this.eventAbort===e&&(this.eventAbort=null),!e.signal.aborted&&!this.stopped&&this.reschedule(this.backoffDelay()))}).catch(n=>{e.signal.aborted||this.stopped||this.generation!==t||(this.failures+=1,this.eventAbort=null,this.dispatch({type:`connection`,connection:rn(n),error:an(n),now:this.clock.now()}),this.reschedule(this.backoffDelay()))})}handleEvent(e){this.dispatch({type:`event`,event:e,now:this.clock.now()}),(e.type===`server_restart`||e.type===`gap`||e.type===`reset`)&&e.payload.resnapshot&&(this.eventAbort?.abort(),this.eventAbort=null,this.refresh())}async catalogSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.catalog(n===void 0?{}:{cursor:n},e);if(r===null?r=i:sn(`catalog`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operationSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null?r=i:sn(`operations`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async refreshSelectedRuntime(e,t){let n=this.getSnapshot().selectedModelId;if(n===null)return;let r;try{r=await this.client.runtime(n,e)}catch(e){throw typeof e==`object`&&e&&`status`in e&&e.status===404?new on(`Selected model changed during observation; refreshing the catalog.`):e}e.aborted||this.generation!==t||this.getSnapshot().selectedModelId===n&&this.dispatch({type:`runtime`,runtime:r,sequence:r.snapshot_sequence,now:this.clock.now()})}reconcilePending(e,t){let n=!1,r=new Set(e.map(e=>e.operation_id));for(let e of this.getSnapshot().pendingReconciliations.values())e.operationId!==null&&r.has(e.operationId)?this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}):t-e.createdAt>=en&&(this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}),this.dispatch({type:`connection`,connection:`stale`,error:{code:`unknown_post_unresolved`,message:`A previous control request could not be matched to an operation after reconciliation.`,retryable:!0},now:t}),n=!0);return n}reschedule(e){this.stopped||this.visibility.hidden()||(this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=this.clock.setTimeout(()=>{this.timer=null,this.refresh()},e))}pollDelay(){return Zt}backoffDelay(){let e=Math.min($t,500*2**Math.min(6,this.failures));return Math.round(e/2+this.random()*(e/2))}isCurrent(e,t){return this.inflight===e&&this.generation===t&&!e.signal.aborted}};function nn(e){let t=Jt(e.resourceFences);if(e.serverInstanceId!==null&&t!==null)return{lastEventId:null,serverInstanceId:e.serverInstanceId,afterSequence:t};if(e.lastEventId!==null)return{lastEventId:e.lastEventId,serverInstanceId:null,afterSequence:null}}function rn(e){if(e instanceof on)return`stale`;let t=typeof e==`object`&&e&&`status`in e&&typeof e.status==`number`?e.status:null;return t===401||e instanceof Error&&/401|unauthorized/i.test(e.message)?`unauthorized`:t===403||e instanceof Error&&/403|forbidden/i.test(e.message)?`forbidden`:e instanceof DOMException&&e.name===`AbortError`?`stale`:`offline`}function an(e){return e instanceof on?{code:e.code,message:e.message,retryable:!0}:{code:`sync_error`,message:e instanceof Error?e.message.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`):`Unknown WebUI client error`,retryable:!0}}var on=class extends Error{code=`snapshot_mismatch`;constructor(e){super(e),this.name=`SnapshotConsistencyError`}};function sn(e,t,n){if(n.server_instance_id!==t.server_instance_id)throw new on(`${e} pagination crossed a server restart; refresh the authoritative snapshot.`);if(n.snapshot_sequence!==t.snapshot_sequence)throw new on(`${e} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`)}var cn={setTimeout:(e,t)=>globalThis.setTimeout(e,t),clearTimeout:e=>globalThis.clearTimeout(e),now:()=>Date.now()},ln={hidden:()=>typeof document<`u`&&document.hidden,subscribe:e=>typeof document>`u`?()=>void 0:(document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e))},un=_.createContext(null),dn=_.createContext(null);function fn({children:e,apiBase:t,fetchImpl:n}){let[r,i]=_.useReducer(It,void 0,Ft),a=_.useRef(r);a.current=r;let o=_.useRef(0),s=_.useRef(null),c=_.useMemo(()=>new wt({apiBase:t,fetchImpl:n,onUnauthorized:()=>{o.current+=1,s.current?.stop(),i({type:`logout`,now:Date.now()})}}),[t,n]);_.useEffect(()=>{let e=new tn({client:c,dispatch:i,getSnapshot:()=>a.current});return s.current=e,()=>{e.dispose(),s.current=null,c.abortAll()}},[c]);let l=_.useMemo(()=>({getTokenCount:(e,t,n)=>c.tokenCount(d(e),t,n),getSettings:(e,t)=>c.settings(d(e),t),patchSettings:(e,t,n)=>c.patchSettings(d(e),t,n),getModelProps:(e,t)=>c.modelProps(d(e),t),login:async e=>{o.current+=1;let t=o.current;c.abortAll(),c.setBearerToken(e),i({type:`login-start`});try{let e=await c.bootstrap();if(o.current!==t)return;i({type:`login-success`,bootstrap:e,now:Date.now()}),s.current?.start()}catch(e){throw o.current===t&&(c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})),e}},logout:()=>{o.current+=1,s.current?.stop(),c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})},refresh:async()=>{await s.current?.refresh()},selectModel:e=>{i({type:`select-model`,modelId:e}),s.current?.selectionChanged()},loadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},unloadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},downloadModel:async e=>{await u(`download`,e.idempotency_key,void 0,()=>c.download(e))},removeModel:async e=>{await u(`removal`,e.idempotency_key,e.model_id,()=>c.removeModel(e))},refreshRuntime:async e=>{let t=o.current,n=await c.runtime(e);return o.current===t&&i({type:`runtime`,runtime:n,sequence:n.snapshot_sequence,now:Date.now()}),n},refreshCatalog:async e=>{await u(`catalog-refresh`,e,void 0,()=>c.refreshCatalog(e))},cancelOperation:async e=>{let t=o.current;await c.cancelOperation(e),o.current===t&&await s.current?.refresh()},streamChatCompletions:async(e,t,n,r)=>{await c.chatCompletions(d(e),t,n,r)},streamResponses:async(e,t,n,r)=>{await c.responses(d(e),t,n,r)}}),[c]);async function u(e,t,n,r){let i=o.current;try{let a=await r();if(o.current!==i)return;s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:a.operation_id,modelId:n,createdAt:Date.now()})}catch(r){throw o.current===i&&!(r instanceof St)&&s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:null,modelId:n,createdAt:Date.now()}),r}}function d(e){let t=a.current.catalog.find(t=>t.identity.id===e);if(t===void 0)throw Error(`Selected model is not present in the catalog snapshot.`);if(t.identity.inference_id.length===0)throw Error(`Selected model does not expose an inference model id.`);return t.identity.inference_id}return(0,x.jsx)(dn.Provider,{value:l,children:(0,x.jsx)(un.Provider,{value:r,children:e})})}function pn(){let e=_.useContext(un);if(e===null)throw Error(`useWebUi must be used within WebUiProvider.`);return e}function mn(){let e=_.useContext(dn);if(e===null)throw Error(`useWebUiActions must be used within WebUiProvider.`);return e}var hn=Object.freeze({}),gn=[`max_tokens`,`temperature`,`top_p`,`top_k`,`min_p`,`repetition_penalty`,`seed`];function _n(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Request defaults must be an object.`);let t={};for(let[n,r]of Object.entries(e)){if(!gn.includes(n))throw Error(`Unsupported request default: ${n}`);if(r!==void 0){if(typeof r!=`number`||!Number.isFinite(r))throw Error(`${n} must be a finite number; leave blank to inherit.`);if([`max_tokens`,`top_k`,`seed`].includes(n)&&!Number.isSafeInteger(r))throw Error(`${n} must be a safe integer.`);if(n===`max_tokens`&&r<1||n===`temperature`&&r<0||n===`top_k`&&r<0||[`top_p`,`min_p`].includes(n)&&(r<0||r>1)||n===`repetition_penalty`&&r<=0||n===`seed`&&r<0)throw Error(`${n} is outside its supported range.`);t[n]=r}}return Object.freeze(t)}var vn=hn,yn=new Set;function bn(e){return yn.add(e),()=>yn.delete(e)}function xn(e){vn=_n(e);for(let e of yn)e()}function Sn(){xn(hn)}function Cn(){return{defaults:(0,_.useSyncExternalStore)(bn,()=>vn,()=>hn),setDefaults:xn,reset:Sn}}function wn(e,t,n){return e===null||e<=0||t===null||n===void 0?`unknown`:t+n>e?`exceeds`:`raw-fits`}function Tn({modelId:e,nCtx:t,locale:n}){let r=mn(),{defaults:i}=Cn(),[a,o]=(0,_.useState)(``),[s,c]=(0,_.useState)(null),[l,u]=(0,_.useState)(!1),[d,f]=(0,_.useState)(!1),p=(0,_.useRef)(null);(0,_.useEffect)(()=>()=>p.current?.abort(),[]);let m=(e,t)=>n===`ko`?t:e,h=async()=>{p.current?.abort();let t=new AbortController;p.current=t,u(!0),f(!1),c(null);try{let n=await r.getTokenCount(e,a,t.signal);t.signal.aborted||c(n)}catch{t.signal.aborted||f(!0)}finally{t.signal.aborted||u(!1)}},g=wn(t,s,i.max_tokens);return(0,x.jsxs)(`div`,{children:[(0,x.jsx)(I,{label:m(`Optional raw prompt budget check`,`선택적 원시 프롬프트 예산 확인`),value:a,onChange:e=>{p.current?.abort(),u(!1),o(e),c(null)},hint:m(`Sent only when Check is pressed; never persisted. Raw tokens exclude chat templates, conversation history and media. Server validation is final.`,`확인을 누를 때만 전송하며 저장하지 않습니다. 원시 토큰은 채팅 템플릿, 대화 기록, 미디어를 제외합니다. 최종 검증은 서버가 수행합니다.`)}),(0,x.jsx)(k,{disabled:l||a.length===0,onClick:()=>void h(),children:m(`Check with model tokenizer`,`모델 토크나이저로 확인`)}),(0,x.jsx)(`p`,{role:`status`,children:d?m(`Tokenization unavailable; budget unknown.`,`토큰화 불가; 예산을 알 수 없습니다.`):s===null?m(`Not measured`,`측정하지 않음`):`${s} ${m(`raw tokens`,`원시 토큰`)}. ${g===`exceeds`?m(`Raw input + requested output already exceeds observed context.`,`원시 입력 + 요청 출력이 관측 컨텍스트를 초과합니다.`):g===`unknown`?m(`Context/output is unknown; no fit claim.`,`컨텍스트/출력을 알 수 없어 수용 가능 여부를 판단하지 않습니다.`):m(`Raw input + output fits, but templated chat may still exceed the context.`,`원시 입력 + 출력은 들어가지만 템플릿 적용 채팅은 컨텍스트를 초과할 수 있습니다.`)}`})]})}function En({locale:e}){let{defaults:t,setDefaults:n,reset:r}=Cn(),[i,a]=(0,_.useState)(()=>Object.fromEntries(Object.entries(t).map(([e,t])=>[e,String(t)]))),[o,s]=(0,_.useState)(``),[c,l]=(0,_.useState)(!1),u=(t,n)=>e===`ko`?n:t;return(0,x.jsxs)(`section`,{className:`screen-stack`,children:[(0,x.jsx)(`h2`,{children:u(`Generation · next request`,`생성 · 다음 요청`)}),(0,x.jsx)(`p`,{children:u(`These defaults stay in browser memory only. Blank fields are omitted and inherit server defaults. Existing turns are frozen; changing defaults never alters an in-flight request. System prompts belong to individual conversations in Chat and are not stored here.`,`기본값은 브라우저 메모리에만 남습니다. 빈 필드는 생략되어 서버 기본값을 사용합니다. 진행 중인 요청은 변경되지 않습니다. 시스템 프롬프트는 채팅의 개별 대화에서 설정하며 여기에 저장하지 않습니다.`)}),(0,x.jsx)(`div`,{className:`settings-grid`,children:gn.map(e=>(0,x.jsx)(I,{label:e,value:i[e]??``,onChange:t=>a(n=>({...n,[e]:t})),hint:`${u(`Current next-request default`,`현재 다음 요청 기본값`)}: ${t[e]??u(`inherit`,`상속`)}`},e))}),o?(0,x.jsx)(L,{title:u(`Invalid request defaults`,`잘못된 요청 기본값`),body:o}):null,(0,x.jsxs)(`div`,{children:[(0,x.jsx)(k,{onClick:()=>{try{n(_n(Object.fromEntries(Object.entries(i).filter(([,e])=>e.trim()!==``).map(([e,t])=>[e,Number(t)])))),s(``)}catch(e){s(e instanceof Error?e.message:`Invalid defaults`)}},children:u(`Save session defaults`,`세션 기본값 저장`)}),` `,(0,x.jsx)(k,{onClick:()=>l(!0),children:u(`Reset request defaults…`,`요청 기본값 초기화…`)})]}),(0,x.jsx)(I,{label:u(`Reasoning / structured output`,`추론 / 구조화 출력`),value:u(`Endpoint-specific controls are available only where explicitly supported in Chat.`,`채팅에서 명시적으로 지원되는 엔드포인트에만 제공됩니다.`),disabled:!0,hint:u(`No unsupported parameters are sent.`,`지원되지 않는 매개변수는 전송하지 않습니다.`)}),(0,x.jsxs)(we,{open:c,title:u(`Reset request defaults?`,`요청 기본값을 초기화할까요?`),onClose:()=>l(!1),children:[(0,x.jsx)(`p`,{children:u(`Only browser-session request defaults will be cleared. Server and next-load profiles remain unchanged.`,`브라우저 세션 요청 기본값만 지워집니다. 서버와 다음 로드 프로필은 변경되지 않습니다.`)}),(0,x.jsx)(k,{onClick:()=>{r(),a({}),s(``),l(!1)},children:u(`Reset request defaults`,`요청 기본값 초기화`)})]})]})}function Dn({modelId:e,locale:t}){let n=mn(),[r,i]=(0,_.useState)(null),[a,o]=(0,_.useState)({}),[s,c]=(0,_.useState)({}),[l,u]=(0,_.useState)(``),[d,f]=(0,_.useState)(!1),[p,m]=(0,_.useState)(!1),h=(0,_.useRef)(null),g=(e,n)=>t===`ko`?n:e;(0,_.useEffect)(()=>{let t=new AbortController;return h.current=t,f(!0),n.getSettings(e,t.signal).then(e=>{t.signal.aborted||i(e)}).catch(()=>{t.signal.aborted||u(`Settings unavailable. The server may have disabled --settings.`)}).finally(()=>{t.signal.aborted||f(!1)}),()=>t.abort()},[n,e]);let v=async()=>{let t=h.current;if(!(t===null||t.signal.aborted)){f(!0);try{let r=await n.getSettings(e,t.signal);t.signal.aborted||(i(r),u(g(`Current values refreshed. Review the retained draft before applying.`,`현재 값을 새로 고쳤습니다. 남아 있는 초안을 검토한 후 적용하세요.`)))}catch{t.signal.aborted||u(g(`Refresh failed. No changes submitted.`,`새로 고침에 실패했습니다. 변경 사항은 전송되지 않았습니다.`))}finally{t.signal.aborted||f(!1)}}},y=async()=>{let t=h.current;if(t===null||t.signal.aborted||r===null)return;let s={},l={};for(let e of r.schema)if(Object.hasOwn(a,e.name))try{s[e.name]=Le(e,a[e.name])}catch(t){l[e.name]=t instanceof Error?t.message:`Invalid value`}if(c(l),!(Object.keys(l).length>0)){f(!0);try{let a=await n.getSettings(e,t.signal);if(t.signal.aborted)return;if(a.fingerprint!==r.fingerprint){i(a),u(g(`Another client changed these settings. Current values refreshed; review and Apply again to reconfirm.`,`다른 클라이언트가 설정을 변경했습니다. 갱신된 현재 값을 검토하고 다시 적용하여 확인하세요.`));return}let l=await n.patchSettings(e,s,t.signal);if(t.signal.aborted)return;c(Object.fromEntries(l.rejected.map(e=>[e.name,e.reason]))),o(e=>Object.fromEntries(Object.entries(e).filter(([e])=>!Object.hasOwn(l.applied,e)))),u(l.rejected.length>0?g(`${Object.keys(l.applied).length} applied; ${l.rejected.length} rejected. Rejected drafts retained.`,`${Object.keys(l.applied).length}개 적용, ${l.rejected.length}개 거부됨. 거부된 초안은 유지됩니다.`):g(`Accepted fields applied; reading effective values.`,`승인된 필드를 적용했습니다. 실제 값을 다시 읽습니다.`));let d=await n.getSettings(e,t.signal);t.signal.aborted||i(d)}catch{t.signal.aborted||u(g(`The outcome may be unknown. Refresh effective values before retrying; no automatic PATCH retry.`,`결과를 확정할 수 없습니다. 재시도 전에 실제 값을 새로 고치세요. PATCH는 자동 재시도하지 않습니다.`))}finally{t.signal.aborted||f(!1)}}};return(0,x.jsxs)(`section`,{className:`screen-stack`,children:[(0,x.jsx)(`h2`,{children:g(`Loaded model · live server values`,`로드된 모델 · 실시간 서버 값`)}),(0,x.jsx)(`p`,{children:g(`Changes affect newly admitted requests. Fingerprints detect already-observed external changes, not atomic compare-and-swap. Numeric bounds remain enforced by the server; this schema publishes types and allowed enums, not numeric min/max.`,`변경은 새로 수락되는 요청에 적용됩니다. 지문은 관측된 외부 변경을 감지하지만 원자적 비교 교환은 아닙니다. 숫자 범위는 서버가 검증합니다. 스키마에는 타입과 허용 열거값만 있습니다.`)}),l?(0,x.jsx)(L,{tone:`warning`,title:g(`Settings result`,`설정 결과`),body:l}):null,(0,x.jsxs)(`div`,{children:[(0,x.jsx)(k,{onClick:()=>void v(),disabled:d,children:g(`Refresh current values`,`현재 값 새로 고침`)}),` `,(0,x.jsx)(k,{onClick:()=>void y(),disabled:d||Object.keys(a).length===0,children:g(`Apply live draft`,`실시간 초안 적용`)}),` `,(0,x.jsx)(k,{onClick:()=>m(!0),disabled:d||r===null,children:g(`Reset live draft…`,`실시간 초안 초기화…`)})]}),r?.schema.filter(e=>e.mutable).map(e=>(0,x.jsx)(I,{label:e.name,value:a[e.name]??Re(e,r.current[e.name]),onChange:t=>o(n=>({...n,[e.name]:t})),disabled:d,error:s[e.name],hint:`${e.help} Current: ${Re(e,r.current[e.name])}. Type: ${e.type}${e.allowed===null?``:`; allowed: ${e.allowed.join(`, `)}`}`},e.name)),(0,x.jsxs)(`details`,{children:[(0,x.jsx)(`summary`,{children:g(`Server startup · restart required (read-only)`,`서버 시작 · 재시작 필요 (읽기 전용)`)}),r?.schema.filter(e=>!e.mutable).map(e=>(0,x.jsx)(I,{label:e.name,value:Re(e,r.current[e.name]),disabled:!0,hint:e.reason??e.help},e.name))]}),(0,x.jsxs)(we,{open:p,title:g(`Reset live draft only?`,`실시간 초안만 초기화할까요?`),onClose:()=>m(!1),children:[(0,x.jsx)(`p`,{children:g(`Stage server startup defaults for mutable fields. No server change occurs until Apply.`,`변경 가능한 필드에 서버 시작 기본값을 초안으로 설정합니다. 적용 전에는 서버가 변경되지 않습니다.`)}),(0,x.jsx)(k,{onClick:()=>{r!==null&&o(Object.fromEntries(r.schema.filter(e=>e.mutable).map(e=>[e.name,Re(e,e.default)]))),c({}),m(!1)},children:g(`Reset draft`,`초안 초기화`)})]})]})}var On=[`fp16`,`float16`,`int8`,`i8`,`turbo4-asym`,`fp16+turbo4`,`turbo3-asym`,`fp16+turbo3`,`turbo3`,`turbo4`,`turbo4-sym`,`turbo4-delegated`,`fp16+turbo4-delegated`],kn=Object.freeze({});function An(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Profile must be an object.`);let t={};for(let[n,r]of Object.entries(e)){if(![`ctx_size`,`n_parallel`,`kv_cache_mode`].includes(n))throw Error(`Unsafe or unsupported profile field: ${n}`);if(r!=null){if(n===`kv_cache_mode`){if(typeof r!=`string`||!On.includes(r))throw Error(`Unknown KV cache mode.`);t.kv_cache_mode=r}else{if(typeof r!=`number`||!Number.isInteger(r)||r<1||r>(n===`ctx_size`?262144:32))throw Error(`${n} is outside its supported range.`);n===`ctx_size`?t.ctx_size=r:t.n_parallel=r}}}return Object.freeze(t)}function jn(e){if(e.length>65536)throw Error(`Profile import exceeds 64 KiB.`);let t=JSON.parse(e);if(typeof t!=`object`||!t||Array.isArray(t))throw Error(`Invalid profile document.`);let n=t;if(n.version===0&&Object.keys(n).every(e=>[`version`,`profile`].includes(e)))return{version:1,reusable:An(n.profile),models:{}};if(n.version!==1||Object.keys(n).some(e=>![`version`,`reusable`,`models`].includes(e)))throw Error(`Unsupported profile version or fields.`);if(typeof n.models!=`object`||n.models===null||Array.isArray(n.models))throw Error(`Invalid model profiles.`);let r=Object.create(null);if(Object.keys(n.models).length>200)throw Error(`Too many model profiles.`);for(let[e,t]of Object.entries(n.models)){if(!/^mdl_[A-Za-z0-9_-]{43}$/.test(e))throw Error(`Model profiles require opaque catalog IDs.`);r[e]=An(t)}return{version:1,reusable:An(n.reusable),models:r}}var Mn=`mlxcel.webui.load-profiles.v1`,Nn={version:1,reusable:kn,models:{}},Pn=!1,Fn=new Set;function In(e){return Fn.add(e),()=>Fn.delete(e)}function Ln(){if(!Pn){Pn=!0;try{let e=localStorage.getItem(Mn);e!==null&&(Nn=jn(e))}catch{}}}function Rn(e){let t=JSON.stringify(e),n=jn(t);localStorage.setItem(Mn,t),Nn=n;for(let e of Fn)e()}function zn(e){Ln();let t=(0,_.useSyncExternalStore)(In,()=>Nn);return{reusable:t.reusable,modelProfile:e===null?kn:t.models[e]??kn,profile:e!==null&&Object.hasOwn(t.models,e)?t.models[e]:t.reusable,save:(t,n)=>{let r=An(t);if(n===`model`&&e===null)throw Error(`Select a model first.`);Rn(n===`reusable`?{...Nn,reusable:r}:{...Nn,models:{...Nn.models,[e]:r}})},reset:(t=e===null?`reusable`:`model`)=>{let n=Object.fromEntries(Object.entries(Nn.models).filter(([t])=>t!==e));Rn(t===`reusable`?{...Nn,reusable:kn}:{...Nn,models:n})},exportJson:()=>JSON.stringify(Nn,null,2),importJson:e=>Rn(jn(e))}}function Bn(e){let t=An(e);return[`mlxcel-server --webui`,t.ctx_size===void 0?``:`--ctx-size ${t.ctx_size}`,t.n_parallel===void 0?``:`--parallel ${t.n_parallel}`,t.kv_cache_mode===void 0?``:`--kv-cache-mode ${t.kv_cache_mode}`].filter(Boolean).join(` `)}function Vn({model:e,locale:t,single:n}){let r=zn(e?.identity.id??null),[i,a]=(0,_.useState)(r.profile),[o,s]=(0,_.useState)(e===null?`reusable`:`model`),[c,l]=(0,_.useState)(``),[u,d]=(0,_.useState)(``),[f,p]=(0,_.useState)(!1),m=(e,n)=>t===`ko`?n:e,h=o===`reusable`?r.reusable:r.modelProfile;(0,_.useEffect)(()=>{a(h)},[h]);let g=e=>{try{e(),d(m(`Browser profile saved. Nothing loaded or changed on the server.`,`브라우저 프로필을 저장했습니다. 서버에서 로드하거나 변경한 내용은 없습니다.`))}catch(e){d(e instanceof Error?e.message:`Profile operation failed`)}};return(0,x.jsxs)(`section`,{className:`screen-stack`,children:[(0,x.jsx)(`h2`,{children:m(`Next load · browser profile`,`다음 로드 · 브라우저 프로필`)}),(0,x.jsx)(`p`,{children:m(`Explicit CLI settings take precedence over this profile; the profile overrides a preset only where CLI has not pinned a value. Effective resolved values must be read after loading. Saving is not loading. A loaded model requires an explicit unload and load from Models; failed loads retain this pending profile.`,`명시적 CLI 설정이 프로필보다 우선합니다. CLI로 고정되지 않은 값에만 프로필이 프리셋보다 우선합니다. 실제 값은 로드 후 확인하세요. 저장은 로드가 아닙니다. 로드된 모델은 모델 화면에서 명시적으로 언로드 후 로드해야 하며, 실패 시 프로필은 대기 상태로 남습니다.`)}),n?(0,x.jsx)(L,{tone:`info`,title:m(`Single-model mode: restart required`,`단일 모델 모드: 재시작 필요`),body:m(`These preferences cannot be applied to this running worker. Restart with validated CLI flags, or start model-free mode to use load operations.`,`실행 중인 워커에 적용할 수 없습니다. 검증된 CLI 플래그로 재시작하거나 모델 없는 모드에서 로드 작업을 사용하세요.`)}):null,(0,x.jsx)(F,{locale:t,label:m(`Profile scope`,`프로필 범위`),value:o,onChange:e=>s(e),options:[{value:`reusable`,label:m(`Reusable defaults`,`재사용 기본값`)},...e===null?[]:[{value:`model`,label:e.identity.display_name}]]}),(0,x.jsxs)(`div`,{className:`settings-grid`,children:[(0,x.jsx)(I,{label:`ctx_size (1–262144)`,value:i.ctx_size?.toString()??``,onChange:e=>a({...i,ctx_size:e.trim()===``?void 0:Number(e)}),hint:m(`Blank: inherit. Actual per-slot context is resolved by the server, not estimated here.`,`빈 값: 상속. 슬롯별 실제 컨텍스트는 서버가 결정하며 여기서 추정하지 않습니다.`)}),(0,x.jsx)(I,{label:`n_parallel (1–32)`,value:i.n_parallel?.toString()??``,onChange:e=>a({...i,n_parallel:e.trim()===``?void 0:Number(e)}),hint:m(`Scheduler/worker geometry changes only on the next explicit load.`,`스케줄러/워커 구성은 다음 명시적 로드에서만 변경됩니다.`)}),(0,x.jsx)(F,{locale:t,label:`kv_cache_mode`,value:i.kv_cache_mode??``,onChange:e=>a({...i,kv_cache_mode:e===``?void 0:e}),options:[{value:``,label:m(`Inherit`,`상속`)},...On.map(e=>({value:e,label:e}))]})]}),(0,x.jsx)(`p`,{children:m(`Backend/model restrictions are validated by the existing loader; accepting a browser profile does not promise a model supports every mode. Draft models, adapters, distributed topology and other worker geometry are read-only in v1.`,`백엔드/모델 제약은 기존 로더가 검증합니다. 브라우저 프로필 저장이 모든 모드의 지원을 보장하지 않습니다. 초안 모델, 어댑터, 분산 구성 및 기타 워커 설정은 v1에서 읽기 전용입니다.`)}),u?(0,x.jsx)(L,{tone:`info`,title:m(`Profile result`,`프로필 결과`),body:u}):null,(0,x.jsxs)(`div`,{children:[(0,x.jsx)(k,{onClick:()=>g(()=>r.save(An(i),o)),children:m(`Save pending profile`,`대기 프로필 저장`)}),` `,(0,x.jsx)(k,{onClick:()=>{a(h),d(``)},children:m(`Discard draft`,`초안 취소`)}),` `,(0,x.jsx)(k,{onClick:()=>p(!0),children:m(`Reset selected profile…`,`선택 프로필 초기화…`)})]}),(0,x.jsxs)(`p`,{children:[m(`Saved pending profile (not active)`,`저장된 대기 프로필 (미적용)`),`: `,(0,x.jsx)(`code`,{children:JSON.stringify(h)})]}),(0,x.jsx)(`p`,{children:m(`Sanitized CLI flags · display only; model source omitted`,`안전한 CLI 플래그 · 표시 전용; 모델 소스 생략`)}),(0,x.jsx)(`code`,{children:Bn(h)}),(0,x.jsxs)(`details`,{children:[(0,x.jsx)(`summary`,{children:m(`Import / export browser profiles`,`브라우저 프로필 가져오기 / 내보내기`)}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:m(`Versioned profile JSON`,`버전이 있는 프로필 JSON`)}),(0,x.jsx)(`textarea`,{value:c,onChange:e=>l(e.currentTarget.value),maxLength:65536,rows:6})]}),(0,x.jsx)(k,{onClick:()=>l(r.exportJson()),children:m(`Export to text`,`텍스트로 내보내기`)}),` `,(0,x.jsx)(k,{onClick:()=>g(()=>r.importJson(c)),children:m(`Validate and import`,`검증 후 가져오기`)})]}),(0,x.jsxs)(we,{open:f,title:m(`Reset this browser profile?`,`이 브라우저 프로필을 초기화할까요?`),onClose:()=>p(!1),children:[(0,x.jsx)(`p`,{children:m(`Resets only the selected scope. Model scope falls back to reusable defaults; reusable scope leaves model-specific profiles intact. Does not change a running worker.`,`선택한 범위만 초기화합니다. 모델 범위는 재사용 기본값으로 돌아가며 재사용 범위는 모델별 프로필을 유지합니다. 실행 중인 워커는 변경하지 않습니다.`)}),(0,x.jsx)(k,{onClick:()=>{g(()=>r.reset(o)),p(!1)},children:m(`Reset profile`,`프로필 초기화`)})]})]})}function Hn({locale:e}){let t=pn(),n=mn(),r=t.catalog.find(e=>e.identity.id===t.selectedModelId)??null,[i,a]=(0,_.useState)(null),o=(t,n)=>e===`ko`?n:t,s=t.auth.status===`authenticated`&&[`ready`,`streaming`,`polling`].includes(t.connection),c=s&&r?.lifecycle.state===`ready`?r.identity.id:null;return(0,_.useEffect)(()=>{if(a(null),c===null)return;let e=new AbortController;return n.getModelProps(c,e.signal).then(t=>{e.signal.aborted||a(t)}).catch(()=>void 0),()=>e.abort()},[n,c,r?.identity.revision]),(0,x.jsxs)(`div`,{className:`screen-stack settings-sections`,children:[(0,x.jsxs)(`section`,{className:`screen-stack`,children:[(0,x.jsx)(`h2`,{children:o(`Privacy · browser only`,`개인정보 · 브라우저 전용`)}),(0,x.jsx)(`p`,{children:o(`Appearance and explicitly saved load profiles use this browser’s preferences. Request defaults stay in memory; API credentials and system prompts are never stored by Settings. Conversation history is controlled separately in Chat.`,`모양과 명시적으로 저장한 로드 프로필은 브라우저 환경설정을 사용합니다. 요청 기본값은 메모리에만 유지하며 API 자격 증명과 시스템 프롬프트는 설정에서 저장하지 않습니다. 대화 기록은 채팅에서 별도로 관리합니다.`)})]}),(0,x.jsx)(En,{locale:e}),s?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(F,{locale:e,label:o(`Selected model for settings`,`설정 대상 모델`),value:t.selectedModelId??``,options:[{value:``,label:o(`No model selected`,`선택한 모델 없음`)},...t.catalog.map(e=>({value:e.identity.id,label:`${e.identity.display_name} · ${e.lifecycle.state}`}))],onChange:e=>n.selectModel(e===``?null:e),hint:o(`Selection changes browser context only. It never loads, unloads or reroutes an existing request.`,`선택은 브라우저 문맥만 변경하며 모델을 로드하거나 언로드하거나 기존 요청의 대상을 변경하지 않습니다.`),testId:`settings-model-selector`}),(0,x.jsx)(Vn,{model:r,locale:e,single:t.bootstrap?.server.mode===`single_model`},r?.identity.id??`reusable`),(0,x.jsxs)(`section`,{className:`screen-stack`,children:[(0,x.jsx)(`h2`,{children:o(`Effective context · observed, not estimated`,`실제 컨텍스트 · 추정이 아닌 관측값`)}),(0,x.jsx)(I,{label:o(`Per-slot context tokens (/props n_ctx)`,`슬롯별 컨텍스트 토큰 (/props n_ctx)`),value:i?.nCtx?.toString()??o(`Unknown`,`알 수 없음`),disabled:!0,hint:o(`Zero/missing is unknown; do not multiply this value into a shared-pool capacity. Unified KV remains separate work (#1815).`,`0 또는 누락은 알 수 없음입니다. 이 값에 슬롯 수를 곱해 공유 풀 용량으로 간주하지 않습니다. 통합 KV는 별도 작업입니다 (#1815).`)}),(0,x.jsx)(I,{label:o(`Configured slots (/props total_slots)`,`설정된 슬롯 (/props total_slots)`),value:i?.totalSlots?.toString()??o(`Unknown`,`알 수 없음`),disabled:!0}),(0,x.jsx)(I,{label:o(`Active resolved KV cache mode (/props)`,`현재 실제 KV 캐시 모드 (/props)`),value:i?.kvCacheMode??o(`Unknown`,`알 수 없음`),disabled:!0}),(0,x.jsxs)(`p`,{children:[o(`Reported context geometry`,`보고된 컨텍스트 구성`),`: `,(0,x.jsx)(`code`,{children:i?.geometry===null||i===null?o(`Unknown / props disabled or unavailable`,`알 수 없음 / props 비활성화 또는 사용 불가`):JSON.stringify(i.geometry)})]}),c===null?null:(0,x.jsx)(Tn,{modelId:c,nCtx:i?.nCtx??null,locale:e},c)]}),c===null?(0,x.jsx)(L,{tone:`info`,title:o(`No ready model selected`,`준비된 모델이 선택되지 않음`),body:o(`Selection never loads a model. Load explicitly from Models to read live settings.`,`모델 선택은 로드를 수행하지 않습니다. 모델 화면에서 명시적으로 로드하여 실시간 설정을 확인하세요.`)}):t.bootstrap?.features.includes(`settings`)===!0?(0,x.jsx)(Dn,{modelId:c,locale:e},c):(0,x.jsx)(L,{tone:`info`,title:o(`Live settings disabled by operator`,`운영자가 실시간 설정을 비활성화함`),body:o(`Restart with --settings to enable this endpoint. WebUI never enables settings, props, metrics or slots implicitly.`,`엔드포인트를 활성화하려면 --settings로 재시작하세요. WebUI는 settings, props, metrics 또는 slots를 자동으로 활성화하지 않습니다.`)})]}):(0,x.jsx)(L,{tone:`warning`,title:o(`Server controls unavailable`,`서버 제어 사용 불가`),body:o(`Connect from Models or Chat, or refresh the connection before changing server settings. Browser appearance and request defaults remain available. Schema mismatch disables server controls.`,`모델이나 채팅에서 연결하거나 연결을 새로 고친 후 서버 설정을 변경하세요. 브라우저 모양과 요청 기본값은 계속 사용할 수 있습니다. 스키마 불일치 시 서버 제어가 비활성화됩니다.`)})]})}var Un=[{id:`models`,key:`nav.models`,icon:`models`},{id:`chat`,key:`nav.chat`,icon:`chat`},{id:`activity`,key:`nav.activity`,icon:`activity`},{id:`settings`,key:`nav.settings`,icon:`settings`}];function Wn(e){let t=(0,_.useRef)(null),n=(0,_.useRef)(e.onCommand),r=(0,_.useRef)(e.onHelp),[i,a]=(0,_.useState)(!1);(0,_.useEffect)(()=>{n.current=e.onCommand,r.current=e.onHelp},[e.onCommand,e.onHelp]),(0,_.useEffect)(()=>{let e=e=>{let t=e.target,i=t instanceof HTMLElement&&(t.isContentEditable||!!t.closest(`[role="combobox"], [role="listbox"]`)||[`INPUT`,`TEXTAREA`,`SELECT`].includes(t.tagName)),o=t instanceof HTMLElement&&!!t.closest(`dialog[open]`);e.isComposing||e.altKey||i||o||((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()===`k`&&(e.preventDefault(),a(!1),n.current()),(e.key===`?`||e.key===`/`&&e.shiftKey)&&(e.preventDefault(),a(!1),r.current()))};return window.addEventListener(`keydown`,e),()=>window.removeEventListener(`keydown`,e)},[]);let o=n=>{let r=Un[(Un.findIndex(t=>t.id===e.route)+n+Un.length)%Un.length];e.onRouteChange(r.id),requestAnimationFrame(()=>t.current?.querySelector(`a[href="#${r.id}"]`)?.focus())},s=e=>{(e.key===`[`||e.key===`]`)&&(e.altKey||e.metaKey||e.ctrlKey||e.nativeEvent.isComposing||(e.preventDefault(),o(e.key===`]`?1:-1)))},c=t=>{e.onRouteChange(t),a(!1)};return(0,x.jsxs)(`div`,{className:`app-shell`,children:[(0,x.jsx)(Gn,{locale:e.locale,route:e.route,onRouteChange:c,onKeyDown:s,ref:t,className:`app-sidebar desktop-sidebar material-glass`,connectionLabel:e.connectionLabel,connectionState:e.connectionState}),(0,x.jsx)(Te,{open:i,title:N(e.locale,`nav.primary`),onClose:()=>a(!1),closeLabel:N(e.locale,`common.close`),testId:`mobile-nav-sheet`,children:(0,x.jsx)(Gn,{locale:e.locale,route:e.route,onRouteChange:c,onKeyDown:s,className:`app-sidebar sheet-sidebar`,connectionLabel:e.connectionLabel,connectionState:e.connectionState})}),(0,x.jsxs)(`main`,{className:`app-main`,"aria-labelledby":`app-title`,children:[(0,x.jsxs)(`header`,{className:`app-toolbar material-glass`,children:[(0,x.jsx)(A,{className:`mobile-menu-button`,label:N(e.locale,`toolbar.menu`),icon:`menu`,onClick:()=>a(!0),"data-testid":P(`toolbar.menu`)}),(0,x.jsxs)(`div`,{className:`toolbar-title`,children:[(0,x.jsx)(`p`,{id:`app-title`,"data-testid":P(`app.title`),children:N(e.locale,`app.title`)}),(0,x.jsx)(`span`,{"data-testid":P(`app.subtitle`),children:N(e.locale,`app.subtitle`)})]}),(0,x.jsx)(`div`,{className:`selected-model`,title:e.selectedModel,children:e.selectedModel}),(0,x.jsxs)(`div`,{className:`toolbar-actions`,children:[e.sessionAction,(0,x.jsx)(A,{label:N(e.locale,`toolbar.command`),icon:`command`,onClick:e.onCommand,"data-testid":P(`toolbar.command`)}),(0,x.jsx)(A,{label:N(e.locale,`toolbar.help`),icon:`help`,onClick:e.onHelp,"data-testid":P(`toolbar.help`)})]})]}),(0,x.jsxs)(`div`,{className:`app-content-grid ${e.inspector?`has-inspector`:``}`.trim(),children:[(0,x.jsx)(`section`,{className:`app-content`,children:e.children}),e.inspector]})]})]})}var Gn=_.forwardRef((e,t)=>(0,x.jsxs)(`aside`,{className:e.className,"aria-label":N(e.locale,`nav.primary`),onKeyDown:e.onKeyDown,ref:t,tabIndex:-1,children:[(0,x.jsxs)(`a`,{className:`brand-mark`,href:`#models`,"aria-label":N(e.locale,`nav.home`),onClick:t=>{t.preventDefault(),e.onRouteChange(`models`)},children:[(0,x.jsx)(`span`,{className:`brand-symbol`,"aria-hidden":`true`,children:`mx`}),(0,x.jsx)(`span`,{className:`brand-name`,children:`mlxcel`})]}),(0,x.jsx)(`nav`,{className:`app-nav`,children:Un.map(t=>(0,x.jsxs)(`a`,{href:`#${t.id}`,"aria-label":N(e.locale,t.key),"aria-current":e.route===t.id?`page`:void 0,"data-testid":P(t.key),onClick:n=>{n.preventDefault(),e.onRouteChange(t.id)},children:[(0,x.jsx)(te,{name:t.icon}),(0,x.jsx)(`span`,{className:`app-nav-label`,children:N(e.locale,t.key)})]},t.id))}),(0,x.jsxs)(`footer`,{children:[(0,x.jsx)(`span`,{className:`connection-dot`,"data-state":e.connectionState,"aria-hidden":`true`}),(0,x.jsx)(`span`,{"data-testid":P(`connection.ready`),children:e.connectionLabel})]})]}));Gn.displayName=`Sidebar`;var Kn={theme:`system`,material:`glass`,glassIntensity:35,reduceMotion:!1,reduceTransparency:!1,highContrast:`system`,locale:`en`},qn=`mlxcel.webui.appearance`;function Jn(e){let t=typeof e==`number`&&Number.isFinite(e)?e:Kn.glassIntensity;return Math.max(0,Math.min(100,Math.round(t)))}function Yn(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function Xn(e){return e===`system`||e===`light`||e===`dark`}function Zn(e){return e===`glass`||e===`tinted`||e===`opaque`}function Qn(e){return e===`system`||e===`on`||e===`off`}function $n(e){return Qn(e)?e:e===!0?`on`:e===!1?`off`:Kn.highContrast}function er(e){return e===`en`||e===`ko`}function tr(){if(typeof window>`u`)return Kn;let e;try{e=window.localStorage.getItem(qn)}catch{return Kn}if(!e)return Kn;try{let t=JSON.parse(e);return Yn(t)?{theme:Xn(t.theme)?t.theme:Kn.theme,material:Zn(t.material)?t.material:Kn.material,glassIntensity:Jn(t.glassIntensity),reduceMotion:t.reduceMotion===!0,reduceTransparency:t.reduceTransparency===!0,highContrast:$n(t.highContrast),locale:er(t.locale)?t.locale:Kn.locale}:Kn}catch{return Kn}}function nr(e){try{window.localStorage.setItem(qn,JSON.stringify(e))}catch{}}function rr(){return typeof CSS>`u`||typeof CSS.supports!=`function`?!1:CSS.supports(`backdrop-filter: blur(1px)`)||CSS.supports(`-webkit-backdrop-filter: blur(1px)`)}function ir(e,t){e.dataset.theme=t.theme,e.dataset.material=t.reduceTransparency?`opaque`:t.material,e.dataset.glassIntensity=String(Jn(t.glassIntensity)),e.dataset.reduceMotion=String(t.reduceMotion),e.dataset.reduceTransparency=String(t.reduceTransparency),e.dataset.highContrast=t.highContrast,e.dataset.backdropFilter=rr()?`supported`:`unsupported`,e.lang=t.locale}function ar(e){return e.value===null||!Number.isFinite(e.value)?`N/A`:`${new Intl.NumberFormat(void 0,{maximumFractionDigits:2}).format(e.value)} ${e.unit}`}function or(e){let t=e.progress;return!t.indeterminate&&t.total_bytes!==null&&t.total_bytes>0?Math.min(100,100*t.completed_bytes/t.total_bytes):void 0}function sr(e){let t=e.selectedModelId===null?void 0:e.runtimes.get(e.selectedModelId);return JSON.stringify({format:`mlxcel-webui-diagnostics-v1`,connection:e.connection,last_successful_at:e.lastSuccessfulAt,operation_counts:Object.fromEntries([`queued`,`running`,`cancelling`,`succeeded`,`failed`,`cancelled`].map(t=>[t,[...e.operations.values()].filter(e=>e.state===t).length])),catalog_count:e.catalog.length,runtime_present:t!==void 0,history_samples:e.runtimeHistory.length,measurements:t===void 0?[]:Object.values(t.measurements).map(e=>({value:e.value,available:e.value!==null}))},null,2)}function cr(e){let t=URL.createObjectURL(new Blob([sr(e)],{type:`application/json`})),n=document.createElement(`a`);n.href=t,n.download=`mlxcel-diagnostics.json`,n.click(),URL.revokeObjectURL(t)}var lr={title:`Activity`,intro:`Operations and runtime observations. Browsing never loads a model.`,session:`History belongs to this server session: up to 200 terminal operations for one hour. A restart clears it; reconnect reconciles the authoritative snapshot.`,export:`Export sanitized diagnostics`,refresh:`Refresh observations`,stale:`Observations are stale. Last values are not current measurements.`,updated:`Last successful snapshot`,pending:`Waiting for the first snapshot`,operations:`Operations`,empty:`No operations in this session`,emptyBody:`Explicit downloads, loads, drains and removals will appear here.`,cancel:`Request cancellation`,cancelling:`Cancellation requested; waiting for worker acknowledgement.`,failure:`The action failed. Check its state and refresh before retrying from Models.`,cancelFailed:`Cancellation could not be confirmed. Refresh to reconcile the operation.`,cancelUnsupported:`Cancellation is unavailable for this operation; worker execution is not interrupted.`,runtime:`Selected model runtime`,select:`Select a model to observe`,selectBody:`Use the model picker. Selection does not load the model.`,unavailable:`N/A — no authoritative sample is available.`,memory:`Memory figures have separate scopes and may overlap. Do not add process, allocator, device and cache values together.`,timing:`Server counters are not browser TTFT. Chat shows request-send to first reasoning/content delta separately; no rendered word counts are used as tokens.`,slots:`Request slots`,slot:`Slot`,processing:`Processing`,idle:`Idle`,context:`Request context window`,pool:`Shared pool context`,parallel:`Effective parallelism`,noContext:`N/A — context denominator is unknown; no occupancy percentage is inferred.`,history:`Show recent history`,hideHistory:`Hide recent history`,historyNote:`Five-minute in-memory history, at most 150 two-second samples. Hidden tabs stop observation; gaps and counter resets are not interpolated.`,measure:`Measurement`,value:`Value`,scope:`Scope`,observed:`Observed at`,reason:`Availability / source`,metricDetails:`All measurements and sources`,unavailableCount:`Unavailable measurements`,unavailableReason:`Some sources are disabled or do not publish an authoritative sample. Expand details for individual reasons.`,noPrimary:`No primary request counters are available.`,metricLabels:{active_requests:`Active requests`,queued_requests:`Queued requests`,completed_requests_total:`Total completed requests`,completion_tokens_total:`Total completion tokens`,gpu_utilization:`GPU utilization`,ttft:`Time to first token`,decode_rate:`Decode rate`,process_resident_bytes:`Process resident memory`,allocator_active_bytes:`Active allocator memory`,allocator_cache_bytes:`Cached allocator memory`,allocator_peak_bytes:`Peak allocator memory`,device_total_bytes:`Device memory`,model_weights_bytes:`Model weights estimate`,kv_cache_bytes:`KV cache memory`,generation_time_ms_total:`Total generation time`,decode_tokens_total:`Accepted decode tokens`,decode_time_us_total:`Total decode time`,prompt_cache_bytes:`Prompt cache memory`,prompt_cache_entries:`Prompt cache entries`},bytes:`bytes downloaded`,details:`Operation details`,status:`Operation status`,unknownTime:`Unknown`},ur={title:`활동`,intro:`작업과 런타임 관측입니다. 조회는 모델을 로드하지 않습니다.`,session:`기록은 서버 세션에 속합니다. 완료 작업 최대 200개를 한 시간 보관하며, 재시작 시 초기화하고 재연결 시 서버 상태와 대조합니다.`,export:`민감 정보 없는 진단 내보내기`,refresh:`관측 새로고침`,stale:`관측이 최신 상태가 아닙니다. 마지막 값은 현재 측정값이 아닙니다.`,updated:`마지막 성공 스냅샷`,pending:`첫 스냅샷을 기다리는 중`,operations:`작업`,empty:`이 세션의 작업 없음`,emptyBody:`명시적인 다운로드, 로드, 드레인 및 삭제 작업이 표시됩니다.`,cancel:`취소 요청`,cancelling:`취소 요청됨. 작업자 확인을 기다립니다.`,failure:`작업에 실패했습니다. 상태를 확인하고 새로고침한 후 모델 화면에서 재시도하세요.`,cancelFailed:`취소를 확인하지 못했습니다. 새로고침하여 작업 상태를 대조하세요.`,cancelUnsupported:`이 작업은 취소할 수 없습니다. 실행 중인 작업자를 강제로 중단하지 않습니다.`,runtime:`선택한 모델 런타임`,select:`관측할 모델 선택`,selectBody:`모델 선택기를 사용하세요. 선택만으로 모델을 로드하지 않습니다.`,unavailable:`N/A — 확인된 관측값이 없습니다.`,memory:`메모리 수치는 서로 다른 범위이며 중복될 수 있습니다. 프로세스, 할당자, 장치 및 캐시 값을 합산하지 마세요.`,timing:`서버 카운터는 브라우저 TTFT가 아닙니다. 채팅은 요청 전송부터 첫 추론/내용 델타까지 별도로 표시하며, 단어 수를 토큰으로 사용하지 않습니다.`,slots:`요청 슬롯`,slot:`슬롯`,processing:`처리 중`,idle:`유휴`,context:`요청 컨텍스트 한도`,pool:`공유 풀 컨텍스트`,parallel:`실제 병렬도`,noContext:`N/A — 컨텍스트 분모를 알 수 없어 점유율을 추정하지 않습니다.`,history:`최근 기록 보기`,hideHistory:`최근 기록 숨기기`,historyNote:`메모리 내 5분 기록이며 2초 간격 최대 150개입니다. 숨겨진 탭은 관측을 중지합니다. 공백과 카운터 초기화를 보간하지 않습니다.`,metricDetails:`전체 측정값과 출처`,unavailableCount:`사용할 수 없는 측정값`,unavailableReason:`일부 출처가 비활성화되었거나 확인된 관측값을 제공하지 않습니다. 상세 내용을 펼쳐 개별 사유를 확인하세요.`,noPrimary:`사용 가능한 주요 요청 카운터가 없습니다.`,metricLabels:{active_requests:`활성 요청`,queued_requests:`대기 요청`,completed_requests_total:`누적 완료 요청`,completion_tokens_total:`누적 완료 토큰`,gpu_utilization:`GPU 사용률`,ttft:`첫 토큰 도달 시간`,decode_rate:`디코드 속도`,process_resident_bytes:`프로세스 상주 메모리`,allocator_active_bytes:`할당자 활성 메모리`,allocator_cache_bytes:`할당자 캐시 메모리`,allocator_peak_bytes:`할당자 최대 메모리`,device_total_bytes:`장치 메모리`,model_weights_bytes:`모델 가중치 추정량`,kv_cache_bytes:`KV 캐시 메모리`,generation_time_ms_total:`누적 생성 시간`,decode_tokens_total:`수락된 디코드 토큰`,decode_time_us_total:`누적 디코드 시간`,prompt_cache_bytes:`프롬프트 캐시 메모리`,prompt_cache_entries:`프롬프트 캐시 항목`},measure:`측정 항목`,value:`값`,scope:`범위`,observed:`관측 시각`,reason:`가용성 / 출처`,bytes:`다운로드 바이트`,details:`작업 상세`,status:`작업 상태`,unknownTime:`알 수 없음`};function dr(e){return e===`ko`?ur:lr}function fr({operations:e,locale:t,stale:n}){let r=dr(t),i=[...e.values()].sort((e,t)=>Number(Nt(e))-Number(Nt(t))||t.updated_at.localeCompare(e.updated_at));return(0,x.jsxs)(`section`,{"aria-label":r.operations,children:[(0,x.jsx)(`h2`,{children:r.operations}),(0,x.jsxs)(`p`,{role:`status`,"aria-live":`polite`,children:[r.operations,`: `,i.filter(e=>!Nt(e)).length,` `,t===`ko`?`진행 중`:`active`,` · `,i.filter(e=>e.state===`failed`).length,` `,t===`ko`?`실패`:`failed`]}),(0,x.jsx)(`p`,{children:r.session}),i.length===0?(0,x.jsx)(_e,{title:r.empty,body:r.emptyBody}):(0,x.jsx)(`ol`,{className:`activity-operations`,children:i.map(e=>(0,x.jsx)(`li`,{children:(0,x.jsx)(pr,{operation:e,locale:t,stale:n})},e.operation_id))})]})}function pr({operation:e,locale:t,stale:n}){let r=dr(t),i=mn(),a=pn(),o=e.target,s=o.target_kind===`model`?a.catalog.find(e=>e.identity.id===o.model_id)?.identity.display_name??o.model_id:o.target_kind===`download`?o.repo_id:o.target_kind,[c,l]=(0,_.useState)(!1),[u,d]=(0,_.useState)(!1),f=Nt(e),p=async()=>{l(!0),d(!1);try{await i.cancelOperation(e.operation_id)}catch{d(!0)}finally{l(!1)}};return(0,x.jsxs)(`article`,{className:`activity-operation`,children:[(0,x.jsxs)(`header`,{children:[(0,x.jsx)(`strong`,{children:e.kind.replaceAll(`_`,` `)}),(0,x.jsx)(he,{state:e.state===`failed`?`failed`:f?`unloaded`:e.state===`cancelling`?`draining`:`loading`,label:r.status,children:e.state})]}),(0,x.jsx)(`p`,{children:s}),(0,x.jsx)(`p`,{children:(0,x.jsx)(`time`,{dateTime:e.updated_at,children:new Date(e.updated_at).toLocaleString(t)})}),!f&&e.kind===`download`?(0,x.jsx)(ge,{label:r.bytes,value:or(e),detail:`${e.progress.completed_bytes.toLocaleString(t)} / ${e.progress.total_bytes?.toLocaleString(t)??`N/A`} bytes`}):null,e.state===`cancelling`?(0,x.jsx)(`p`,{role:`status`,children:r.cancelling}):null,e.state===`failed`?(0,x.jsx)(L,{title:r.failure,body:e.error?.code??`operation_failed`}):null,(0,x.jsxs)(`details`,{children:[(0,x.jsx)(`summary`,{children:r.details}),(0,x.jsx)(`p`,{children:e.operation_id}),(0,x.jsx)(`p`,{children:e.target.target_kind===`model`?e.target.model_id:e.target.target_kind})]}),!f&&e.state!==`cancelling`?e.cancellable?(0,x.jsx)(k,{disabled:n,busy:c,onClick:()=>{p()},children:r.cancel}):(0,x.jsx)(`p`,{children:r.cancelUnsupported}):null,u?(0,x.jsx)(L,{title:r.cancelFailed,body:r.refresh}):null]})}function mr({runtime:e,locale:t,stale:n=!1}){let r=dr(t);if(e===void 0)return(0,x.jsx)(_e,{title:r.runtime,body:r.unavailable});let i=Object.entries(e.measurements),a=new Set([`active_requests`,`queued_requests`,`completed_requests_total`,`completion_tokens_total`]),o=i.filter(([e,t])=>a.has(e)&&t.value!==null&&Number.isFinite(t.value)),s=i.filter(([,e])=>e.value===null||!Number.isFinite(e.value)).length,c=e=>r.metricLabels[e]??e.replaceAll(`_`,` `);return(0,x.jsxs)(`section`,{"aria-label":r.runtime,children:[(0,x.jsx)(`h2`,{children:r.runtime}),n?(0,x.jsx)(L,{tone:`warning`,title:r.stale,body:r.observed}):null,(0,x.jsx)(`div`,{className:`activity-metrics activity-metrics--summary`,"data-testid":`runtime-summary`,children:o.map(([e,t])=>(0,x.jsxs)(`dl`,{className:`activity-metric`,children:[(0,x.jsx)(`dt`,{children:c(e)}),(0,x.jsx)(`dd`,{children:ar(t)})]},e))}),o.length===0?(0,x.jsx)(`p`,{children:r.noPrimary}):null,s>0?(0,x.jsxs)(`p`,{children:[r.unavailableCount,`: `,s,`. `,r.unavailableReason]}):null,(0,x.jsx)(`h3`,{children:r.slots}),(0,x.jsxs)(`p`,{children:[r.parallel,`: `,e.slots.effective_parallelism??`N/A`,` / `,e.slots.configured_parallelism]}),(0,x.jsxs)(`p`,{children:[r.context,`: `,e.slots.request_context_tokens??`N/A`,` tokens · `,r.pool,`: `,e.slots.shared_pool_context_tokens??`N/A`,` tokens`]}),e.slots.reason===null?null:(0,x.jsx)(`p`,{children:e.slots.reason}),(0,x.jsxs)(`p`,{children:[r.observed,`: `,e.slots.measured_at===null?r.unknownTime:new Date(e.slots.measured_at).toLocaleString(t)]}),e.slots.available?e.slots.items.map(t=>(0,x.jsxs)(`section`,{"aria-label":`${r.slot} ${t.id}`,children:[(0,x.jsxs)(`h4`,{children:[r.slot,` `,t.id,` · `,t.processing?r.processing:r.idle]}),t.prompt_tokens!==null&&e.slots.request_context_tokens!==null&&e.slots.request_context_tokens>0?(0,x.jsx)(ge,{label:`${r.slot} ${t.id} ${r.context}`,value:t.prompt_tokens/e.slots.request_context_tokens*100,detail:`${t.prompt_tokens} / ${e.slots.request_context_tokens} tokens`}):(0,x.jsx)(`p`,{children:r.noContext}),(0,x.jsxs)(`p`,{children:[`Accepted decode: `,t.decoded_tokens??`N/A`,` tokens · Cached prompt: `,t.cached_prompt_tokens??`N/A`,` tokens`]})]},t.id)):e.slots.reason===null?(0,x.jsx)(`p`,{children:r.unavailable}):null,(0,x.jsxs)(`details`,{className:`activity-measurement-details`,children:[(0,x.jsx)(`summary`,{children:r.metricDetails}),(0,x.jsx)(`p`,{children:r.memory}),(0,x.jsx)(`p`,{children:r.timing}),(0,x.jsx)(`div`,{className:`activity-metrics`,children:i.map(([e,n])=>(0,x.jsxs)(`dl`,{className:`activity-metric`,children:[(0,x.jsx)(`dt`,{children:c(e)}),(0,x.jsx)(`dd`,{children:ar(n)}),(0,x.jsxs)(`dd`,{children:[r.scope,`: `,n.scope]}),(0,x.jsxs)(`dd`,{children:[r.observed,`: `,n.measured_at===null?r.unknownTime:new Date(n.measured_at).toLocaleString(t)]}),n.reason===null?null:(0,x.jsx)(`dd`,{children:n.reason})]},e))})]})]})}var hr=(function(){let e=typeof document<`u`&&document.createElement(`link`).relList;return e&&e.supports&&e.supports(`modulepreload`)?`modulepreload`:`preload`})(),gr=function(e,t){return new URL(e,t).href},_r={},vr=function(e,t,n){let r=Promise.resolve();if(t&&t.length>0){let e=document.getElementsByTagName(`link`),i=document.querySelector(`meta[property=csp-nonce]`),a=i?.nonce||i?.getAttribute(`nonce`);function o(e){return Promise.all(e.map(e=>Promise.resolve(e).then(e=>({status:`fulfilled`,value:e}),e=>({status:`rejected`,reason:e}))))}function s(e){return import.meta.resolve?import.meta.resolve(e):new URL(e,import.meta.url).href}r=o(t.map(t=>{if(t=gr(t,n),t=s(t),t in _r)return;_r[t]=!0;let r=t.endsWith(`.css`);for(let n=e.length-1;n>=0;n--){let i=e[n];if(i.href===t&&(!r||i.rel===`stylesheet`))return}let i=document.createElement(`link`);if(i.rel=r?`stylesheet`:hr,r||(i.as=`script`),i.crossOrigin=``,i.href=t,a&&i.setAttribute(`nonce`,a),document.head.appendChild(i),r)return new Promise((e,n)=>{i.addEventListener(`load`,e),i.addEventListener(`error`,()=>n(Error(`Unable to preload CSS for ${t}`)))})}).filter(e=>e!==void 0))}function i(e){let t=new Event(`vite:preloadError`,{cancelable:!0});if(t.payload=e,window.dispatchEvent(t),!t.defaultPrevented)throw e}return r.then(t=>{for(let e of t||[])e.status===`rejected`&&i(e.reason);return e().catch(i)})},yr=(0,_.lazy)(()=>vr(()=>import(`./history-rhrGVPNb.js`),[],import.meta.url));function br({locale:e}){let t=pn(),n=mn(),r=dr(e),[i,a]=(0,_.useState)(!1),o=![`ready`,`streaming`,`polling`].includes(t.connection),s=t.runtimeHistory.at(-1)?.receivedAt??null,c=o||s===null||t.lastUpdatedAt!==null&&t.lastUpdatedAt-s>4e3,l=t.selectedModelId===null?void 0:t.runtimes.get(t.selectedModelId);return(0,x.jsxs)(`div`,{className:`screen-stack`,"data-testid":`activity-page`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`h1`,{"data-testid":P(`activity.title`),children:r.title}),(0,x.jsx)(`p`,{children:r.intro}),(0,x.jsxs)(`p`,{children:[r.updated,`: `,t.lastSuccessfulAt===null?r.pending:(0,x.jsx)(`time`,{children:new Date(t.lastSuccessfulAt).toLocaleString(e)})]})]}),(0,x.jsx)(F,{locale:e,label:r.select,value:t.selectedModelId??``,onChange:e=>n.selectModel(e||null),options:[{value:``,label:r.select},...t.catalog.map(e=>({value:e.identity.id,label:e.identity.display_name}))]}),(0,x.jsxs)(`div`,{className:`activity-toolbar`,children:[(0,x.jsx)(k,{onClick:()=>{n.refresh()},children:r.refresh}),(0,x.jsx)(k,{onClick:()=>cr(t),children:r.export})]}),o?(0,x.jsx)(L,{tone:`warning`,title:r.stale,body:r.refresh}):null,(0,x.jsx)(fr,{operations:t.operations,locale:e,stale:o}),t.selectedModelId===null?(0,x.jsx)(_e,{title:r.select,body:r.selectBody}):(0,x.jsx)(mr,{runtime:l,locale:e,stale:c}),(0,x.jsx)(k,{onClick:()=>a(!i),children:i?r.hideHistory:r.history}),i?(0,x.jsx)(_.Suspense,{fallback:(0,x.jsx)(`p`,{children:r.pending}),children:(0,x.jsx)(yr,{samples:t.runtimeHistory,locale:e})}):null]})}function xr(e){return e instanceof St?e.status===401?`wrong-key`:e.status===403?`forbidden`:`generic`:e instanceof R||e instanceof Error&&e.name===`ValidationError`?`schema`:e instanceof TypeError||e instanceof Error&&/fetch|network|offline|connection/i.test(e.message)?`offline`:`generic`}function Sr(e,t){let n=Mr(t);return n===null?N(e,`model.selected.none`):`${n.identity.display_name} · ${Nr(e,n.lifecycle.state)}`}function Cr(e,t){return t.bootstrap===null?N(e,`connection.ready`):N(e,`connection.footer.connected`,{mode:t.bootstrap.server.mode,version:t.bootstrap.server.build.version,sequence:Pr(e,t),status:wr(e,t.connection)})}function wr(e,t){return N(e,{idle:`connection.status.idle`,bootstrapping:`connection.status.bootstrapping`,ready:`connection.status.ready`,streaming:`connection.status.streaming`,polling:`connection.status.polling`,offline:`connection.status.offline`,stale:`connection.status.stale`,unauthorized:`connection.status.unauthorized`,forbidden:`connection.status.forbidden`,"schema-mismatch":`connection.status.schema_mismatch`,error:`connection.status.error`}[t])}function Tr(e){return e.snapshot.auth.status===`authenticated`?e.snapshot.connection===`schema-mismatch`?(0,x.jsx)(Or,{...e}):(0,x.jsx)(Dr,{...e}):(0,x.jsx)(Er,{...e})}function Er(e){return e.authFailure===`schema`?(0,x.jsx)(Or,{...e}):(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title}),(0,x.jsx)(`p`,{"data-testid":P(`connection.prompt.body`),children:N(e.locale,`connection.prompt.body`)})]}),(0,x.jsx)(Ae,{title:N(e.locale,`state.unauthorized.title`),body:N(e.locale,`state.unauthorized.body`),tokenLabel:N(e.locale,`login.token.label`),tokenHelp:N(e.locale,`login.token.help`),submitLabel:N(e.locale,`login.submit`),logoutLabel:N(e.locale,`login.logout`),error:e.authFailure===null?void 0:jr(e.locale,e.authFailure),busy:e.snapshot.auth.status===`authenticating`,onSubmit:e.onLogin,onLogout:e.onLogout,testId:P(`auth.login`)})]})}function Dr(e){let t=e.snapshot.connection===`offline`||e.snapshot.connection===`stale`||e.snapshot.connection===`error`||e.snapshot.connection===`unauthorized`||e.snapshot.connection===`forbidden`;return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title}),(0,x.jsx)(`p`,{"data-testid":P(`connection.authenticated.body`),children:N(e.locale,`connection.authenticated.body`)})]}),(0,x.jsx)(L,{tone:`info`,title:N(e.locale,`connection.authenticated.title`),body:kr(e.locale,e.snapshot),testId:P(`connection.authenticated.detail`)}),t?(0,x.jsx)(L,{title:N(e.locale,`connection.error.title`),body:Ar(e.locale,e.snapshot.connection),action:(0,x.jsx)(k,{onClick:e.onRetry,children:N(e.locale,`common.retry`)}),testId:P(`connection.error.title`)}):null]})}function Or(e){return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title})]}),(0,x.jsx)(je,{title:N(e.locale,`state.schema_mismatch.title`),body:N(e.locale,`state.schema_mismatch.body`),actionLabel:N(e.locale,`common.reload`),onRecover:e.onRecoverSchema})]})}function kr(e,t){let n=t.bootstrap;return n===null?N(e,`connection.prompt.detail`):N(e,`connection.authenticated.detail`,{mode:n.server.mode,version:n.server.build.version,status:wr(e,t.connection),count:t.catalogSequence===null?N(e,`connection.snapshot.pending`):String(t.catalog.length),operations:t.resourceFences.operationsSnapshot===null?N(e,`connection.snapshot.pending`):String(t.operations.size),sequence:Pr(e,t)})}function Ar(e,t){return t===`offline`?N(e,`state.offline.body`):t===`stale`?N(e,`connection.error.stale`):t===`forbidden`?N(e,`connection.error.forbidden`):t===`unauthorized`?N(e,`connection.error.unauthorized`):N(e,`connection.error.generic`)}function jr(e,t){return t===`wrong-key`?N(e,`login.error.wrong_key`):t===`offline`?N(e,`login.error.offline`):t===`forbidden`?N(e,`login.error.forbidden`):t===`schema`?N(e,`login.error.schema`):N(e,`login.error.generic`)}function Mr(e){return e.selectedModelId===null?null:e.catalog.find(t=>t.identity.id===e.selectedModelId)??null}function Nr(e,t){return N(e,{unloaded:`models.status.unloaded`,loading:`models.status.loading`,ready:`models.status.ready`,draining:`models.status.draining`,unloading:`models.status.unloading`,failed:`models.status.failed`}[t])}function Pr(e,t){return String(t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot??N(e,`connection.snapshot.pending`))}var Fr=e=>[`succeeded`,`failed`,`cancelled`].includes(e.state),Ir=e=>e.auth.status===`authenticated`&&e.bootstrap!==null&&e.catalogSequence!==null&&[`ready`,`streaming`,`polling`].includes(e.connection);function Lr(e,t){return Ir(e)&&e.bootstrap?.actions[t]?.state===`enabled`}function Rr(e,t){return[...e.operations.values()].some(e=>!Fr(e)&&e.target.target_kind===`model`&&(e.target.model_id===t||e.target.eviction_target_id===t))||[...e.pendingReconciliations.values()].some(e=>e.modelId===t)}function zr(e,t){return Lr(e,`load`)&&t.supported&&t.complete&&t.metadata.support.runnable_on_backend&&!t.lifecycle.busy&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Rr(e,t.identity.id)}function Br(e,t){return Lr(e,`unload`)&&t.lifecycle.state===`ready`&&!Rr(e,t.identity.id)}function Vr(e,t){return Lr(e,`cache_delete`)&&t.identity.source===`cache`&&t.removable&&t.removal.eligible&&!t.lifecycle.busy&&t.lifecycle.active_requests===0&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Rr(e,t.identity.id)}function Hr(e,t){return Ir(e)&&t.lifecycle.state===`ready`&&t.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available)}function Ur(e,t){return e.catalog.filter(n=>n.identity.id!==t&&Br(e,n)&&!n.lifecycle.busy&&n.lifecycle.active_requests===0&&n.lifecycle.draining_requests===0)}function Wr(e,t){if(e===null)return N(t,`models.library.unknown`);if(e===0)return`0 B`;let n=Math.min(4,Math.floor(Math.log(e)/Math.log(1024)));return`${(e/1024**n).toLocaleString(t,{maximumFractionDigits:1})} ${[`B`,`KiB`,`MiB`,`GiB`,`TiB`][n]}`}function Gr(e,t){return e instanceof St?e.envelope?.error.code===`stale_revision`?N(t,`models.library.stale`):e.message:N(t,`models.library.pending`)}function Kr(e,t){let n=t.query.trim().toLocaleLowerCase();return e.filter(e=>(!n||[e.identity.display_name,e.identity.inference_id,e.metadata.architecture??``].some(e=>e.toLocaleLowerCase().includes(n)))&&(!t.source||e.identity.source===t.source)&&(!t.task||e.capabilities.some(e=>e.task===t.task))&&(!t.status||e.lifecycle.state===t.status)).sort((e,n)=>{let r=e=>t.sort===`status`?e.lifecycle.state:t.sort===`source`?e.identity.source:t.sort===`task`?e.capabilities.map(e=>e.task).join(`,`):e.identity.display_name;return r(e).localeCompare(r(n))||e.identity.id.localeCompare(n.identity.id)})}function qr(e){let t=e.split(`/`);return t.length===2&&t.every(e=>e.length<=96&&/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(e))}function Jr(e){return e.length===0||e.length<=128&&e.split(`/`).every(e=>e!==`.`&&e!==`..`&&/^[A-Za-z0-9._~+-]+$/.test(e))}function Yr({value:e,state:t,locale:n,busy:r,onClose:i,onConfirm:a}){let[o,s]=(0,_.useState)(``),[c,l]=(0,_.useState)(),u=e.kind===`capacity`?Ur(t,e.entry.identity.id):[],d=e.kind===`cancel`?e.operation.target.target_kind===`download`?e.operation.target.repo_id:e.operation.operation_id:e.entry.identity.display_name,f=N(n,e.kind===`delete`?`models.library.delete`:e.kind===`unload`?`models.unload`:e.kind===`cancel`?`models.library.cancel_download`:`models.library.capacity`),p=e.kind===`capacity`?N(n,`models.library.capacity_body`):e.kind===`cancel`?N(n,`models.library.cancel_body`,{name:d}):N(n,e.kind===`delete`?`models.library.delete_body`:`models.library.unload_body`,{name:d,source:e.entry.identity.source,count:String(e.entry.lifecycle.active_requests)}),m=t.serverInstanceId===e.instance&&(e.kind===`cancel`||t.catalog.some(t=>t.identity.id===e.entry.identity.id&&t.identity.revision===e.entry.identity.revision)),h=m&&(e.kind===`delete`?o===e.entry.identity.id:e.kind!==`capacity`||u.some(e=>e.identity.id===o&&e.identity.revision===c));return(0,x.jsxs)(we,{open:!0,title:f,onClose:i,closeLabel:N(n,`common.close`),testId:`models-confirm`,children:[(0,x.jsx)(`p`,{children:p}),e.kind===`delete`?(0,x.jsx)(`code`,{className:`models-wrap`,children:e.entry.identity.id}):null,e.kind===`capacity`&&o&&!h?(0,x.jsx)(`p`,{role:`alert`,children:N(n,`models.library.stale`)}):null,m?null:(0,x.jsx)(`p`,{role:`alert`,children:N(n,`models.library.stale`)}),e.kind===`delete`?(0,x.jsx)(I,{label:N(n,`models.library.confirm_name`),value:o,onChange:s,testId:`models-confirm-name`}):null,e.kind===`capacity`?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(`ul`,{children:t.catalog.filter(e=>[`ready`,`loading`,`draining`,`unloading`].includes(e.lifecycle.state)).map(e=>(0,x.jsxs)(`li`,{children:[e.identity.display_name,` · `,e.lifecycle.state,` · `,N(n,`models.library.active`),`:`,` `,e.lifecycle.active_requests]},e.identity.id))}),(0,x.jsx)(F,{locale:n,label:N(n,`models.library.eviction`),value:o,onChange:e=>{s(e),l(u.find(t=>t.identity.id===e)?.identity.revision)},options:[{value:``,label:N(n,`models.library.choose`)},...u.map(e=>({value:e.identity.id,label:e.identity.display_name}))],testId:`models-eviction-target`})]}):null,(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(k,{onClick:i,children:N(n,`models.library.cancel`)}),(0,x.jsx)(k,{tone:`danger`,"data-testid":`models-confirm-submit`,busy:r,disabled:!h||r,onClick:()=>a(e.kind===`capacity`?o:void 0,c),children:N(n,e.kind===`capacity`?`models.library.evict_load`:`models.library.confirm`)})]})]})}function Xr({locale:e,state:t,busy:n,onClose:r,onDownload:i,initial:a,error:o}){let[s,c]=(0,_.useState)(a?.repo??``),[l,u]=(0,_.useState)(a?.revision??``),[d,f]=(0,_.useState)(!1),p=qr(s.trim())&&Jr(l.trim());return(0,x.jsx)(we,{open:!0,title:N(e,`models.library.add`),onClose:r,closeLabel:N(e,`common.close`),testId:`models-add-dialog`,children:(0,x.jsxs)(`form`,{onSubmit:e=>{e.preventDefault(),p&&d&&!n&&i(s.trim(),l.trim())},children:[o?(0,x.jsx)(`p`,{role:`alert`,children:o}):null,(0,x.jsx)(`p`,{children:N(e,`models.library.network`)}),(0,x.jsx)(`ul`,{children:t.bootstrap?.roots.filter(e=>e.kind===`cache`).map((e,t)=>(0,x.jsx)(`li`,{children:e.display_name},t))}),(0,x.jsx)(I,{label:N(e,`models.library.repo`),value:s,onChange:c,testId:`models-repo`,error:s&&!qr(s.trim())?N(e,`models.library.repo_invalid`):void 0}),(0,x.jsx)(I,{label:N(e,`models.library.revision`),value:l,onChange:u,testId:`models-revision`}),(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:d,onChange:e=>f(e.currentTarget.checked),"data-testid":`models-public-repo`}),(0,x.jsx)(`span`,{children:N(e,`models.library.public`)})]}),(0,x.jsx)(`p`,{children:N(e,`models.library.private`)}),(0,x.jsx)(`pre`,{className:`models-wrap`,children:`hf download OWNER/REPO --local-dir /path/to/models/checkpoint`}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(k,{onClick:r,children:N(e,`models.library.cancel`)}),(0,x.jsx)(k,{type:`submit`,tone:`primary`,busy:n,disabled:!p||!d||n,"data-testid":`models-download-submit`,children:N(e,`models.library.add`)})]})]})})}function Zr({locale:e}){return(0,x.jsxs)(`p`,{"data-testid":`models-pending-profile`,children:[e===`ko`?`대기 중인 브라우저 프로필이며 아직 적용되지 않았습니다. 명시적 CLI/환경 설정이 우선합니다. 실제 값은 로드 후 확인하세요. `:`Pending browser profile; not applied yet. Explicit CLI/environment settings take precedence. Read effective values after loading. `,(0,x.jsx)(`a`,{href:`#settings`,children:e===`ko`?`프로필 편집`:`Edit profile`})]})}function Qr({entry:e,profile:t,state:n,locale:r,busy:i,onAction:a,onChat:o}){let s=e.metadata,c=n.runtimes.get(e.identity.id),l=c?.revision===e.identity.revision?c.settings.effective.ctx_size:null,u=e=>e==null?N(r,`models.library.unknown`):String(e),d=e=>N(r,e?`models.library.yes`:`models.library.no`),f=[[N(r,`models.library.source`),e.identity.source],[N(r,`models.library.architecture`),u(s.architecture)],[N(r,`models.library.backend`),d(s.support.runnable_on_backend)],[N(r,`models.library.tested`),d(s.support.tested_checkpoint)],[N(r,`models.library.disk`),Wr(s.disk_bytes,r)],[N(r,`models.library.memory`),Wr(s.memory_estimate_bytes,r)],[N(r,`models.library.context`),u(typeof l==`number`?l:null)],[N(r,`models.library.profile`),Object.keys(t).length?JSON.stringify(t):N(r,`models.library.defaults`)],[N(r,`models.library.active`),String(e.lifecycle.active_requests)],[N(r,`models.library.worker`),d(e.lifecycle.worker_exit_observed)]],p=[s.support.reason,s.support.architecturally_supported_reason,s.support.runnable_on_backend_reason,s.support.complete_reason,s.support.tested_checkpoint_reason,...Object.values(s.unknown_reasons),...e.capabilities.map(e=>e.reason)].filter(e=>!!e);return(0,x.jsxs)(De,{title:N(r,`models.library.details`),children:[(0,x.jsx)(`h3`,{children:e.identity.display_name}),(0,x.jsx)(`code`,{className:`models-wrap`,children:e.identity.id}),(0,x.jsx)(he,{state:e.lifecycle.state,children:Nr(r,e.lifecycle.state)}),(0,x.jsx)(`dl`,{children:f.map(([e,t])=>(0,x.jsxs)(_.Fragment,{children:[(0,x.jsx)(`dt`,{children:e}),(0,x.jsx)(`dd`,{children:t})]},e))}),Object.keys(t).length?(0,x.jsx)(Zr,{locale:r}):null,(0,x.jsx)(`p`,{children:N(r,`models.library.tested_help`)}),[...new Set(p)].map(e=>(0,x.jsx)(`p`,{children:e},e)),e.lifecycle.last_error?(0,x.jsxs)(`p`,{role:`alert`,children:[N(r,`models.library.last_error`),`: `,e.lifecycle.last_error]}):null,(0,x.jsxs)(`div`,{className:`button-row`,children:[(0,x.jsx)(k,{"data-testid":`models-load`,disabled:i||!zr(n,e),onClick:()=>a(`load`),children:N(r,`models.load`)}),(0,x.jsx)(k,{"data-testid":`models-use-chat`,disabled:i||!Hr(n,e),onClick:o,children:N(r,`models.library.chat`)}),(0,x.jsx)(k,{"data-testid":`models-unload`,disabled:i||!Br(n,e),onClick:()=>a(`unload`),children:N(r,`models.unload`)}),e.identity.source===`cache`?(0,x.jsx)(k,{tone:`danger`,"data-testid":`models-delete`,disabled:i||!Vr(n,e),onClick:()=>a(`delete`),children:N(r,`models.library.delete`)}):null]}),Hr(n,e)?null:(0,x.jsx)(`p`,{children:N(r,`models.library.chat_reason`)}),e.removal.eligible?null:(0,x.jsxs)(`p`,{children:[e.removal.reason,` `,e.removal.instructions]}),Object.entries(n.bootstrap?.actions??{}).filter(([,e])=>e.state!==`enabled`).map(([e,t])=>(0,x.jsxs)(`p`,{children:[e,`: `,t.reason,` `,t.instructions]},e)),(0,x.jsx)(`a`,{href:`https://github.com/lablup/mlxcel/blob/main/docs/llama-server-compat.md`,target:`_blank`,rel:`noreferrer`,children:N(r,`models.library.api`)}),(0,x.jsx)(`ul`,{children:e.capabilities.map(e=>(0,x.jsxs)(`li`,{children:[e.task,` · `,e.phase,` · `,d(e.available)]},`${e.task}-${e.phase}`))})]})}function $r({state:e,locale:t,busy:n,downloadPending:r,onCancel:i,onRetry:a,onCapacity:o}){let s=[...e.operations.values()].filter(e=>[`download`,`model_load`,`model_unload`,`model_removal`,`catalog_refresh`].includes(e.kind)).sort((e,t)=>Number(Fr(e))-Number(Fr(t))||t.created_at.localeCompare(e.created_at));return!s.length&&!e.pendingReconciliations.size?(0,x.jsx)(x.Fragment,{}):(0,x.jsxs)(`section`,{className:`models-operations`,"aria-label":N(t,`models.library.operations`),children:[(0,x.jsx)(`h2`,{children:N(t,`models.library.operations`)}),e.pendingReconciliations.size?(0,x.jsx)(`p`,{role:`status`,"data-testid":`models-pending`,children:N(t,`models.library.pending`)}):null,(0,x.jsx)(`ul`,{className:`ds-list`,children:s.slice(0,64).map(s=>(0,x.jsx)(`li`,{"data-testid":`models-operation`,children:(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`strong`,{children:ti(s,e)}),(0,x.jsxs)(`p`,{children:[(0,x.jsx)(`code`,{children:s.operation_id}),s.target.target_kind===`model`?(0,x.jsxs)(`span`,{children:[` `,`· `,(0,x.jsx)(`code`,{children:s.target.model_id})]}):null,` `,`· `,N(t,`models.library.operation_state`),`: `,s.state]}),s.kind===`download`?(0,x.jsx)(ge,{label:N(t,`models.library.progress`),value:!s.progress.indeterminate&&s.progress.total_bytes!==null&&s.progress.total_bytes>0?s.progress.completed_bytes/s.progress.total_bytes*100:void 0,detail:`${Wr(s.progress.completed_bytes,t)} / ${Wr(s.progress.total_bytes,t)}`}):null,s.error?(0,x.jsxs)(`p`,{role:`alert`,children:[s.error.code,`: `,s.error.message]}):null,s.result&&`lifecycle`in s.result?(0,x.jsxs)(`p`,{children:[s.result.lifecycle.state,` · `,N(t,`models.library.worker`),`:`,` `,N(t,s.result.lifecycle.worker_exit_observed?`models.library.yes`:`models.library.no`)]}):null,s.cancel_reason?(0,x.jsx)(`p`,{children:s.cancel_reason}):null,(0,x.jsxs)(`div`,{className:`button-row`,children:[s.kind===`model_load`&&s.state===`failed`&&s.error?.code===`conflict`&&s.target.target_kind===`model`?(0,x.jsx)(k,{disabled:n||!e.catalog.some(t=>t.identity.id===ei(s)&&zr(e,t)),onClick:()=>o(ei(s)),children:N(t,`models.library.capacity`)}):null,s.kind===`download`&&!Fr(s)?(0,x.jsx)(k,{disabled:n||!Ir(e)||!s.cancellable||s.state===`cancelling`,onClick:()=>i(s),children:N(t,`models.library.cancel_download`)}):null,s.kind===`download`&&(s.state===`failed`||s.state===`cancelled`)?(0,x.jsx)(k,{disabled:n||r||!Lr(e,`download`),onClick:()=>a(s),children:N(t,`models.library.retry_download`)}):null]})]})},s.operation_id))})]})}function ei(e){return e.target.target_kind===`model`?e.target.model_id:``}function ti(e,t){let n=e.target;return n.target_kind===`download`?n.repo_id:n.target_kind===`model`?t.catalog.find(e=>e.identity.id===n.model_id)?.identity.display_name??n.model_id:e.kind}var ni=25;function ri({locale:e}){let t=pn(),n=mn(),[r,i]=(0,_.useState)({query:``,source:``,task:``,status:``,sort:`name`}),[a,o]=(0,_.useState)(0),[s,c]=(0,_.useState)(!1),l=(0,_.useRef)(!1),[u,d]=(0,_.useState)(null),[f,p]=(0,_.useState)(null),[m,h]=(0,_.useState)(null),[g,v]=(0,_.useState)(`copy`),y=t.catalog.find(e=>e.identity.id===t.selectedModelId),{profile:b}=zn(y?.identity.id??null),{profile:ee}=zn(f?.kind===`capacity`?f.entry.identity.id:null),te=Kr(t.catalog,r),ne=Math.max(1,Math.ceil(te.length/ni)),S=Math.min(a,ne-1),C=[...t.operations.values()].some(e=>e.kind===`download`&&!Fr(e))||[...t.pendingReconciliations.values()].some(e=>e.kind===`download`),w=[...t.operations.values()].some(e=>e.kind===`catalog_refresh`&&!Fr(e))||[...t.pendingReconciliations.values()].some(e=>e.kind===`catalog-refresh`),re=e=>{i({...r,...e}),o(0)},T=async(n,r)=>{if(l.current||!Ir(t))return!1;l.current=!0,c(!0),d(null);try{return await n(),!0}catch(t){return d(Gr(t,e)),r?.(t),!1}finally{l.current=!1,c(!1)}},ie=(r,i,a=b)=>{if(!zr(t,r)){d(N(e,`models.library.stale`));return}if(i&&!Ur(t,r.identity.id).some(e=>e.identity.id===i)){d(N(e,`models.library.stale`));return}T(()=>n.loadModel({action:`load`,...Object.keys(a).length?{load_profile:{...a}}:{},model_id:r.identity.id,expected_revision:r.identity.revision,idempotency_key:crypto.randomUUID(),...i?{eviction_target_id:i}:{}}),e=>{e instanceof St&&e.envelope?.error.code===`conflict`&&p({kind:`capacity`,entry:r,instance:t.serverInstanceId})})},E=e=>{y&&(e===`load`?ie(y):p({kind:e,entry:y,instance:t.serverInstanceId}))},ae=(r,i)=>{let a=f;if(a&&!l.current){if(a.instance!==t.serverInstanceId){d(N(e,`models.library.stale`)),p(null);return}if(a.kind===`cancel`){let e=t.operations.get(a.operation.operation_id);if(!e||!e.cancellable||Fr(e)||e.state===`cancelling`){p(null);return}T(()=>n.cancelOperation(e.operation_id))}else{let o=t.catalog.find(e=>e.identity.id===a.entry.identity.id);if(!o||o.identity.revision!==a.entry.identity.revision){d(N(e,`models.library.stale`)),p(null);return}if(a.kind===`capacity`){if(!Ur(t,o.identity.id).some(e=>e.identity.id===r&&e.identity.revision===i)){d(N(e,`models.library.stale`));return}ie(o,r,ee)}else a.kind===`unload`&&Br(t,o)?T(()=>n.unloadModel({action:`unload`,model_id:o.identity.id,expected_revision:o.identity.revision,idempotency_key:crypto.randomUUID()})):a.kind===`delete`&&Vr(t,o)?T(()=>n.removeModel({model_id:o.identity.id,expected_revision:o.identity.revision,idempotency_key:crypto.randomUUID()})):d(N(e,`models.library.stale`))}p(null)}},oe=e=>{e.target.target_kind===`download`&&h({repo:e.target.repo_id,revision:e.target.revision??``})},D=`mlxcel-server --webui --models-dir /path/to/models --no-models-autoload`,O=[{id:`name`,header:N(e,`models.library.name`),noResize:!0,render:r=>(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(k,{tone:`ghost`,"aria-label":N(e,`models.library.inspect`,{name:r.identity.display_name}),onClick:()=>n.selectModel(r.identity.id),children:r.identity.display_name}),r.identity.id===t.selectedModelId?(0,x.jsx)(`small`,{children:N(e,`models.library.selected`)}):null,(0,x.jsxs)(`p`,{children:[r.identity.source,` · `,r.metadata.quantization??N(e,`models.library.unknown`)]})]})},{id:`task`,header:N(e,`models.library.task`),noResize:!0,render:t=>t.capabilities.map(e=>e.task).filter((e,t,n)=>n.indexOf(e)===t).join(`, `)||N(e,`models.library.unknown`)},{id:`status`,header:N(e,`models.library.status`),noResize:!0,render:t=>(0,x.jsx)(he,{state:t.lifecycle.state,children:Nr(e,t.lifecycle.state)})},{id:`support`,header:N(e,`models.library.support`),noResize:!0,render:t=>(0,x.jsxs)(x.Fragment,{children:[N(e,t.metadata.support.architecturally_supported?`models.library.supported`:`models.library.unsupported`),(0,x.jsx)(`br`,{}),N(e,t.complete?`models.library.complete`:`models.library.incomplete`)]})}];return(0,x.jsxs)(`div`,{className:`screen-stack models-library`,"data-testid":`models-library`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:N(e,`routes.models.eyebrow`)}),(0,x.jsx)(`h1`,{tabIndex:-1,"data-dialog-focus-fallback":!0,"data-testid":P(`models.title`),children:N(e,`models.title`)}),(0,x.jsx)(`p`,{children:N(e,`models.library.subtitle`)})]}),Ir(t)?null:(0,x.jsx)(L,{title:N(e,`models.library.stale`),body:t.error?.message??N(e,`models.library.waiting`),action:(0,x.jsx)(k,{onClick:()=>{n.refresh()},children:N(e,`models.library.refresh`)}),testId:`connection-error-title`}),u?(0,x.jsx)(L,{title:N(e,`models.library.error`),body:u,action:(0,x.jsx)(k,{onClick:()=>{d(null),n.refresh()},children:N(e,`models.library.refresh`)}),testId:`models-action-error`}):null,t.bootstrap?.server.mode===`single_model`?(0,x.jsx)(L,{tone:`info`,title:N(e,`models.library.single`),body:`mlxcel-server --webui`,testId:`models-read-only`}):null,(0,x.jsxs)(`div`,{className:`button-row`,children:[(0,x.jsx)(k,{tone:`primary`,onClick:()=>h({repo:``,revision:``}),disabled:s||C||!Lr(t,`download`),"data-testid":`models-add`,children:N(e,`models.library.add`)}),(0,x.jsx)(k,{disabled:s||w||!Ir(t)||t.bootstrap?.server.mode===`single_model`,onClick:()=>{T(()=>n.refreshCatalog(crypto.randomUUID()))},"data-testid":`models-rescan`,children:N(e,`models.library.rescan`)}),(0,x.jsx)(k,{onClick:()=>{n.refresh()},disabled:s,children:N(e,`models.library.refresh`)})]}),Object.entries(t.bootstrap?.actions??{}).filter(([,e])=>e.state!==`enabled`).map(([e,t])=>(0,x.jsxs)(`p`,{children:[e,`: `,t.reason,` `,t.instructions]},e)),(0,x.jsx)(`p`,{"data-testid":`connection-authenticated-detail`,children:kr(e,t)}),(0,x.jsxs)(`details`,{open:t.catalog.length===0,children:[(0,x.jsx)(`summary`,{children:N(e,`models.library.roots`)}),(0,x.jsx)(`ul`,{children:t.bootstrap?.roots.map((t,n)=>(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`strong`,{children:t.display_name}),` · `,t.kind,t.error?(0,x.jsxs)(`p`,{role:`alert`,children:[N(e,`models.library.permission`),` `,t.error]}):null]},n))}),(0,x.jsx)(`p`,{children:N(e,`models.library.roots_help`)}),(0,x.jsx)(`pre`,{className:`models-wrap`,children:D}),(0,x.jsx)(k,{onClick:()=>{navigator.clipboard?.writeText(D).then(()=>v(`copied`)).catch(()=>v(`copy_failed`)),navigator.clipboard||v(`copy_failed`)},children:N(e,`models.library.${g}`)}),(0,x.jsx)(`span`,{role:`status`,children:g===`copy`?``:N(e,`models.library.${g}`)})]}),(0,x.jsxs)(`div`,{className:`models-toolbar`,children:[(0,x.jsx)(I,{label:N(e,`models.library.search`),value:r.query,onChange:e=>re({query:e}),testId:`models-search`}),(0,x.jsx)(F,{locale:e,label:N(e,`models.library.source`),value:r.source,onChange:e=>re({source:e}),options:[{value:``,label:N(e,`models.library.all`)},...[...new Set(t.catalog.map(e=>e.identity.source))].sort().map(e=>({value:e,label:e}))]}),(0,x.jsx)(F,{locale:e,label:N(e,`models.library.task`),value:r.task,onChange:e=>re({task:e}),options:[{value:``,label:N(e,`models.library.all`)},...[...new Set(t.catalog.flatMap(e=>e.capabilities.map(e=>e.task)))].sort().map(e=>({value:e,label:e}))]}),(0,x.jsx)(F,{locale:e,label:N(e,`models.library.status`),value:r.status,onChange:e=>re({status:e}),options:[{value:``,label:N(e,`models.library.all`)},...[`unloaded`,`loading`,`ready`,`draining`,`unloading`,`failed`].map(t=>({value:t,label:Nr(e,t)}))]}),(0,x.jsx)(F,{locale:e,label:N(e,`models.library.sort`),value:r.sort,onChange:e=>re({sort:e}),options:[`name`,`status`,`source`,`task`].map(t=>({value:t,label:N(e,`models.library.sort_${t}`)}))})]}),(0,x.jsxs)(`div`,{className:`models-layout`,children:[(0,x.jsxs)(`section`,{children:[(0,x.jsx)(ye,{columns:O,rows:te.slice(S*ni,(S+1)*ni),getRowKey:e=>e.identity.id,ariaLabel:N(e,`models.title`),testId:`models-table`,rowClassName:e=>e.identity.id===t.selectedModelId?`models-selected`:void 0,loading:t.catalogSequence===null,loadingState:(0,x.jsx)(`p`,{role:`status`,children:N(e,`models.library.waiting`)}),emptyState:(0,x.jsx)(_e,{title:N(e,t.catalog.length===0?`models.empty.title`:`models.library.filtered`),body:N(e,t.catalog.length===0?`models.empty.body`:`models.library.filtered_body`),testId:`models-empty`})}),(0,x.jsxs)(`nav`,{className:`models-pagination`,"aria-label":N(e,`models.title`),children:[(0,x.jsx)(k,{disabled:S===0,onClick:()=>o(S-1),children:N(e,`models.library.previous`)}),(0,x.jsx)(`span`,{role:`status`,children:N(e,`models.library.page`,{page:String(S+1),pages:String(ne),count:String(te.length)})}),(0,x.jsx)(k,{disabled:S>=ne-1,onClick:()=>o(S+1),children:N(e,`models.library.next`)})]})]}),y?(0,x.jsx)(Qr,{entry:y,profile:b,state:t,locale:e,busy:s,onAction:E,onChat:()=>{Hr(t,y)&&(n.selectModel(y.identity.id),window.location.hash=`chat`)}}):null]}),(0,x.jsx)($r,{state:t,locale:e,busy:s,downloadPending:C,onCancel:e=>p({kind:`cancel`,operation:e,instance:t.serverInstanceId}),onRetry:oe,onCapacity:e=>{let n=t.catalog.find(t=>t.identity.id===e);n&&zr(t,n)&&p({kind:`capacity`,entry:n,instance:t.serverInstanceId})}}),f?(0,x.jsx)(Yr,{value:f,state:t,locale:e,busy:s,onClose:()=>p(null),onConfirm:ae},f.kind):null,m?(0,x.jsx)(Xr,{locale:e,state:t,initial:m,error:u,busy:s||C||!Lr(t,`download`),onClose:()=>h(null),onDownload:(e,r)=>{Lr(t,`download`)&&!C&&T(()=>n.downloadModel({repo_id:e,revision:r||null,idempotency_key:crypto.randomUUID()})).then(e=>{e&&h(null)})}}):null]})}var ii=e=>e===`ko`?`ko-KR`:`en-US`;function ai(e,t){if(!Number.isFinite(e)||e<0)return t===`ko`?`알 수 없음`:`unknown`;let n=[`B`,`KiB`,`MiB`,`GiB`,`TiB`],r=e,i=0;for(;r>=1024&&i<n.length-1;)r/=1024,i+=1;let a=i===0?0:1;return`${new Intl.NumberFormat(ii(t),{maximumFractionDigits:a}).format(r)} ${n[i]}`}function oi(e,t){return e===null?t===`ko`?`아직 측정되지 않음`:`not yet measured`:`${new Intl.NumberFormat(ii(t),{maximumFractionDigits:1}).format(e)} tokens/s`}function si(e){let[t,n]=(0,_.useState)(!1),[r,i]=(0,_.useState)(`controls`),[a,o]=(0,_.useState)(`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`),[s,c]=(0,_.useState)(`ready`);return(0,x.jsxs)(`div`,{className:`screen-stack gallery-screen`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:N(e.locale,`gallery.issue`)}),(0,x.jsx)(`h1`,{"data-testid":P(`gallery.title`),children:N(e.locale,`gallery.title`)}),(0,x.jsx)(`p`,{"data-testid":P(`gallery.subtitle`),children:N(e.locale,`gallery.subtitle`)})]}),(0,x.jsx)(ve,{active:r,onChange:i,label:N(e.locale,`gallery.tab.sections`),tabs:[{id:`controls`,label:N(e.locale,`gallery.tab.controls`),panel:(0,x.jsx)(ci,{locale:e.locale,value:a,onValueChange:o,selectValue:s,onSelectChange:c,onOpenDialog:()=>n(!0)})},{id:`states`,label:N(e.locale,`gallery.tab.states`),panel:(0,x.jsx)(li,{locale:e.locale})},{id:`data`,label:N(e.locale,`gallery.tab.data`),panel:(0,x.jsx)(ui,{locale:e.locale})}]}),(0,x.jsxs)(we,{open:t,title:N(e.locale,`models.delete.confirm.title`),onClose:()=>n(!1),testId:`gallery-dialog`,closeLabel:N(e.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":P(`models.delete.confirm.body`),children:N(e.locale,`models.delete.confirm.body`,{model:a})}),(0,x.jsx)(I,{label:N(e.locale,`models.delete.confirm.token_label`),value:``,placeholder:N(e.locale,`gallery.delete_token`)}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(k,{onClick:()=>n(!1),children:N(e.locale,`common.cancel`)}),(0,x.jsx)(k,{tone:`danger`,children:N(e.locale,`common.delete`)})]})]})]})}function ci(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsxs)(`article`,{className:`surface-card`,children:[(0,x.jsx)(`h2`,{children:N(e.locale,`gallery.controls.title`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(k,{tone:`primary`,children:N(e.locale,`gallery.controls.primary`)}),(0,x.jsx)(k,{children:N(e.locale,`gallery.controls.secondary`)}),(0,x.jsx)(k,{tone:`danger`,children:N(e.locale,`gallery.controls.danger`)}),(0,x.jsx)(k,{busy:!0,children:N(e.locale,`gallery.controls.busy`)})]}),(0,x.jsx)(I,{label:N(e.locale,`gallery.field.repo`),value:e.value,onChange:e.onValueChange,error:N(e.locale,`gallery.field.repo_error`),hint:N(e.locale,`gallery.field.repo_hint`),testId:`gallery-field`}),(0,x.jsx)(F,{locale:e.locale,label:N(e.locale,`gallery.select.native`),value:e.selectValue,onChange:e.onSelectChange,options:[{value:`ready`,label:N(e.locale,`models.status.ready`)},{value:`unloaded`,label:N(e.locale,`models.status.unloaded`)}]})]}),(0,x.jsxs)(`article`,{className:`surface-card`,children:[(0,x.jsx)(`h2`,{children:N(e.locale,`gallery.overlays.title`)}),(0,x.jsx)(`p`,{children:N(e.locale,`gallery.long_cjk`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(Ee,{label:N(e.locale,`gallery.tooltip`),children:(0,x.jsx)(k,{children:N(e.locale,`gallery.hover_focus`)})}),(0,x.jsx)(k,{tone:`primary`,onClick:e.onOpenDialog,children:N(e.locale,`gallery.dialog.open`)})]})]})]})}function li(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsx)(Ae,{title:N(e.locale,`state.unauthorized.title`),body:N(e.locale,`state.unauthorized.body`),tokenLabel:N(e.locale,`login.token.label`),tokenHelp:N(e.locale,`login.token.help`),submitLabel:N(e.locale,`login.submit`),logoutLabel:N(e.locale,`login.logout`),onSubmit:()=>void 0,onLogout:()=>void 0,testId:`gallery-login`}),(0,x.jsx)(je,{title:N(e.locale,`state.schema_mismatch.title`),body:N(e.locale,`state.schema_mismatch.body`),actionLabel:N(e.locale,`common.reload`),onRecover:()=>void 0}),(0,x.jsx)(_e,{title:N(e.locale,`models.empty.title`),body:N(e.locale,`models.empty.body`),action:(0,x.jsx)(k,{tone:`primary`,children:N(e.locale,`common.add_model`)}),testId:`gallery-empty`}),(0,x.jsx)(L,{tone:`warning`,title:N(e.locale,`gallery.states.load_failed`),body:N(e.locale,`gallery.states.load_failed_body`),action:(0,x.jsx)(k,{children:N(e.locale,`common.retry`)})}),(0,x.jsxs)(`div`,{className:`gallery-progress-samples`,children:[(0,x.jsx)(ge,{label:N(e.locale,`gallery.download`),detail:N(e.locale,`activity.progress.indeterminate`,{bytes:ai(67108864,e.locale)})}),(0,x.jsx)(ge,{label:N(e.locale,`gallery.progress.measured`),value:37})]}),(0,x.jsxs)(ke,{label:N(e.locale,`gallery.lifecycle.samples`),children:[(0,x.jsx)(`li`,{children:(0,x.jsx)(he,{state:`loading`,children:N(e.locale,`models.status.loading`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(he,{state:`draining`,children:N(e.locale,`models.status.draining`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(he,{state:`unloading`,children:N(e.locale,`models.status.unloading`)})})]})]})}function ui(e){return(0,x.jsxs)(`div`,{className:`gallery-grid dense-gallery`,children:[(0,x.jsxs)(`article`,{className:`surface-card table-card`,tabIndex:0,"aria-label":N(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`h2`,{children:N(e.locale,`gallery.data.title`)}),(0,x.jsxs)(Oe,{caption:N(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`thead`,{children:(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`th`,{children:N(e.locale,`gallery.data.name`)}),(0,x.jsx)(`th`,{children:N(e.locale,`gallery.data.status`)}),(0,x.jsx)(`th`,{children:N(e.locale,`gallery.data.rate`)})]})}),(0,x.jsxs)(`tbody`,{children:[(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:N(e.locale,`models.long_name`),"aria-label":N(e.locale,`models.long_name`),children:N(e.locale,`models.long_name`)})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(he,{state:`ready`,children:N(e.locale,`models.status.ready`)})}),(0,x.jsx)(`td`,{children:oi(39.4,e.locale)})]}),(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:`granite-4.0-h-tiny-4bit`,"aria-label":`granite-4.0-h-tiny-4bit`,children:`granite-4.0-h-tiny-4bit`})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(he,{state:`unloaded`,children:N(e.locale,`models.status.unloaded`)})}),(0,x.jsx)(`td`,{children:oi(null,e.locale)})]})]})]})]}),(0,x.jsxs)(De,{title:N(e.locale,`gallery.data.inspector`),children:[(0,x.jsx)(`p`,{className:`sample-label`,children:N(e.locale,`gallery.sample.label`)}),(0,x.jsx)(`p`,{children:N(e.locale,`models.unsupported.reason`)}),(0,x.jsx)(he,{state:`failed`,children:N(e.locale,`models.status.failed`)}),(0,x.jsx)(`hr`,{}),(0,x.jsx)(`p`,{children:N(e.locale,`gallery.sample.reasoning_body`)}),(0,x.jsx)(`code`,{children:N(e.locale,`gallery.sample.tool_preview`)})]})]})}var di=[`models`,`chat`,`activity`,`settings`,`gallery`],fi=[`models`,`chat`,`activity`,`settings`];function pi(){let e=window.location.hash.slice(1).replace(/^\//,``);return di.includes(e)?e:`models`}function mi(){let e=pn(),t=mn(),[n,r]=(0,_.useState)(pi),[i,a]=(0,_.useState)(tr),[o,s]=(0,_.useState)(null),[c,l]=(0,_.useState)(null),u=(0,_.useRef)(0);(0,_.useEffect)(()=>()=>{u.current+=1},[]),(0,_.useEffect)(()=>{let e=()=>r(pi());return window.addEventListener(`hashchange`,e),()=>window.removeEventListener(`hashchange`,e)},[]),(0,_.useEffect)(()=>{ir(document.documentElement,i),nr(i)},[i]),(0,_.useEffect)(()=>{let e=()=>{document.documentElement.dataset.documentHidden=String(document.hidden)};return e(),document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e)},[]),(0,_.useEffect)(()=>{e.auth.status===`authenticated`&&l(null)},[e.auth.status]);let d=e=>{r(e),window.history.replaceState(null,``,`#${e}`)},f=e=>{let n=u.current+1;u.current=n,l(null),t.login(e).catch(e=>{u.current===n&&(e instanceof DOMException&&e.name===`AbortError`||l(xr(e)))})},p=()=>{u.current+=1,l(null),t.logout()},m=hi(n,i,a,{snapshot:e,authFailure:c,login:f,logout:p,retry:()=>{l(null),t.refresh()},recoverSchema:()=>{p(),window.location.reload()}});return(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(Wn,{locale:i.locale,route:n,onRouteChange:d,onCommand:()=>s(`command`),onHelp:()=>s(`help`),selectedModel:Sr(i.locale,e),connectionLabel:Cr(i.locale,e),connectionState:e.connection,sessionAction:e.auth.tokenPresent?(0,x.jsx)(A,{label:N(i.locale,`toolbar.logout`),icon:`key`,onClick:p,"data-testid":P(`toolbar.logout`)}):null,inspector:null,children:m}),(0,x.jsx)(xi,{open:o===`command`,locale:i.locale,onClose:()=>s(null),onNavigate:e=>{d(e),s(null)}}),(0,x.jsxs)(we,{open:o===`help`,title:N(i.locale,`help.title`),onClose:()=>s(null),testId:`help-dialog`,closeLabel:N(i.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":P(`help.body`),children:N(i.locale,`help.body`)}),(0,x.jsxs)(`ul`,{className:`shortcut-list`,children:[(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`⌘/Ctrl`}),` + `,(0,x.jsx)(`kbd`,{children:`K`}),` `,N(i.locale,`command.search`)]}),(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`Esc`}),` `,N(i.locale,`common.cancel`)]}),(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`[`}),` / `,(0,x.jsx)(`kbd`,{children:`]`}),` `,N(i.locale,`nav.models`)]})]})]})]})}function hi(e,t,n,r){return e===`models`?(0,x.jsx)(gi,{locale:t.locale,context:r}):e===`chat`?(0,x.jsx)(_i,{locale:t.locale,context:r}):e===`activity`?(0,x.jsx)(vi,{locale:t.locale,context:r}):e===`settings`?(0,x.jsx)(yi,{appearance:t,setAppearance:n}):(0,x.jsx)(si,{locale:t.locale})}function gi(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,x.jsx)(ri,{locale:e.locale}):(0,x.jsx)(Tr,{locale:e.locale,eyebrow:N(e.locale,`routes.models.eyebrow`),title:N(e.locale,`models.title`),titleTestId:P(`models.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function _i(e){return(0,x.jsx)(Tr,{locale:e.locale,eyebrow:N(e.locale,`routes.chat.eyebrow`),title:N(e.locale,`chat.title`),titleTestId:P(`chat.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function vi(e){return e.context.snapshot.auth.status===`authenticated`&&![`schema-mismatch`,`forbidden`,`unauthorized`].includes(e.context.snapshot.connection)?(0,x.jsx)(br,{locale:e.locale}):(0,x.jsx)(Tr,{locale:e.locale,eyebrow:N(e.locale,`routes.activity.eyebrow`),title:N(e.locale,`activity.title`),titleTestId:P(`activity.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function yi(e){let t=t=>e.setAppearance({...e.appearance,...t});return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:N(e.appearance.locale,`routes.settings.eyebrow`)}),(0,x.jsx)(`h1`,{tabIndex:-1,"data-dialog-focus-fallback":!0,"data-testid":P(`settings.title`),children:N(e.appearance.locale,`settings.title`)}),(0,x.jsx)(`p`,{"data-testid":P(`settings.appearance`),children:N(e.appearance.locale,`settings.browser_only`)})]}),(0,x.jsxs)(`div`,{className:`settings-grid`,children:[(0,x.jsx)(F,{locale:e.appearance.locale,label:N(e.appearance.locale,`settings.theme`),value:e.appearance.theme,onChange:e=>t({theme:e}),options:[{value:`system`,label:N(e.appearance.locale,`settings.theme.system`)},{value:`light`,label:N(e.appearance.locale,`settings.theme.light`)},{value:`dark`,label:N(e.appearance.locale,`settings.theme.dark`)}],testId:P(`settings.theme`)}),(0,x.jsx)(F,{locale:e.appearance.locale,label:N(e.appearance.locale,`settings.material`),value:e.appearance.material,onChange:e=>t({material:e}),options:[{value:`glass`,label:N(e.appearance.locale,`settings.material.glass`)},{value:`tinted`,label:N(e.appearance.locale,`settings.material.tinted`)},{value:`opaque`,label:N(e.appearance.locale,`settings.material.opaque`)}],testId:P(`settings.material`)}),(0,x.jsx)(F,{locale:e.appearance.locale,label:N(e.appearance.locale,`settings.locale`),value:e.appearance.locale,onChange:e=>t({locale:e}),options:[{value:`en`,label:N(e.appearance.locale,`settings.locale.en`)},{value:`ko`,label:N(e.appearance.locale,`settings.locale.ko`)}],testId:P(`settings.locale`)}),(0,x.jsx)(F,{locale:e.appearance.locale,label:N(e.appearance.locale,`settings.high_contrast`),value:e.appearance.highContrast,onChange:e=>t({highContrast:e}),options:[{value:`system`,label:N(e.appearance.locale,`settings.high_contrast.system`)},{value:`on`,label:N(e.appearance.locale,`settings.high_contrast.on`)},{value:`off`,label:N(e.appearance.locale,`settings.high_contrast.off`)}],testId:P(`settings.high_contrast`)}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsxs)(`span`,{children:[N(e.appearance.locale,`settings.glass_intensity`),`: `,e.appearance.glassIntensity]}),(0,x.jsx)(`input`,{type:`range`,min:`0`,max:`100`,value:e.appearance.glassIntensity,onChange:e=>t({glassIntensity:Number(e.currentTarget.value)}),"data-testid":P(`settings.glass_intensity`)})]}),(0,x.jsx)(bi,{label:N(e.appearance.locale,`settings.reduce_motion`),checked:e.appearance.reduceMotion,onChange:e=>t({reduceMotion:e}),testId:P(`settings.reduce_motion`)}),(0,x.jsx)(bi,{label:N(e.appearance.locale,`settings.reduce_transparency`),checked:e.appearance.reduceTransparency,onChange:e=>t({reduceTransparency:e}),testId:P(`settings.reduce_transparency`)})]}),(0,x.jsx)(L,{tone:`info`,title:N(e.appearance.locale,`settings.browser_only`),body:N(e.appearance.locale,`settings.browser_only.body`),testId:P(`settings.browser_only`)}),(0,x.jsx)(Hn,{locale:e.appearance.locale})]})}function bi(e){return(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:e.checked,onChange:t=>e.onChange(t.currentTarget.checked),"data-testid":e.testId}),(0,x.jsx)(`span`,{children:e.label})]})}function xi(e){let[t,n]=(0,_.useState)(``),r=(e.locale&&window.location.hash.includes(`gallery`)?di:fi).filter(e=>e.includes(t.toLowerCase()));return(0,x.jsxs)(we,{open:e.open,title:N(e.locale,`command.title`),onClose:e.onClose,testId:`command-dialog`,closeLabel:N(e.locale,`common.close`),children:[(0,x.jsx)(I,{label:N(e.locale,`command.search`),value:t,onChange:n,testId:P(`command.search`)}),r.length===0?(0,x.jsx)(`p`,{"data-testid":P(`command.no_results`),children:N(e.locale,`command.no_results`)}):(0,x.jsx)(`div`,{className:`command-list`,children:r.map(t=>(0,x.jsx)(k,{onClick:()=>e.onNavigate(t),children:N(e.locale,t===`models`?`nav.models`:t===`chat`?`nav.chat`:t===`activity`?`nav.activity`:t===`settings`?`nav.settings`:`nav.gallery`)},t))})]})}var Si=document.getElementById(`root`);if(Si===null)throw Error(`Missing #root element for mlxcel WebUI.`);(0,v.createRoot)(Si).render((0,x.jsx)(_.StrictMode,{children:(0,x.jsx)(fn,{children:(0,x.jsx)(mi,{})})}));export{c as i,b as n,u as r,dr as t};