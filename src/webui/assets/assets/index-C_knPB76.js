var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},c=(n,r,o)=>(o=n==null?{}:e(i(n)),s(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n)),l=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.portal`),r=Symbol.for(`react.fragment`),i=Symbol.for(`react.strict_mode`),a=Symbol.for(`react.profiler`),o=Symbol.for(`react.consumer`),s=Symbol.for(`react.context`),c=Symbol.for(`react.forward_ref`),l=Symbol.for(`react.suspense`),u=Symbol.for(`react.memo`),d=Symbol.for(`react.lazy`),f=Symbol.for(`react.activity`),p=Symbol.for(`react.view_transition`),m=Symbol.iterator;function h(e){return typeof e!=`object`||!e?null:(e=m&&e[m]||e[`@@iterator`],typeof e==`function`?e:null)}var g={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},_=Object.assign,v={};function y(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}y.prototype.isReactComponent={},y.prototype.setState=function(e,t){if(typeof e!=`object`&&typeof e!=`function`&&e!=null)throw Error(`takes an object of state variables to update or a function which returns an object of state variables.`);this.updater.enqueueSetState(this,e,t,`setState`)},y.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,`forceUpdate`)};function b(){}b.prototype=y.prototype;function x(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}var S=x.prototype=new b;S.constructor=x,_(S,y.prototype),S.isPureReactComponent=!0;var C=Array.isArray;function w(){}var T={H:null,A:null,T:null,S:null},E=Object.prototype.hasOwnProperty;function D(e,n,r){var i=r.ref;return{$$typeof:t,type:e,key:n,ref:i===void 0?null:i,props:r}}function O(e,t){return D(e.type,t,e.props)}function k(e){return typeof e==`object`&&!!e&&e.$$typeof===t}function A(e){var t={"=":`=0`,":":`=2`};return`$`+e.replace(/[=:]/g,function(e){return t[e]})}var j=/\/+/g;function ee(e,t){return typeof e==`object`&&e&&e.key!=null?A(``+e.key):t.toString(36)}function te(e){switch(e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason;default:switch(typeof e.status==`string`?e.then(w,w):(e.status=`pending`,e.then(function(t){e.status===`pending`&&(e.status=`fulfilled`,e.value=t)},function(t){e.status===`pending`&&(e.status=`rejected`,e.reason=t)})),e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason}}throw e}function ne(e,r,i,a,o){var s=typeof e;(s===`undefined`||s===`boolean`)&&(e=null);var c=!1;if(e===null)c=!0;else switch(s){case`bigint`:case`string`:case`number`:c=!0;break;case`object`:switch(e.$$typeof){case t:case n:c=!0;break;case d:return c=e._init,ne(c(e._payload),r,i,a,o)}}if(c)return o=o(e),c=a===``?`.`+ee(e,0):a,C(o)?(i=``,c!=null&&(i=c.replace(j,`$&/`)+`/`),ne(o,r,i,``,function(e){return e})):o!=null&&(k(o)&&(o=O(o,i+(o.key==null||e&&e.key===o.key?``:(``+o.key).replace(j,`$&/`)+`/`)+c)),r.push(o)),1;c=0;var l=a===``?`.`:a+`:`;if(C(e))for(var u=0;u<e.length;u++)a=e[u],s=l+ee(a,u),c+=ne(a,r,i,s,o);else if(u=h(e),typeof u==`function`)for(e=u.call(e),u=0;!(a=e.next()).done;)a=a.value,s=l+ee(a,u++),c+=ne(a,r,i,s,o);else if(s===`object`){if(typeof e.then==`function`)return ne(te(e),r,i,a,o);throw r=String(e),Error(`Objects are not valid as a React child (found: `+(r===`[object Object]`?`object with keys {`+Object.keys(e).join(`, `)+`}`:r)+`). If you meant to render a collection of children, use an array instead.`)}return c}function M(e,t,n){if(e==null)return e;var r=[],i=0;return ne(e,r,``,``,function(e){return t.call(n,e,i++)}),r}function re(e){if(e._status===-1){var t=e._result,n=t();n.then(function(t){(e._status===0||e._status===-1)&&(e._status=1,e._result=t,n.status===void 0&&(n.status=`fulfilled`,n.value=t))},function(t){(e._status===0||e._status===-1)&&(e._status=2,e._result=t,n.status===void 0&&(n.status=`rejected`,n.reason=t))}),e._status===-1&&(e._status=0,e._result=n)}if(e._status===1)return e._result.default;throw e._result}var ie=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)};function N(e){var t=T.T,n={};n.types=t===null?null:t.types,T.T=n;try{var r=e(),i=T.S;i!==null&&i(n,r),typeof r==`object`&&r&&typeof r.then==`function`&&r.then(w,ie)}catch(e){ie(e)}finally{t!==null&&n.types!==null&&(t.types=n.types),T.T=t}}function ae(e){var t=T.T;if(t!==null){var n=t.types;n===null?t.types=[e]:n.indexOf(e)===-1&&n.push(e)}else N(ae.bind(null,e))}var oe={map:M,forEach:function(e,t,n){M(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return M(e,function(){t++}),t},toArray:function(e){return M(e,function(e){return e})||[]},only:function(e){if(!k(e))throw Error(`React.Children.only expected to receive a single React element child.`);return e}};e.Activity=f,e.Children=oe,e.Component=y,e.Fragment=r,e.Profiler=a,e.PureComponent=x,e.StrictMode=i,e.Suspense=l,e.ViewTransition=p,e.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=T,e.__COMPILER_RUNTIME={__proto__:null,c:function(e){return T.H.useMemoCache(e)}},e.addTransitionType=ae,e.cache=function(e){return function(){return e.apply(null,arguments)}},e.cacheSignal=function(){return null},e.cloneElement=function(e,t,n){if(e==null)throw Error(`The argument must be a React element, but you passed `+e+`.`);var r=_({},e.props),i=e.key;if(t!=null)for(a in t.key!==void 0&&(i=``+t.key),t)!E.call(t,a)||a===`key`||a===`__self`||a===`__source`||a===`ref`&&t.ref===void 0||(r[a]=t[a]);var a=arguments.length-2;if(a===1)r.children=n;else if(1<a){for(var o=Array(a),s=0;s<a;s++)o[s]=arguments[s+2];r.children=o}return D(e.type,i,r)},e.createContext=function(e){return e={$$typeof:s,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null},e.Provider=e,e.Consumer={$$typeof:o,_context:e},e},e.createElement=function(e,t,n){var r,i={},a=null;if(t!=null)for(r in t.key!==void 0&&(a=``+t.key),t)E.call(t,r)&&r!==`key`&&r!==`__self`&&r!==`__source`&&(i[r]=t[r]);var o=arguments.length-2;if(o===1)i.children=n;else if(1<o){for(var s=Array(o),c=0;c<o;c++)s[c]=arguments[c+2];i.children=s}if(e&&e.defaultProps)for(r in o=e.defaultProps,o)i[r]===void 0&&(i[r]=o[r]);return D(e,a,i)},e.createRef=function(){return{current:null}},e.forwardRef=function(e){return{$$typeof:c,render:e}},e.isValidElement=k,e.lazy=function(e){return{$$typeof:d,_payload:{_status:-1,_result:e},_init:re}},e.memo=function(e,t){return{$$typeof:u,type:e,compare:t===void 0?null:t}},e.startTransition=N,e.unstable_useCacheRefresh=function(){return T.H.useCacheRefresh()},e.use=function(e){return T.H.use(e)},e.useActionState=function(e,t,n){return T.H.useActionState(e,t,n)},e.useCallback=function(e,t){return T.H.useCallback(e,t)},e.useContext=function(e){return T.H.useContext(e)},e.useDebugValue=function(){},e.useDeferredValue=function(e,t){return T.H.useDeferredValue(e,t)},e.useEffect=function(e,t){return T.H.useEffect(e,t)},e.useEffectEvent=function(e){return T.H.useEffectEvent(e)},e.useId=function(){return T.H.useId()},e.useImperativeHandle=function(e,t,n){return T.H.useImperativeHandle(e,t,n)},e.useInsertionEffect=function(e,t){return T.H.useInsertionEffect(e,t)},e.useLayoutEffect=function(e,t){return T.H.useLayoutEffect(e,t)},e.useMemo=function(e,t){return T.H.useMemo(e,t)},e.useOptimistic=function(e,t){return T.H.useOptimistic(e,t)},e.useReducer=function(e,t,n){return T.H.useReducer(e,t,n)},e.useRef=function(e){return T.H.useRef(e)},e.useState=function(e){return T.H.useState(e)},e.useSyncExternalStore=function(e,t,n){return T.H.useSyncExternalStore(e,t,n)},e.useTransition=function(){return T.H.useTransition()},e.version=`19.3.0`})),u=o(((e,t)=>{t.exports=l()})),d=o((e=>{function t(e,t){var n=e.length;e.push(t);a:for(;0<n;){var r=n-1>>>1,a=e[r];if(0<i(a,t))e[r]=t,e[n]=a,n=r;else break a}}function n(e){return e.length===0?null:e[0]}function r(e){if(e.length===0)return null;var t=e[0],n=e.pop();if(n!==t){e[0]=n;a:for(var r=0,a=e.length,o=a>>>1;r<o;){var s=2*(r+1)-1,c=e[s],l=s+1,u=e[l];if(0>i(c,n))l<a&&0>i(u,c)?(e[r]=u,e[l]=n,r=l):(e[r]=c,e[s]=n,r=s);else if(l<a&&0>i(u,n))e[r]=u,e[l]=n,r=l;else break a}}return t}function i(e,t){var n=e.sortIndex-t.sortIndex;return n===0?e.id-t.id:n}if(e.unstable_now=void 0,typeof performance==`object`&&typeof performance.now==`function`){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var c=[],l=[],u=1,d=null,f=3,p=!1,m=!1,h=!1,g=!1,_=typeof setTimeout==`function`?setTimeout:null,v=typeof clearTimeout==`function`?clearTimeout:null,y=typeof setImmediate<`u`?setImmediate:null;function b(e){for(var i=n(l);i!==null;){if(i.callback===null)r(l);else if(i.startTime<=e)r(l),i.sortIndex=i.expirationTime,t(c,i);else break;i=n(l)}}function x(e){if(h=!1,b(e),!m){if(n(c)!==null)m=!0,S||(S=!0,O());else{var t=n(l);t!==null&&j(x,t.startTime-e)}}}var S=!1,C=-1,w=5,T=-1;function E(){return g?!0:!(e.unstable_now()-T<w)}function D(){if(g=!1,S){var t=e.unstable_now();T=t;var i=!0;try{a:{m=!1,h&&(h=!1,v(C),C=-1),p=!0;var a=f;try{b:{for(b(t),d=n(c);d!==null&&!(d.expirationTime>t&&E());){var o=d.callback;if(typeof o==`function`){d.callback=null,f=d.priorityLevel;var s=o(d.expirationTime<=t);if(t=e.unstable_now(),typeof s==`function`){d.callback=s,b(t),i=!0;break b}d===n(c)&&r(c),b(t)}else r(c);d=n(c)}if(d!==null)i=!0;else{var u=n(l);u!==null&&j(x,u.startTime-t),i=!1}}break a}finally{d=null,f=a,p=!1}i=void 0}}finally{i?O():S=!1}}}var O;if(typeof y==`function`)O=function(){y(D)};else if(typeof MessageChannel<`u`){var k=new MessageChannel,A=k.port2;k.port1.onmessage=D,O=function(){A.postMessage(null)}}else O=function(){_(D,0)};function j(t,n){C=_(function(){t(e.unstable_now())},n)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(e){e.callback=null},e.unstable_forceFrameRate=function(e){0>e||125<e?console.error(`forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported`):w=0<e?Math.floor(1e3/e):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_next=function(e){switch(f){case 1:case 2:case 3:var t=3;break;default:t=f}var n=f;f=t;try{return e()}finally{f=n}},e.unstable_requestPaint=function(){g=!0},e.unstable_runWithPriority=function(e,t){switch(e){case 1:case 2:case 3:case 4:case 5:break;default:e=3}var n=f;f=e;try{return t()}finally{f=n}},e.unstable_scheduleCallback=function(r,i,a){var o=e.unstable_now();switch(typeof a==`object`&&a?(a=a.delay,a=typeof a==`number`&&0<a?o+a:o):a=o,r){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=a+s,r={id:u++,callback:i,priorityLevel:r,startTime:a,expirationTime:s,sortIndex:-1},a>o?(r.sortIndex=a,t(l,r),n(c)===null&&r===n(l)&&(h?(v(C),C=-1):h=!0,j(x,a-o))):(r.sortIndex=s,t(c,r),m||p||(m=!0,S||(S=!0,O()))),r},e.unstable_shouldYield=E,e.unstable_wrapCallback=function(e){var t=f;return function(){var n=f;f=t;try{return e.apply(this,arguments)}finally{f=n}}}})),f=o(((e,t)=>{t.exports=d()})),p=o((e=>{var t=u();function n(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function r(){}var i={d:{f:r,r:function(){throw Error(n(522))},D:r,C:r,L:r,m:r,X:r,S:r,M:r},p:0,findDOMNode:null},a=Symbol.for(`react.portal`),o=Symbol.for(`react.recoverable`),s=Symbol.for(`react.optimistic_key`);function c(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:a,key:r==null?null:r===s?s:``+r,children:e,containerInfo:t,implementation:n}}var l=t.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function d(e,t){if(e===`font`)return``;if(typeof t==`string`)return t===`use-credentials`?t:``}e.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=i,e.browser=function(e){return{$$typeof:o,_reason:e}},e.createPortal=function(e,t){var r=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)throw Error(n(299));return c(e,t,null,r)},e.flushSync=function(e){var t=l.T,n=i.p;try{if(l.T=null,i.p=2,e)return e()}finally{l.T=t,i.p=n,i.d.f()}},e.preconnect=function(e,t){typeof e==`string`&&(t?(t=t.crossOrigin,t=typeof t==`string`?t===`use-credentials`?t:``:void 0):t=null,i.d.C(e,t))},e.prefetchDNS=function(e){typeof e==`string`&&i.d.D(e)},e.preinit=function(e,t){if(typeof e==`string`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin),a=typeof t.integrity==`string`?t.integrity:void 0,o=typeof t.fetchPriority==`string`?t.fetchPriority:void 0;n===`style`?i.d.S(e,typeof t.precedence==`string`?t.precedence:void 0,{crossOrigin:r,integrity:a,fetchPriority:o}):n===`script`&&i.d.X(e,{crossOrigin:r,integrity:a,fetchPriority:o,nonce:typeof t.nonce==`string`?t.nonce:void 0})}},e.preinitModule=function(e,t){if(typeof e==`string`){if(typeof t==`object`&&t){if(t.as==null||t.as===`script`){var n=d(t.as,t.crossOrigin);i.d.M(e,{crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}}else t??i.d.M(e)}},e.preload=function(e,t){if(typeof e==`string`&&typeof t==`object`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin);i.d.L(e,n,{crossOrigin:r,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,type:typeof t.type==`string`?t.type:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0,referrerPolicy:typeof t.referrerPolicy==`string`?t.referrerPolicy:void 0,imageSrcSet:typeof t.imageSrcSet==`string`?t.imageSrcSet:void 0,imageSizes:typeof t.imageSizes==`string`?t.imageSizes:void 0,media:typeof t.media==`string`?t.media:void 0})}},e.preloadModule=function(e,t){if(typeof e==`string`){if(t){var n=d(t.as,t.crossOrigin);i.d.m(e,{as:typeof t.as==`string`&&t.as!==`script`?t.as:void 0,crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}else i.d.m(e)}},e.requestFormReset=function(e){i.d.r(e)},e.unstable_batchedUpdates=function(e,t){return e(t)},e.useFormState=function(e,t,n){return l.H.useFormState(e,t,n)},e.useFormStatus=function(){return l.H.useHostTransitionStatus()},e.version=`19.3.0`})),m=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=p()})),h=o((e=>{var t=f(),n=u(),r=m();function i(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function a(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function o(e){for(var t=e,n=t;n&&!n.alternate;)t=n,t.flags&4098&&(e=t.return),n=t.return;for(;t.return;)t=t.return;return t.tag===3?e:null}function s(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function c(e){if(e.tag===31){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function l(e){if(o(e)!==e)throw Error(i(188))}function d(e){var t=e.alternate;if(!t){if(t=o(e),t===null)throw Error(i(188));return t===e?e:null}for(var n=e,r=t;;){var a=n.return;if(a===null)break;var s=a.alternate;if(s===null){if(r=a.return,r!==null){n=r;continue}break}if(a.child===s.child){for(s=a.child;s;){if(s===n)return l(a),e;if(s===r)return l(a),t;s=s.sibling}throw Error(i(188))}if(n.return!==r.return)n=a,r=s;else{for(var c=!1,u=a.child;u;){if(u===n){c=!0,n=a,r=s;break}if(u===r){c=!0,r=a,n=s;break}u=u.sibling}if(!c){for(u=s.child;u;){if(u===n){c=!0,n=s,r=a;break}if(u===r){c=!0,r=s,n=a;break}u=u.sibling}if(!c)throw Error(i(189))}}if(n.alternate!==r)throw Error(i(190))}if(n.tag!==3)throw Error(i(188));return n.stateNode.current===n?e:t}function p(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e;for(e=e.child;e!==null;){if(t=p(e),t!==null)return t;e=e.sibling}return null}function h(e,t,n,r,i,a){for(;e!==null;){if((e.tag===5||e.tag===27||e.tag===6)&&n(e,r,i,a)||(e.tag!==22||e.memoizedState===null)&&(t||e.tag!==5&&e.tag!==27)&&h(e.child,t,n,r,i,a))return!0;e=e.sibling}return!1}function g(e){for(e=e.return;e!==null;){if(e.tag===3||e.tag===5||e.tag===27)return e;e=e.return}return null}function _(e){var t=!1;for(e=e.return;e!==null&&(e.tag===4&&(t=!0),e.tag!==3&&e.tag!==5&&e.tag!==27);)e=e.return;return t}function v(e){var t=[null,null],n=g(e);return n===null||y(t,e,n.child,{foundSelf:!1}),t}function y(e,t,n,r){for(;n!==null;){if(n===t)r.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(r.foundSelf)return e[1]=n,!0;e[0]=n}else if((n.tag!==22||n.memoizedState===null)&&y(e,t,n.child,r))return!0;n=n.sibling}return!1}function b(e){switch(e.tag){case 5:case 27:case 6:return e.stateNode;case 3:return e.stateNode.containerInfo;default:throw Error(i(559))}}var x=null,S=null;function C(e,t,n){return e===n||e===t&&(x=e,!0)}function w(e,t,n){return e===n?(S=e,!1):e===t&&(S!==null&&(x=e),!0)}function T(e){if(e===null)return null;do e=e===null?null:e.return;while(e&&e.tag!==5&&e.tag!==27&&e.tag!==3);return e||null}function E(e,t,n){for(var r=0,i=e;i;i=n(i))r++;i=0;for(var a=t;a;a=n(a))i++;for(;0<r-i;)e=n(e),r--;for(;0<i-r;)t=n(t),i--;for(;r--;){if(e===t||t!==null&&e===t.alternate)return e;e=n(e),t=n(t)}return null}var D=Object.assign,O=Symbol.for(`react.element`),k=Symbol.for(`react.transitional.element`),A=Symbol.for(`react.portal`),j=Symbol.for(`react.fragment`),ee=Symbol.for(`react.strict_mode`),te=Symbol.for(`react.profiler`),ne=Symbol.for(`react.consumer`),M=Symbol.for(`react.context`),re=Symbol.for(`react.forward_ref`),ie=Symbol.for(`react.suspense`),N=Symbol.for(`react.suspense_list`),ae=Symbol.for(`react.memo`),oe=Symbol.for(`react.lazy`),se=Symbol.for(`react.activity`),ce=Symbol.for(`react.legacy_hidden`),P=Symbol.for(`react.memo_cache_sentinel`),F=Symbol.for(`react.view_transition`),le=Symbol.for(`react.recoverable`),ue=Symbol.iterator;function de(e){return typeof e!=`object`||!e?null:(e=ue&&e[ue]||e[`@@iterator`],typeof e==`function`?e:null)}var I=Symbol.for(`react.client.reference`);function fe(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===I?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case j:return`Fragment`;case te:return`Profiler`;case ee:return`StrictMode`;case ie:return`Suspense`;case N:return`SuspenseList`;case se:return`Activity`;case F:return`ViewTransition`}if(typeof e==`object`)switch(e.$$typeof){case A:return`Portal`;case M:return e.displayName||`Context`;case ne:return(e._context.displayName||`Context`)+`.Consumer`;case re:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case ae:return t=e.displayName||null,t===null?fe(e.type)||`Memo`:t;case oe:t=e._payload,e=e._init;try{return fe(e(t))}catch{}}return null}var pe=Array.isArray,L=n.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,R=r.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,me={pending:!1,data:null,method:null,action:null},he=[],ge=-1;function _e(e){return{current:e}}function ve(e){0>ge||(e.current=he[ge],he[ge]=null,ge--)}function z(e,t){ge++,he[ge]=e.current,e.current=t}var ye=_e(null),be=_e(null),xe=_e(null),Se=_e(null);function Ce(e,t){switch(z(xe,t),z(be,e),z(ye,null),t.nodeType){case 9:case 11:e=(e=t.documentElement)&&(e=e.namespaceURI)?up(e):0;break;default:if(e=t.tagName,t=t.namespaceURI)t=up(t),e=dp(t,e);else switch(e){case`svg`:e=1;break;case`math`:e=2;break;default:e=0}}ve(ye),z(ye,e)}function we(){ve(ye),ve(be),ve(xe)}function Te(e){var t=e.memoizedState;t!==null&&(sh._currentValue=t.memoizedState,z(Se,e)),t=ye.current;var n=dp(t,e.type);t!==n&&(z(be,e),z(ye,n))}function Ee(e){be.current===e&&(ve(ye),ve(be)),Se.current===e&&(ve(Se),sh._currentValue=me)}var De,Oe;function ke(e){if(De===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);De=t&&t[1]||``,Oe=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+De+e+Oe}var Ae=!1;function je(e,t){if(!e||Ae)return``;Ae=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}n=!1;try{var i=Object.getOwnPropertyDescriptor(e.prototype,`props`);Object.defineProperty(e.prototype,"props",{configurable:!0,set:function(){throw Error()}}),n=!0,new e}finally{n&&(i===void 0?delete e.prototype.props:Object.defineProperty(e.prototype,"props",i))}}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,"name",{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Ae=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?ke(n):``}function B(e,t){switch(e.tag){case 26:case 27:case 5:return ke(e.type);case 16:return ke(`Lazy`);case 13:return e.child!==t&&t!==null?ke(`Suspense Fallback`):ke(`Suspense`);case 19:return ke(`SuspenseList`);case 0:case 15:return je(e.type,!1);case 11:return je(e.type.render,!1);case 1:return je(e.type,!0);case 31:return ke(`Activity`);case 30:return ke(`ViewTransition`);default:return``}}function V(e){try{var t=``,n=null;do t+=B(e,n),n=e,e=e.return;while(e);return t}catch(e){return`
Error generating stack: `+e.message+`
`+e.stack}}var Me=Object.prototype.hasOwnProperty,Ne=t.unstable_scheduleCallback,Pe=t.unstable_cancelCallback,Fe=t.unstable_shouldYield,Ie=t.unstable_requestPaint,Le=t.unstable_now,Re=t.unstable_getCurrentPriorityLevel,ze=t.unstable_ImmediatePriority,Be=t.unstable_UserBlockingPriority,Ve=t.unstable_NormalPriority,He=t.unstable_LowPriority,Ue=t.unstable_IdlePriority,We=t.log,Ge=t.unstable_setDisableYieldValue,Ke=null,qe=null;function Je(e){if(typeof We==`function`&&Ge(e),qe&&typeof qe.setStrictMode==`function`)try{qe.setStrictMode(Ke,e)}catch{}}var Ye=Math.clz32?Math.clz32:Qe,Xe=Math.log,Ze=Math.LN2;function Qe(e){return e>>>=0,e===0?32:31-(Xe(e)/Ze|0)|0}var $e=256,et=262144,tt=4194304;function nt(e){var t=e&42;if(t!==0)return t;switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return e&-e;case 262144:case 524288:case 1048576:case 2097152:return e&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return e&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return e}}function rt(e,t,n){var r=e.pendingLanes;if(r===0)return 0;var i=0,a=e.suspendedLanes,o=e.pingedLanes;e=e.warmLanes;var s=r&134217727;return s===0?(s=r&~a,s===0?o===0?n||(n=r&~e,n!==0&&(i=nt(n))):i=nt(o):i=nt(s)):(r=s&~a,r===0?(o&=s,o===0?n||(n=s&~e,n!==0&&(i=nt(n))):i=nt(o)):i=nt(r)),i===0?0:t!==0&&t!==i&&(t&a)===0&&(a=i&-i,n=t&-t,a>=n||a===32&&n&4194048)?t:i}function it(e,t){return(e.pendingLanes&~(e.suspendedLanes&~e.pingedLanes)&t)===0}function at(e,t){t&8&&(t|=t&32);var n=e.entangledLanes;if(n!==0)for(e=e.entanglements,n&=t;0<n;){var r=31-Ye(n),i=1<<r;t|=e[r],n&=~i}return t}function ot(e,t){switch(e){case 1:case 2:case 4:case 8:case 64:return t+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function st(){var e=tt;return tt<<=1,!(tt&62914560)&&(tt=4194304),e}function ct(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function lt(e,t){e.pendingLanes|=t,t!==268435456&&(e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0)}function ut(e,t,n,r,i,a){var o=e.pendingLanes;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0,e.expiredLanes&=n,e.entangledLanes&=n,e.errorRecoveryDisabledLanes&=n,e.shellSuspendCounter=0;var s=e.entanglements,c=e.expirationTimes,l=e.hiddenUpdates;for(n=o&~n;0<n;){var u=31-Ye(n),d=1<<u;s[u]=0,c[u]=-1;var f=l[u];if(f!==null)for(l[u]=null,u=0;u<f.length;u++){var p=f[u];p!==null&&(p.lane&=-536870913)}n&=~d}r!==0&&dt(e,r,0),a!==0&&i===0&&e.tag!==0&&(e.suspendedLanes|=a&~(o&~t))}function dt(e,t,n){e.pendingLanes|=t,e.suspendedLanes&=~t;var r=31-Ye(t);e.entangledLanes|=t,e.entanglements[r]=e.entanglements[r]|1073741824|n&261930}function ft(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-Ye(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}function pt(e,t){var n=t&-t;return n=n&42?1:mt(n),(n&(e.suspendedLanes|t))===0?n:0}function mt(e){switch(e){case 2:e=1;break;case 8:e=4;break;case 32:e=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:e=128;break;case 268435456:e=134217728;break;default:e=0}return e}function ht(e){return e&=-e,2<e?8<e?e&134217727?32:268435456:8:2}function gt(){var e=R.p;return e===0?(e=window.event,e===void 0?32:Ch(e.type)):e}function _t(e,t){var n=R.p;try{return R.p=e,t()}finally{R.p=n}}var vt=Math.random().toString(36).slice(2),yt=`__reactFiber$`+vt,bt=`__reactProps$`+vt,xt=`__reactContainer$`+vt,St=`__reactEvents$`+vt,Ct=`__reactListeners$`+vt,wt=`__reactHandles$`+vt,Tt=`__reactResources$`+vt,Et=`__reactMarker$`+vt,Dt=`__reactLoad$`+vt;function Ot(e){delete e[yt],delete e[bt],delete e[Ct],delete e[wt]}function kt(e){var t;if(t=e[yt])return t;for(var n=e.parentNode;n;){if(t=n[xt]||n[yt]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=fm(e);e!==null;){if(n=e[yt])return n;e=fm(e)}return t}e=n,n=e.parentNode}return null}function At(e){if(e=e[yt]||e[xt]){var t=e.tag;if(t===5||t===6||t===13||t===31||t===26||t===27||t===3)return e}return null}function jt(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e.stateNode;throw Error(i(33))}function Mt(e){var t=e[Tt];return t||=e[Tt]={hoistableStyles:new Map,hoistableScripts:new Map},t}function Nt(e){e[Et]=!0}function Pt(e){e[Dt]=void 0}var Ft=new Set,It={};function Lt(e,t){Rt(e,t),Rt(e+`Capture`,t)}function Rt(e,t){for(It[e]=t,e=0;e<t.length;e++)Ft.add(t[e])}var zt=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),Bt={},H={};function Vt(e){return Me.call(H,e)?!0:Me.call(Bt,e)?!1:zt.test(e)?H[e]=!0:(Bt[e]=!0,!1)}var U=!1;function Ht(){var e=U;return U=!1,e}function Ut(e,t,n){if(Vt(t)){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:e.removeAttribute(t);return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`){e.removeAttribute(t);return}}e.setAttribute(t,n)}}}function Wt(e,t,n){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(t);return}e.setAttribute(t,n)}}function Gt(e,t,n,r){if(r===null)e.removeAttribute(n);else{switch(typeof r){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(n);return}e.setAttributeNS(t,n,r)}}function Kt(e){switch(typeof e){case`bigint`:case`boolean`:case`number`:case`string`:case`undefined`:return e;case`object`:return e;default:return``}}function qt(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()===`input`&&(t===`checkbox`||t===`radio`)}function Jt(e,t,n){var r=Object.getOwnPropertyDescriptor(e.constructor.prototype,t);if(!e.hasOwnProperty(t)&&r!==void 0&&typeof r.get==`function`&&typeof r.set==`function`){var i=r.get,a=r.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(e){n=``+e,a.call(this,e)}}),Object.defineProperty(e,t,{enumerable:r.enumerable}),{getValue:function(){return n},setValue:function(e){n=``+e},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function Yt(e){if(!e._valueTracker){var t=qt(e)?`checked`:`value`;e._valueTracker=Jt(e,t,``+e[t])}}function Xt(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r=``;return e&&(r=qt(e)?e.checked?`true`:`false`:e.value),e=r,e!==n&&(t.setValue(e),!0)}var Zt=/[\n"\\]/g;function Qt(e){return e.replace(Zt,function(e){return`\\`+e.charCodeAt(0).toString(16)+` `})}function $t(e,t,n,r,i,a,o,s){e.name=``,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`?e.type=o:e.removeAttribute(`type`),t==null?o!==`submit`&&o!==`reset`||e.removeAttribute(`value`):o===`number`?(t===0&&e.value===``||e.value!=t)&&(e.value=``+Kt(t)):e.value!==``+Kt(t)&&(e.value=``+Kt(t)),t==null?n==null?r!=null&&e.removeAttribute(`value`):tn(e,Kt(n)):o===`number`&&e.value==t?tn(e,Kt(e.value)):tn(e,Kt(t)),i==null&&a!=null&&(e.defaultChecked=!!a),i!=null&&(e.checked=i&&typeof i!=`function`&&typeof i!=`symbol`),s!=null&&typeof s!=`function`&&typeof s!=`symbol`&&typeof s!=`boolean`?e.name=``+Kt(s):e.removeAttribute(`name`)}function en(e,t,n,r,i,a,o,s){if(a!=null&&typeof a!=`function`&&typeof a!=`symbol`&&typeof a!=`boolean`&&(e.type=a),t!=null||n!=null){if(!(a!==`submit`&&a!==`reset`||t!=null)){Yt(e);return}n=n==null?``:``+Kt(n),t=t==null?n:``+Kt(t),s||t===e.value||(e.value=t),e.defaultValue=t}r??=i,r=typeof r!=`function`&&typeof r!=`symbol`&&!!r,e.checked=s?e.checked:!!r,e.defaultChecked=!!r,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`&&(e.name=o),Yt(e)}function tn(e,t){e.defaultValue!==``+t&&(e.defaultValue=``+t)}function nn(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t[`$`+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty(`$`+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=``+Kt(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function rn(e,t,n){if(t!=null&&(t=``+Kt(t),t!==e.value&&(e.value=t),n==null)){e.defaultValue!==t&&(e.defaultValue=t);return}e.defaultValue=n==null?``:``+Kt(n)}function an(e,t,n,r){if(t==null){if(r!=null){if(n!=null)throw Error(i(92));if(pe(r)){if(1<r.length)throw Error(i(93));r=r[0]}n=r}n??=``,t=n}n=Kt(t),e.defaultValue=n,r=e.textContent,r===n&&r!==``&&r!==null&&(e.value=r),Yt(e)}function on(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var sn=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `));function cn(e,t,n){var r=t.indexOf(`--`)===0;n==null||typeof n==`boolean`||n===``?r?e.setProperty(t,``):t===`float`?e.cssFloat=``:e[t]=``:r?e.setProperty(t,n):typeof n!=`number`||n===0||sn.has(t)?t===`float`?e.cssFloat=n:e[t]=(``+n).trim():e[t]=n+`px`}function ln(e,t,n){if(t!=null&&typeof t!=`object`)throw Error(i(62));if(e=e.style,n!=null){for(var r in n)!n.hasOwnProperty(r)||t!=null&&t.hasOwnProperty(r)||(r.indexOf(`--`)===0?e.setProperty(r,``):r===`float`?e.cssFloat=``:e[r]=``,U=!0);for(var a in t)r=t[a],t.hasOwnProperty(a)&&n[a]!==r&&(cn(e,a,r),U=!0)}else for(var o in t)t.hasOwnProperty(o)&&cn(e,o,t[o])}function un(e){if(e.indexOf(`-`)===-1)return!1;switch(e){case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:return!1;default:return!0}}var dn=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`maskType`,`mask-type`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),fn=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function pn(e){return fn.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}function mn(){}var hn=null;function gn(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var _n=null,vn=null;function yn(e){var t=At(e);if(t&&(e=t.stateNode)){var n=e[bt]||null;a:switch(e=t.stateNode,t.type){case`input`:if($t(e,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),t=n.name,n.type===`radio`&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll(`input[name="`+Qt(``+t)+`"][type="radio"]`),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var a=r[bt]||null;if(!a)throw Error(i(90));$t(r,a.value,a.defaultValue,a.defaultValue,a.checked,a.defaultChecked,a.type,a.name)}}for(t=0;t<n.length;t++)r=n[t],r.form===e.form&&Xt(r)}break a;case`textarea`:rn(e,n.value,n.defaultValue);break a;case`select`:t=n.value,t!=null&&nn(e,!!n.multiple,t,!1)}}}var bn=!1;function xn(e,t,n){if(bn)return e(t,n);bn=!0;try{return e(t)}finally{if(bn=!1,(_n!==null||vn!==null)&&(Rd(),_n&&(t=_n,e=vn,vn=_n=null,yn(t),e)))for(t=0;t<e.length;t++)yn(e[t])}}function Sn(e,t){var n=e.stateNode;if(n===null)return null;var r=n[bt]||null;if(r===null)return null;n=r[t];a:switch(t){case`onClick`:case`onClickCapture`:case`onDoubleClick`:case`onDoubleClickCapture`:case`onMouseDown`:case`onMouseDownCapture`:case`onMouseMove`:case`onMouseMoveCapture`:case`onMouseUp`:case`onMouseUpCapture`:case`onMouseEnter`:(r=!r.disabled)||(e=e.type,r=e!==`button`&&e!==`input`&&e!==`select`&&e!==`textarea`),e=!r;break a;default:e=!1}if(e)return null;if(n&&typeof n!=`function`)throw Error(i(231,t,typeof n));return n}var Cn=!(typeof window>`u`||window.document===void 0||window.document.createElement===void 0),wn=!1;if(Cn)try{var Tn={};Object.defineProperty(Tn,"passive",{get:function(){wn=!0}}),window.addEventListener(`test`,Tn,Tn),window.removeEventListener(`test`,Tn,Tn)}catch{wn=!1}var En=null,Dn=null,On=null;function kn(){if(On)return On;var e,t=Dn,n=t.length,r,i=`value`in En?En.value:En.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return On=i.slice(e,1<r?1-r:void 0)}function An(e){var t=e.keyCode;return`charCode`in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function jn(){return!0}function Mn(){return!1}function Nn(e){function t(t,n,r,i,a){for(var o in this._reactName=t,this._targetInst=r,this.type=n,this.nativeEvent=i,this.target=a,this.currentTarget=null,e)e.hasOwnProperty(o)&&(t=e[o],this[o]=t?t(i):i[o]);return this.isDefaultPrevented=(i.defaultPrevented==null?!1===i.returnValue:i.defaultPrevented)?jn:Mn,this.isPropagationStopped=Mn,this}return D(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var e=this.nativeEvent;e&&(e.preventDefault?e.preventDefault():typeof e.returnValue!=`unknown`&&(e.returnValue=!1),this.isDefaultPrevented=jn)},stopPropagation:function(){var e=this.nativeEvent;e&&(e.stopPropagation?e.stopPropagation():typeof e.cancelBubble!=`unknown`&&(e.cancelBubble=!0),this.isPropagationStopped=jn)},persist:function(){},isPersistent:jn}),t}var Pn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Fn=Nn(Pn),In=D({},Pn,{view:0,detail:0}),Ln=Nn(In),Rn,zn,Bn,Vn=D({},In,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:Qn,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return`movementX`in e?e.movementX:(e!==Bn&&(Bn&&e.type===`mousemove`?(Rn=e.screenX-Bn.screenX,zn=e.screenY-Bn.screenY):zn=Rn=0,Bn=e),Rn)},movementY:function(e){return`movementY`in e?e.movementY:zn}}),Hn=Nn(Vn),Un=Nn(D({},Vn,{dataTransfer:0})),Wn=Nn(D({},In,{relatedTarget:0})),Gn=Nn(D({},Pn,{animationName:0,elapsedTime:0,pseudoElement:0})),Kn=Nn(D({},Pn,{clipboardData:function(e){return`clipboardData`in e?e.clipboardData:window.clipboardData}})),qn=Nn(D({},Pn,{data:0})),Jn={Esc:`Escape`,Spacebar:` `,Left:`ArrowLeft`,Up:`ArrowUp`,Right:`ArrowRight`,Down:`ArrowDown`,Del:`Delete`,Win:`OS`,Menu:`ContextMenu`,Apps:`ContextMenu`,Scroll:`ScrollLock`,MozPrintableKey:`Unidentified`},Yn={8:`Backspace`,9:`Tab`,12:`Clear`,13:`Enter`,16:`Shift`,17:`Control`,18:`Alt`,19:`Pause`,20:`CapsLock`,27:`Escape`,32:` `,33:`PageUp`,34:`PageDown`,35:`End`,36:`Home`,37:`ArrowLeft`,38:`ArrowUp`,39:`ArrowRight`,40:`ArrowDown`,45:`Insert`,46:`Delete`,112:`F1`,113:`F2`,114:`F3`,115:`F4`,116:`F5`,117:`F6`,118:`F7`,119:`F8`,120:`F9`,121:`F10`,122:`F11`,123:`F12`,144:`NumLock`,145:`ScrollLock`,224:`Meta`},Xn={Alt:`altKey`,Control:`ctrlKey`,Meta:`metaKey`,Shift:`shiftKey`};function Zn(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=Xn[e])?!!t[e]:!1}function Qn(){return Zn}var $n=Nn(D({},In,{key:function(e){if(e.key){var t=Jn[e.key]||e.key;if(t!==`Unidentified`)return t}return e.type===`keypress`?(e=An(e),e===13?`Enter`:String.fromCharCode(e)):e.type===`keydown`||e.type===`keyup`?Yn[e.keyCode]||`Unidentified`:``},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:Qn,charCode:function(e){return e.type===`keypress`?An(e):0},keyCode:function(e){return e.type===`keydown`||e.type===`keyup`?e.keyCode:0},which:function(e){return e.type===`keypress`?An(e):e.type===`keydown`||e.type===`keyup`?e.keyCode:0}})),er=Nn(D({},Vn,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0})),tr=Nn(D({},Pn,{submitter:0})),nr=Nn(D({},In,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:Qn})),rr=Nn(D({},Pn,{propertyName:0,elapsedTime:0,pseudoElement:0})),ir=Nn(D({},Vn,{deltaX:function(e){return`deltaX`in e?e.deltaX:`wheelDeltaX`in e?-e.wheelDeltaX:0},deltaY:function(e){return`deltaY`in e?e.deltaY:`wheelDeltaY`in e?-e.wheelDeltaY:`wheelDelta`in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0})),ar=Nn(D({},Pn,{newState:0,oldState:0,source:0})),or=[9,13,27,32],sr=Cn&&`CompositionEvent`in window,cr=null;Cn&&`documentMode`in document&&(cr=document.documentMode);var lr=Cn&&`TextEvent`in window&&!cr,ur=Cn&&(!sr||cr&&8<cr&&11>=cr),dr=` `,fr=!1;function pr(e,t){switch(e){case`keyup`:return or.indexOf(t.keyCode)!==-1;case`keydown`:return t.keyCode!==229;case`keypress`:case`mousedown`:case`focusout`:return!0;default:return!1}}function mr(e){return e=e.detail,typeof e==`object`&&`data`in e?e.data:null}var hr=!1;function gr(e,t){switch(e){case`compositionend`:return mr(t);case`keypress`:return t.which===32?(fr=!0,dr):null;case`textInput`:return e=t.data,e===dr&&fr?null:e;default:return null}}function _r(e,t){if(hr)return e===`compositionend`||!sr&&pr(e,t)?(e=kn(),On=Dn=En=null,hr=!1,e):null;switch(e){case`paste`:return null;case`keypress`:if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case`compositionend`:return ur&&t.locale!==`ko`?null:t.data;default:return null}}var vr={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function yr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t===`input`?!!vr[e.type]:t===`textarea`}function br(e,t,n,r){_n?vn?vn.push(r):vn=[r]:_n=r,t=Jf(t,`onChange`),0<t.length&&(n=new Fn(`onChange`,`change`,null,n,r),e.push({event:n,listeners:t}))}var xr=null,Sr=null;function Cr(e){Vf(e,0)}function wr(e){if(Xt(jt(e)))return e}function Tr(e,t){if(e===`change`)return t}var Er=!1;if(Cn){var Dr;if(Cn){var Or=`oninput`in document;if(!Or){var kr=document.createElement(`div`);kr.setAttribute(`oninput`,`return;`),Or=typeof kr.oninput==`function`}Dr=Or}else Dr=!1;Er=Dr&&(!document.documentMode||9<document.documentMode)}function Ar(){xr&&(xr.detachEvent(`onpropertychange`,jr),Sr=xr=null)}function jr(e){if(e.propertyName===`value`&&wr(Sr)){var t=[];br(t,Sr,e,gn(e)),xn(Cr,t)}}function Mr(e,t,n){e===`focusin`?(Ar(),xr=t,Sr=n,xr.attachEvent(`onpropertychange`,jr)):e===`focusout`&&Ar()}function Nr(e){if(e===`selectionchange`||e===`keyup`||e===`keydown`)return wr(Sr)}function Pr(e,t){if(e===`click`)return wr(t)}function Fr(e,t){if(e===`input`||e===`change`)return wr(t)}function Ir(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var Lr=typeof Object.is==`function`?Object.is:Ir;function Rr(e,t){if(Lr(e,t))return!0;if(typeof e!=`object`||!e||typeof t!=`object`||!t)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!Me.call(t,i)||!Lr(e[i],t[i]))return!1}return!0}function zr(e){if(e||=typeof document<`u`?document:void 0,e===void 0)return null;try{return e.activeElement||e.body}catch{return e.body}}function Br(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Vr(e,t){var n=Br(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}a:{for(;n;){if(n.nextSibling){n=n.nextSibling;break a}n=n.parentNode}n=void 0}n=Br(n)}}function Hr(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Hr(e,t.parentNode):`contains`in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function Ur(e){e=e!=null&&e.ownerDocument!=null&&e.ownerDocument.defaultView!=null?e.ownerDocument.defaultView:window;for(var t=zr(e.document);t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href==`string`}catch{n=!1}if(n)e=t.contentWindow;else break;t=zr(e.document)}return t}function Wr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t===`input`&&(e.type===`text`||e.type===`search`||e.type===`tel`||e.type===`url`||e.type===`password`)||t===`textarea`||e.contentEditable===`true`)}var Gr=Cn&&`documentMode`in document&&11>=document.documentMode,Kr=null,qr=null,Jr=null,Yr=!1;function Xr(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;Yr||Kr==null||Kr!==zr(r)||(r=Kr,`selectionStart`in r&&Wr(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),Jr&&Rr(Jr,r)||(Jr=r,r=Jf(qr,`onSelect`),0<r.length&&(t=new Fn(`onSelect`,`select`,null,t,n),e.push({event:t,listeners:r}),t.target=Kr)))}function Zr(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n[`Webkit`+e]=`webkit`+t,n[`Moz`+e]=`moz`+t,n}var Qr={animationend:Zr(`Animation`,`AnimationEnd`),animationiteration:Zr(`Animation`,`AnimationIteration`),animationstart:Zr(`Animation`,`AnimationStart`),transitionrun:Zr(`Transition`,`TransitionRun`),transitionstart:Zr(`Transition`,`TransitionStart`),transitioncancel:Zr(`Transition`,`TransitionCancel`),transitionend:Zr(`Transition`,`TransitionEnd`)},$r={},ei={};Cn&&(ei=document.createElement(`div`).style,`AnimationEvent`in window||(delete Qr.animationend.animation,delete Qr.animationiteration.animation,delete Qr.animationstart.animation),`TransitionEvent`in window||delete Qr.transitionend.transition);function ti(e){if($r[e])return $r[e];if(!Qr[e])return e;var t=Qr[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in ei)return $r[e]=t[n];return e}var ni=ti(`animationend`),ri=ti(`animationiteration`),ii=ti(`animationstart`),ai=ti(`transitionrun`),oi=ti(`transitionstart`),si=ti(`transitioncancel`),ci=ti(`transitionend`),li=new Map,ui=`abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel`.split(` `);ui.push(`scrollEnd`);function di(e,t){li.set(e,t),Lt(t,[e])}var fi=0;function pi(e,t){if(e.name!=null&&e.name!==`auto`)return e.name;if(t.autoName!==null)return t.autoName;e=yd.identifierPrefix;var n=fi++;return e=`_`+e+`t_`+n.toString(32)+`_`,t.autoName=e}function mi(e){if(e==null||typeof e==`string`)return e;var t=null,n=Dd;if(n!==null)for(var r=0;r<n.length;r++){var i=e[n[r]];if(i!=null){if(i===`none`)return`none`;t=t==null?i:t+(` `+i)}}return t??e.default}function hi(e,t){return e=mi(e),t=mi(t),t==null?e===`auto`?null:e:t===`auto`?null:t}var gi=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)},_i=[],vi=0,yi=0;function bi(){for(var e=vi,t=yi=vi=0;t<e;){var n=_i[t];_i[t++]=null;var r=_i[t];_i[t++]=null;var i=_i[t];_i[t++]=null;var a=_i[t];if(_i[t++]=null,r!==null&&i!==null){var o=r.pending;o===null?i.next=i:(i.next=o.next,o.next=i),r.pending=i}a!==0&&wi(n,i,a)}}function xi(e,t,n,r){_i[vi++]=e,_i[vi++]=t,_i[vi++]=n,_i[vi++]=r,yi|=r,e.lanes|=r,e=e.alternate,e!==null&&(e.lanes|=r)}function Si(e,t,n,r){return xi(e,t,n,r),Ti(e)}function Ci(e,t){return xi(e,null,null,t),Ti(e)}function wi(e,t,n){e.lanes|=n;var r=e.alternate;r!==null&&(r.lanes|=n);for(var i=!1,a=e.return;a!==null;)a.childLanes|=n,r=a.alternate,r!==null&&(r.childLanes|=n),a.tag===22&&(e=a.stateNode,e===null||e._visibility&1||(i=!0)),e=a,a=a.return;return e.tag===3?(a=e.stateNode,i&&t!==null&&(i=31-Ye(n),e=a.hiddenUpdates,r=e[i],r===null?e[i]=[t]:r.push(t),t.lane=n|536870912),a):null}function Ti(e){if(50<Od)throw Od=0,kd=null,Error(i(185));for(var t=e.return;t!==null;)e=t,t=e.return;return e.tag===3?e.stateNode:null}var Ei={};function Di(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Oi(e,t,n,r){return new Di(e,t,n,r)}function ki(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Ai(e,t){var n=e.alternate;return n===null?(n=Oi(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&1206910976,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n.refCleanup=e.refCleanup,n}function ji(e,t){e.flags&=1206910978;var n=e.alternate;return n===null?(e.childLanes=0,e.lanes=t,e.child=null,e.subtreeFlags=0,e.memoizedProps=null,e.memoizedState=null,e.updateQueue=null,e.dependencies=null,e.stateNode=null):(e.childLanes=n.childLanes,e.lanes=n.lanes,e.child=n.child,e.subtreeFlags=0,e.deletions=null,e.memoizedProps=n.memoizedProps,e.memoizedState=n.memoizedState,e.updateQueue=n.updateQueue,e.type=n.type,t=n.dependencies,e.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),e}function Mi(e,t,n,r,a,o){var s=0;if(r=e,typeof r==`function`)ki(r)&&(s=1);else if(typeof r==`string`)s=qm(e,n,ye.current)?26:e===`html`||e===`head`||e===`body`?27:5;else a:switch(r){case se:return e=Oi(31,n,t,a),e.elementType=se,e.lanes=o,e;case j:return Ni(n.children,a,o,t);case ee:s=8,a|=24;break;case te:return e=Oi(12,n,t,a|2),e.elementType=te,e.lanes=o,e;case ie:return e=Oi(13,n,t,a),e.elementType=ie,e.lanes=o,e;case N:return e=Oi(19,n,t,a),e.elementType=N,e.lanes=o,e;case ce:case F:return e=a|32,e=Oi(30,n,t,e),e.elementType=F,e.lanes=o,e.stateNode={autoName:null,paired:null,clones:null,ref:null},e;default:if(typeof r==`object`&&r)switch(r.$$typeof){case M:s=10;break a;case ne:s=9;break a;case re:s=11;break a;case ae:s=14;break a;case oe:s=16,r=null;break a}s=29,n=Error(i(130,e===null?`null`:typeof e,``)),r=null}return t=Oi(s,n,t,a),t.elementType=e,t.type=r,t.lanes=o,t}function Ni(e,t,n,r){return e=Oi(7,e,r,t),e.lanes=n,e}function Pi(e,t,n){return e=Oi(6,e,null,t),e.lanes=n,e}function Fi(e){var t=Oi(18,null,null,0);return t.stateNode=e,t}function Ii(e,t,n){return t=Oi(4,e.children===null?[]:e.children,e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}var Li=new WeakMap;function Ri(e,t){if(typeof e==`object`&&e){var n=Li.get(e);return n===void 0?(t={value:e,source:t,stack:V(t)},Li.set(e,t),t):n}return{value:e,source:t,stack:V(t)}}var zi=[],Bi=0,Vi=null,Hi=0,Ui=[],Wi=0,Gi=null,Ki=1,qi=``;function Ji(e,t){zi[Bi++]=Hi,zi[Bi++]=Vi,Vi=e,Hi=t}function Yi(e,t,n){Ui[Wi++]=Ki,Ui[Wi++]=qi,Ui[Wi++]=Gi,Gi=e;var r=Ki;e=qi;var i=32-Ye(r)-1;r&=~(1<<i),n+=1;var a=32-Ye(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,Ki=1<<32-Ye(t)+i|n<<i|r,qi=a+e}else Ki=1<<a|n<<i|r,qi=e}function Xi(e){e.return!==null&&(Ji(e,1),Yi(e,1,0))}function Zi(e){for(;e===Vi;)Vi=zi[--Bi],zi[Bi]=null,Hi=zi[--Bi],zi[Bi]=null;for(;e===Gi;)Gi=Ui[--Wi],Ui[Wi]=null,qi=Ui[--Wi],Ui[Wi]=null,Ki=Ui[--Wi],Ui[Wi]=null}function Qi(e,t){Ui[Wi++]=Ki,Ui[Wi++]=qi,Ui[Wi++]=Gi,Ki=t.id,qi=t.overflow,Gi=e}var $i=null,ea=null,W=!1,ta=null,na=!1,ra=Error(i(519));function ia(e){throw ua(Ri(Error(i(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?`text`:`HTML`,``)),e)),ra}function aa(e){var t=e.stateNode,n=e.type,r=e.memoizedProps;switch(t[yt]=e,t[bt]=r,n){case`dialog`:Q(`cancel`,t),Q(`close`,t);break;case`iframe`:case`object`:case`embed`:Q(`load`,t);break;case`video`:case`audio`:for(n=0;n<zf.length;n++)Q(zf[n],t);break;case`source`:Q(`error`,t);break;case`img`:case`image`:case`link`:Q(`error`,t),Q(`load`,t);break;case`details`:Q(`toggle`,t);break;case`input`:Q(`invalid`,t),en(t,r.value,r.defaultValue,r.checked,r.defaultChecked,r.type,r.name,!0);break;case`select`:Q(`invalid`,t);break;case`textarea`:Q(`invalid`,t),an(t,r.value,r.defaultValue,r.children)}n=r.children,typeof n!=`string`&&typeof n!=`number`&&typeof n!=`bigint`||t.textContent===``+n||!0===r.suppressHydrationWarning||ep(t.textContent,n)?(r.popover!=null&&(Q(`beforetoggle`,t),Q(`toggle`,t)),r.onScroll!=null&&Q(`scroll`,t),r.onScrollEnd!=null&&Q(`scrollend`,t),r.onClick!=null&&(t.onclick=mn),t=!0):t=!1,t||ia(e,!0)}function oa(e){for($i=e.return;$i;)switch($i.tag){case 5:case 31:case 13:na=!1;return;case 27:case 3:na=!0;return;default:$i=$i.return}}function sa(e){if(e!==$i)return!1;if(!W)return oa(e),W=!0,!1;var t=e.tag,n;if((n=t!==3&&t!==27)&&((n=t===5)&&(n=e.type,n=n===`form`||n===`button`||pp(e.type,e.memoizedProps)),n=!n),n&&ea&&ia(e),oa(e),t===13){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));ea=dm(e)}else if(t===31){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));ea=dm(e)}else t===27?(t=ea,Sp(e.type)?(e=um,um=null,ea=e):ea=t):ea=$i?lm(e.stateNode.nextSibling):null;return!0}function ca(){ea=$i=null,W=!1}function la(){var e=ta;return e!==null&&(dd===null?dd=e:dd.push.apply(dd,e),ta=null),e}function ua(e){ta===null?ta=[e]:ta.push(e)}var da=_e(null),fa=null,pa=null;function ma(e,t,n){z(da,t._currentValue),t._currentValue=n}function ha(e){e._currentValue=da.current,ve(da)}function ga(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)===t?r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t):(e.childLanes|=t,r!==null&&(r.childLanes|=t)),e===n)break;e=e.return}}function _a(e,t,n,r){var a=e.child;for(a!==null&&(a.return=e);a!==null;){var o=a.dependencies;if(o!==null){var s=a.child;o=o.firstContext;a:for(;o!==null;){var c=o;o=a;for(var l=0;l<t.length;l++)if(c.context===t[l]){o.lanes|=n,c=o.alternate,c!==null&&(c.lanes|=n),ga(o.return,n,e),r||(s=null);break a}o=c.next}}else if(a.tag===18){if(s=a.return,s===null)throw Error(i(341));s.lanes|=n,o=s.alternate,o!==null&&(o.lanes|=n),ga(s,n,e),s=null}else a.tag===13&&a.memoizedState!==null&&a.memoizedState.dehydrated===null?(a.lanes|=n,s=a.alternate,s!==null&&(s.lanes|=n),ga(a.return,n,e),s=a.child,s=s===null?null:s.sibling):s=a.child;if(s!==null)s.return=a;else for(s=a;s!==null;){if(s===e){s=null;break}if(a=s.sibling,a!==null){a.return=s.return,s=a;break}s=s.return}a=s}}function va(e,t,n,r){e=null;for(var a=t,o=!1;a!==null;){if(!o){if(a.flags&524288)o=!0;else if(a.flags&262144)break}if(a.tag===10){var s=a.alternate;if(s===null)throw Error(i(387));if(s=s.memoizedProps,s!==null){var c=a.type;Lr(a.pendingProps.value,s.value)||(e===null?e=[c]:e.push(c))}}else if(a===Se.current){if(s=a.alternate,s===null)throw Error(i(387));s.memoizedState.memoizedState!==a.memoizedState.memoizedState&&(e===null?e=[sh]:e.push(sh))}a=a.return}return e!==null&&_a(t,e,n,r),t.flags|=262144,e!==null}function ya(e){for(e=e.firstContext;e!==null;){if(!Lr(e.context._currentValue,e.memoizedValue))return!0;e=e.next}return!1}function ba(e){fa=e,pa=null,e=e.dependencies,e!==null&&(e.firstContext=null)}function xa(e){return Ca(fa,e)}function Sa(e,t){return fa===null&&ba(e),Ca(e,t)}function Ca(e,t){var n=t._currentValue;if(t={context:t,memoizedValue:n,next:null},pa===null){if(e===null)throw Error(i(308));pa=t,e.dependencies={lanes:0,firstContext:t},e.flags|=524288}else pa=pa.next=t;return n}var wa=typeof AbortController<`u`?AbortController:function(){var e=[],t=this.signal={aborted:!1,addEventListener:function(t,n){e.push(n)}};this.abort=function(){t.aborted=!0,e.forEach(function(e){return e()})}},Ta=t.unstable_scheduleCallback,Ea=t.unstable_NormalPriority,Da={$$typeof:M,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function Oa(){return{controller:new wa,data:new Map,refCount:0}}function ka(e){e.refCount--,e.refCount===0&&Ta(Ea,function(){e.controller.abort()})}function Aa(e,t){if(e.pendingLanes&4194048){var n=e.transitionTypes;for(n===null&&(n=e.transitionTypes=[]),e=0;e<t.length;e++){var r=t[e];n.indexOf(r)===-1&&n.push(r)}}}var ja=null;function Ma(e){var t=e.transitionTypes;return e.transitionTypes=null,t}var Na=null,Pa=0,Fa=0,Ia=null;function La(e,t){if(Na===null){var n=Na=[];Pa=0,Fa=Pf(),Ia={status:`pending`,value:void 0,then:function(e){n.push(e)}}}return Pa++,t.then(Ra,Ra),t}function Ra(){if(--Pa===0&&(ja=null,Na!==null)){Ia!==null&&(Ia.status=`fulfilled`);var e=Na;Na=null,Fa=0,Ia=null;for(var t=0;t<e.length;t++)(0,e[t])()}}function za(e,t){var n=[],r={status:`pending`,value:null,reason:null,then:function(e){n.push(e)}};return e.then(function(){r.status=`fulfilled`,r.value=t;for(var e=0;e<n.length;e++)(0,n[e])(t)},function(e){for(r.status=`rejected`,r.reason=e,e=0;e<n.length;e++)(0,n[e])(void 0)}),r}var Ba=L.S;L.S=function(e,t){if(md=Le(),typeof t==`object`&&t&&typeof t.then==`function`&&La(e,t),ja!==null)for(var n=bf;n!==null;)Aa(n,ja),n=n.next;if(n=e.types,n!==null){for(var r=bf;r!==null;)Aa(r,n),r=r.next;if(Fa!==0){r=ja,r===null&&(r=ja=[]);for(var i=0;i<n.length;i++){var a=n[i];r.indexOf(a)===-1&&r.push(a)}}}Ba!==null&&Ba(e,t)};var Va=_e(null);function Ha(){var e=Va.current;return e===null?Qu.pooledCache:e}function Ua(e,t){t===null?z(Va,Va.current):z(Va,t.pool)}function Wa(){var e=Ha();return e===null?null:{parent:Da._currentValue,pool:e}}var Ga=Error(i(460)),Ka=Error(i(474)),qa=Error(i(542)),Ja={then:function(){}};function Ya(e){return e=e.status,e===`fulfilled`||e===`rejected`}function Xa(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(mn,mn),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,eo(e),e===void 0&&!(`reason`in t)?Error(i(600)):e;default:if(typeof t.status==`string`)t.then(mn,mn);else{if(e=Qu,e!==null&&100<e.shellSuspendCounter)throw Error(i(482));e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})}switch(t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,eo(e),e}throw Qa=t,Ga}}function Za(e){try{var t=e._init;return t(e._payload)}catch(e){throw typeof e==`object`&&e&&typeof e.then==`function`?(Qa=e,Ga):e}}var Qa=null;function $a(){if(Qa===null)throw Error(i(459));var e=Qa;return Qa=null,e}function eo(e){if(e===Ga||e===qa)throw Error(i(483))}var to=null,no=0;function ro(e){var t=no;return no+=1,to===null&&(to=[]),Xa(to,e,t)}function io(e,t){t=t.props.ref,e.ref=t===void 0?null:t}function ao(e,t){throw t.$$typeof===O?Error(i(525)):(e=Object.prototype.toString.call(t),Error(i(31,e===`[object Object]`?`object with keys {`+Object.keys(t).join(`, `)+`}`:e)))}function oo(e){function t(t,n){if(e){var r=t.deletions;r===null?(t.deletions=[n],t.flags|=16):r.push(n)}}function n(n,r){if(!e)return null;for(;r!==null;)t(n,r),r=r.sibling;return null}function r(e){for(var t=new Map;e!==null;)e.key===null?t.set(e.index,e):t.set(e.key,e),e=e.sibling;return t}function a(e,t){return e=Ai(e,t),e.index=0,e.sibling=null,e}function o(t,n,r){return t.index=r,e?(r=t.alternate,r===null?(t.flags|=134217730,n):(r=r.index,r<n?(t.flags|=2,n):r)):(t.flags|=1048576,n)}function s(t){return e&&t.alternate===null&&(t.flags|=134217730),t}function c(e,t,n,r){return t===null||t.tag!==6?(t=Pi(n,e.mode,r),t.return=e,t):(t=a(t,n),t.return=e,t)}function l(e,t,n,r){var i=n.type;return i===j?(e=d(e,t,n.props.children,r,n.key),io(e,n),e):t!==null&&(t.elementType===i||typeof i==`object`&&i&&i.$$typeof===oe&&Za(i)===t.type)?(t=a(t,n.props),io(t,n),t.return=e,t):(t=Mi(n.type,n.key,n.props,null,e.mode,r),io(t,n),t.return=e,t)}function u(e,t,n,r){return t===null||t.tag!==4||t.stateNode.containerInfo!==n.containerInfo||t.stateNode.implementation!==n.implementation?(t=Ii(n,e.mode,r),t.return=e,t):(t=a(t,n.children||[]),t.return=e,t)}function d(e,t,n,r,i){return t===null||t.tag!==7?(t=Ni(n,e.mode,r,i),t.return=e,t):(t=a(t,n),t.return=e,t)}function f(e,t,n){if(typeof t==`string`&&t!==``||typeof t==`number`||typeof t==`bigint`)return t=Pi(``+t,e.mode,n),t.return=e,t;if(typeof t==`object`&&t){switch(t.$$typeof){case k:return n=Mi(t.type,t.key,t.props,null,e.mode,n),io(n,t),n.return=e,n;case A:return t=Ii(t,e.mode,n),t.return=e,t;case oe:return t=Za(t),f(e,t,n)}if(pe(t)||de(t))return t=Ni(t,e.mode,n,null),t.return=e,t;if(typeof t.then==`function`)return f(e,ro(t),n);if(t.$$typeof===M)return f(e,Sa(e,t),n);ao(e,t)}return null}function p(e,t,n,r){var i=t===null?null:t.key;if(typeof n==`string`&&n!==``||typeof n==`number`||typeof n==`bigint`)return i===null?c(e,t,``+n,r):null;if(typeof n==`object`&&n){switch(n.$$typeof){case k:return n.key===i?l(e,t,n,r):null;case A:return n.key===i?u(e,t,n,r):null;case oe:return n=Za(n),p(e,t,n,r)}if(pe(n)||de(n))return i===null?d(e,t,n,r,null):null;if(typeof n.then==`function`)return p(e,t,ro(n),r);if(n.$$typeof===M)return p(e,t,Sa(e,n),r);ao(e,n)}return null}function m(e,t,n,r,i){if(typeof r==`string`&&r!==``||typeof r==`number`||typeof r==`bigint`)return e=e.get(n)||null,c(t,e,``+r,i);if(typeof r==`object`&&r){switch(r.$$typeof){case k:return e=e.get(r.key===null?n:r.key)||null,l(t,e,r,i);case A:return e=e.get(r.key===null?n:r.key)||null,u(t,e,r,i);case oe:return r=Za(r),m(e,t,n,r,i)}if(pe(r)||de(r))return e=e.get(n)||null,d(t,e,r,i,null);if(typeof r.then==`function`)return m(e,t,n,ro(r),i);if(r.$$typeof===M)return m(e,t,n,Sa(t,r),i);ao(t,r)}return null}function h(i,a,s,c){for(var l=null,u=null,d=a,h=a=0,g=null;d!==null&&h<s.length;h++){d.index>h?(g=d,d=null):g=d.sibling;var _=p(i,d,s[h],c);if(_===null){d===null&&(d=g);break}e&&d&&_.alternate===null&&t(i,d),a=o(_,a,h),u===null?l=_:u.sibling=_,u=_,d=g}if(h===s.length)return n(i,d),W&&Ji(i,h),l;if(d===null){for(;h<s.length;h++)d=f(i,s[h],c),d!==null&&(a=o(d,a,h),u===null?l=d:u.sibling=d,u=d);return W&&Ji(i,h),l}for(d=r(d);h<s.length;h++)g=m(d,i,h,s[h],c),g!==null&&(e&&(_=g.alternate,_!==null&&d.delete(_.key===null?h:_.key)),a=o(g,a,h),u===null?l=g:u.sibling=g,u=g);return e&&d.forEach(function(e){return t(i,e)}),W&&Ji(i,h),l}function g(a,s,c,l){if(c==null)throw Error(i(151));for(var u=null,d=null,h=s,g=s=0,_=null,v=c.next();h!==null&&!v.done;g++,v=c.next()){h.index>g?(_=h,h=null):_=h.sibling;var y=p(a,h,v.value,l);if(y===null){h===null&&(h=_);break}e&&h&&y.alternate===null&&t(a,h),s=o(y,s,g),d===null?u=y:d.sibling=y,d=y,h=_}if(v.done)return n(a,h),W&&Ji(a,g),u;if(h===null){for(;!v.done;g++,v=c.next())v=f(a,v.value,l),v!==null&&(s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return W&&Ji(a,g),u}for(h=r(h);!v.done;g++,v=c.next())v=m(h,a,g,v.value,l),v!==null&&(e&&(_=v.alternate,_!==null&&h.delete(_.key===null?g:_.key)),s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return e&&h.forEach(function(e){return t(a,e)}),W&&Ji(a,g),u}function _(e,r,o,c){if(typeof o==`object`&&o&&o.type===j&&o.key===null&&o.props.ref===void 0&&(o=o.props.children),typeof o==`object`&&o){switch(o.$$typeof){case k:a:{for(var l=o.key;r!==null;){if(r.key===l){if(l=o.type,l===j){if(r.tag===7){n(e,r.sibling),c=a(r,o.props.children),io(c,o),c.return=e,e=c;break a}}else if(r.elementType===l||typeof l==`object`&&l&&l.$$typeof===oe&&Za(l)===r.type){n(e,r.sibling),c=a(r,o.props),io(c,o),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}o.type===j?(c=Ni(o.props.children,e.mode,c,o.key),io(c,o),c.return=e,e=c):(c=Mi(o.type,o.key,o.props,null,e.mode,c),io(c,o),c.return=e,e=c)}return s(e);case A:a:{for(l=o.key;r!==null;){if(r.key===l){if(r.tag===4&&r.stateNode.containerInfo===o.containerInfo&&r.stateNode.implementation===o.implementation){n(e,r.sibling),c=a(r,o.children||[]),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}c=Ii(o,e.mode,c),c.return=e,e=c}return s(e);case oe:return o=Za(o),_(e,r,o,c)}if(pe(o))return h(e,r,o,c);if(de(o)){if(l=de(o),typeof l!=`function`)throw Error(i(150));return o=l.call(o),g(e,r,o,c)}if(typeof o.then==`function`)return _(e,r,ro(o),c);if(o.$$typeof===M)return _(e,r,Sa(e,o),c);ao(e,o)}return typeof o==`string`&&o!==``||typeof o==`number`||typeof o==`bigint`?(o=``+o,r!==null&&r.tag===6?(n(e,r.sibling),c=a(r,o),c.return=e,e=c):(n(e,r),c=Pi(o,e.mode,c),c.return=e,e=c),s(e)):n(e,r)}return function(e,t,n,r){try{no=0;var i=_(e,t,n,r);return to=null,i}catch(t){if(t===Ga||t===qa)throw t;var a=Oi(29,t,null,e.mode);return a.lanes=r,a.return=e,a}}}var so=oo(!0),co=oo(!1),lo=!1;function uo(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function fo(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,callbacks:null})}function po(e){return{lane:e,tag:0,payload:null,callback:null,next:null}}function mo(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,J&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,t=Ti(e),wi(e,null,n),t}return xi(e,r,t,n),Ti(e)}function ho(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,n&4194048)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ft(e,n)}}function go(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,callbacks:r.callbacks},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}var _o=!1;function vo(){if(_o){var e=Ia;if(e!==null)throw e}}function yo(e,t,n,r){_o=!1;var i=e.updateQueue;lo=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var c=s,l=c.next;c.next=null,o===null?a=l:o.next=l,o=c;var u=e.alternate;u!==null&&(u=u.updateQueue,s=u.lastBaseUpdate,s!==o&&(s===null?u.firstBaseUpdate=l:s.next=l,u.lastBaseUpdate=c))}if(a!==null){var d=i.baseState;o=0,u=l=c=null,s=a;do{var f=s.lane&-536870913,p=f!==s.lane;if(p?(X&f)===f:(r&f)===f){f!==0&&f===Fa&&(_o=!0),u!==null&&(u=u.next={lane:0,tag:s.tag,payload:s.payload,callback:null,next:null});a:{var m=e,h=s;f=t;var g=n;switch(h.tag){case 1:if(m=h.payload,typeof m==`function`){d=m.call(g,d,f);break a}d=m;break a;case 3:m.flags=m.flags&-65537|128;case 0:if(m=h.payload,f=typeof m==`function`?m.call(g,d,f):m,f==null)break a;d=D({},d,f);break a;case 2:lo=!0}}f=s.callback,f!==null&&(e.flags|=64,p&&(e.flags|=8192),p=i.callbacks,p===null?i.callbacks=[f]:p.push(f))}else p={lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},u===null?(l=u=p,c=d):u=u.next=p,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(1);u===null&&(c=d),i.baseState=c,i.firstBaseUpdate=l,i.lastBaseUpdate=u,a===null&&(i.shared.lanes=0),ad|=o,e.lanes=o,e.memoizedState=d}}function bo(e,t){if(typeof e!=`function`)throw Error(i(191,e));e.call(t)}function xo(e,t){var n=e.callbacks;if(n!==null)for(e.callbacks=null,e=0;e<n.length;e++)bo(n[e],t)}var So=_e(null),Co=_e(0);function wo(e,t){e=rd,z(Co,e),z(So,t),rd=e|t.baseLanes}function To(){z(Co,rd),z(So,So.current)}function Eo(){rd=Co.current,ve(So),ve(Co)}var Do=_e(null),Oo=null;function ko(e){var t=e.alternate;z(Po,Po.current&1),z(Do,e),Oo===null&&(t===null||So.current!==null||t.memoizedState!==null)&&(Oo=e)}function Ao(e){z(Po,Po.current),z(Do,e),Oo===null&&(Oo=e)}function jo(e){e.tag===22?(z(Po,Po.current),z(Do,e),Oo===null&&(Oo=e)):Mo()}function Mo(){z(Po,Po.current),z(Do,Do.current)}function No(e){ve(Do),Oo===e&&(Oo=null),ve(Po)}var Po=_e(0);function Fo(e,t){z(Do,Do.current),z(Po,t)}function Io(e){ve(Po),ve(Do),Oo===e&&(Oo=null)}function Lo(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||om(n)||sm(n)))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==`independent`){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Ro=0,G=null,K=null,zo=null,Bo=!1,Vo=!1,Ho=!1,Uo=0,Wo=0,Go=null,Ko=0;function qo(){throw Error(i(321))}function Jo(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!Lr(e[n],t[n]))return!1;return!0}function Yo(e,t,n,r,i,a){return Ro=a,G=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,L.H=e===null||e.memoizedState===null?fc:pc,Ho=!1,a=n(r,i),Ho=!1,Vo&&(a=Zo(t,n,r,i)),Xo(e),a}function Xo(e){L.H=dc;var t=K!==null&&K.next!==null;if(Ro=0,zo=K=G=null,Bo=!1,Wo=0,Go=null,t)throw Error(i(300));e===null||Ac||(e=e.dependencies,e!==null&&ya(e)&&(Ac=!0))}function Zo(e,t,n,r){G=e;var a=0;do{if(Vo&&(Go=null),Wo=0,Vo=!1,25<=a)throw Error(i(301));if(a+=1,zo=K=null,e.updateQueue!=null){var o=e.updateQueue;o.lastEffect=null,o.events=null,o.stores=null,o.memoCache!=null&&(o.memoCache.index=0)}L.H=mc,o=t(n,r)}while(Vo);return o}function Qo(){var e=L.H,t=e.useState()[0];return t=typeof t.then==`function`?as(t):t,e=e.useState()[0],(K===null?null:K.memoizedState)!==e&&(G.flags|=1024),t}function $o(){var e=Uo!==0;return Uo=0,e}function es(e,t,n){t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~n}function ts(e){if(Bo){for(e=e.memoizedState;e!==null;){var t=e.queue;t!==null&&(t.pending=null),e=e.next}Bo=!1}Ro=0,zo=K=G=null,Vo=!1,Wo=Uo=0,Go=null}function ns(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return zo===null?G.memoizedState=zo=e:zo=zo.next=e,zo}function rs(){if(K===null){var e=G.alternate;e=e===null?null:e.memoizedState}else e=K.next;var t=zo===null?G.memoizedState:zo.next;if(t!==null)zo=t,K=e;else{if(e===null)throw G.alternate===null?Error(i(467)):Error(i(310));K=e,e={memoizedState:K.memoizedState,baseState:K.baseState,baseQueue:K.baseQueue,queue:K.queue,next:null},zo===null?G.memoizedState=zo=e:zo=zo.next=e}return zo}function is(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function as(e){var t=Wo;return Wo+=1,Go===null&&(Go=[]),e=Xa(Go,e,t),t=G,(zo===null?t.memoizedState:zo.next)===null&&(t=t.alternate,L.H=t===null||t.memoizedState===null?fc:pc),e}function os(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return as(e);if(e.$$typeof===le)return;if(e.$$typeof===M)return xa(e)}throw Error(i(438,String(e)))}function ss(e){var t=null,n=G.updateQueue;if(n!==null&&(t=n.memoCache),t==null){var r=G.alternate;r!==null&&(r=r.updateQueue,r!==null&&(r=r.memoCache,r!=null&&(t={data:r.data.map(function(e){return e.slice()}),index:0})))}if(t??={data:[],index:0},n===null&&(n=is(),G.updateQueue=n),n.memoCache=t,n=t.data[t.index],n===void 0)for(n=t.data[t.index]=Array(e),r=0;r<e;r++)n[r]=P;return t.index++,n}function cs(e,t){return typeof t==`function`?t(e):t}function ls(e){return us(rs(),K,e)}function us(e,t,n){var r=e.queue;if(r===null)throw Error(i(311));r.lastRenderedReducer=n;var a=e.baseQueue,o=r.pending;if(o!==null){if(a!==null){var s=a.next;a.next=o.next,o.next=s}t.baseQueue=a=o,r.pending=null}if(o=e.baseState,a===null)e.memoizedState=o;else{t=a.next;var c=s=null,l=null,u=t,d=!1;do{var f=u.lane&-536870913;if(f===u.lane?(Ro&f)===f:(X&f)===f){var p=u.revertLane;if(p===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),f===Fa&&(d=!0);else if((Ro&p)===p){u=u.next,p===Fa&&(d=!0);continue}else f={lane:0,revertLane:u.revertLane,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=f,s=o):l=l.next=f,G.lanes|=p,ad|=p;f=u.action,Ho&&n(o,f),o=u.hasEagerState?u.eagerState:n(o,f)}else p={lane:f,revertLane:u.revertLane,gesture:u.gesture,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=p,s=o):l=l.next=p,G.lanes|=f,ad|=f;u=u.next}while(u!==null&&u!==t);if(l===null?s=o:l.next=c,!Lr(o,e.memoizedState)&&(Ac=!0,d&&(n=Ia,n!==null)))throw n;e.memoizedState=o,e.baseState=s,e.baseQueue=l,r.lastRenderedState=o}return a===null&&(r.lanes=0),[e.memoizedState,r.dispatch]}function ds(e){var t=rs(),n=t.queue;if(n===null)throw Error(i(311));n.lastRenderedReducer=e;var r=n.dispatch,a=n.pending,o=t.memoizedState;if(a!==null){n.pending=null;var s=a=a.next;do o=e(o,s.action),s=s.next;while(s!==a);Lr(o,t.memoizedState)||(Ac=!0),t.memoizedState=o,t.baseQueue===null&&(t.baseState=o),n.lastRenderedState=o}return[o,r]}function fs(e,t,n){var r=G,a=rs(),o=W;if(o){if(n===void 0)throw Error(i(407));n=n()}else n=t();var s=!Lr((K||a).memoizedState,n);if(s&&(a.memoizedState=n,Ac=!0),a=a.queue,Ls(hs.bind(null,r,a,e),[e]),e=a.getSnapshot!==t||s||zo!==null&&!!(zo.memoizedState.tag&1),Ms(e?9:8,{destroy:void 0},ms.bind(null,r,a,n,t),null),e){if(r.flags|=2048,Qu===null)throw Error(i(349));o||Ro&127||ps(r,t,n)}return n}function ps(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=G.updateQueue,t===null?(t=is(),G.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function ms(e,t,n,r){t.value=n,t.getSnapshot=r,gs(t)&&_s(e)}function hs(e,t,n){return n(function(){gs(t)&&_s(e)})}function gs(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!Lr(e,n)}catch{return!0}}function _s(e){var t=Ci(e,2);t!==null&&Nd(t,e,2)}function vs(e){var t=ns();if(typeof e==`function`){var n=e;if(e=n(),Ho){Je(!0);try{n()}finally{Je(!1)}}}return t.memoizedState=t.baseState=e,t.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:cs,lastRenderedState:e},t}function ys(e,t,n,r){return e.baseState=n,us(e,K,typeof r==`function`?r:cs)}function bs(e,t,n,r,a){if(cc(e))throw Error(i(485));if(e=t.action,e!==null){var o={payload:a,action:e,next:null,isTransition:!0,status:`pending`,value:null,reason:null,listeners:[],then:function(e){o.listeners.push(e)}};L.T===null?o.isTransition=!1:n(!0),r(o),n=t.pending,n===null?(o.next=t.pending=o,xs(t,o)):(o.next=n.next,t.pending=n.next=o)}}function xs(e,t){var n=t.action,r=t.payload,i=e.state;if(t.isTransition){var a=L.T,o={};o.types=a===null?null:a.types,L.T=o;try{var s=n(i,r),c=L.S;c!==null&&c(o,s),Ss(e,t,s)}catch(n){ws(e,t,n)}finally{a!==null&&o.types!==null&&(a.types=o.types),L.T=a}}else try{a=n(i,r),Ss(e,t,a)}catch(n){ws(e,t,n)}}function Ss(e,t,n){typeof n==`object`&&n&&typeof n.then==`function`?n.then(function(n){Cs(e,t,n)},function(n){return ws(e,t,n)}):Cs(e,t,n)}function Cs(e,t,n){t.status=`fulfilled`,t.value=n,Ts(t),e.state=n,t=e.pending,t!==null&&(n=t.next,n===t?e.pending=null:(n=n.next,t.next=n,xs(e,n)))}function ws(e,t,n){var r=e.pending;if(e.pending=null,r!==null){r=r.next;do t.status=`rejected`,t.reason=n,Ts(t),t=t.next;while(t!==r)}e.action=null}function Ts(e){e=e.listeners;for(var t=0;t<e.length;t++)(0,e[t])()}function Es(e,t){return t}function Ds(e,t){if(W){var n=Qu.formState;if(n!==null){a:{var r=G;if(W){if(ea){b:{for(var i=ea,a=na;i.nodeType!==8;){if(!a){i=null;break b}if(i=lm(i.nextSibling),i===null){i=null;break b}}a=i.data,i=a===`F!`||a===`F`?i:null}if(i){ea=lm(i.nextSibling),r=i.data===`F!`;break a}}ia(r)}r=!1}r&&(t=n[0])}}return n=ns(),n.memoizedState=n.baseState=t,r={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Es,lastRenderedState:t},n.queue=r,n=ac.bind(null,G,r),r.dispatch=n,r=vs(!1),a=sc.bind(null,G,!1,r.queue),r=ns(),i={state:t,dispatch:null,action:e,pending:null},r.queue=i,n=bs.bind(null,G,i,a,n),i.dispatch=n,r.memoizedState=e,[t,n,!1]}function Os(e){return ks(rs(),K,e)}function ks(e,t,n){if(t=us(e,t,Es)[0],e=ls(cs)[0],typeof t==`object`&&t&&typeof t.then==`function`)try{var r=as(t)}catch(e){throw e===Ga?qa:e}else r=t;t=rs();var i=t.queue,a=i.dispatch;return n!==t.memoizedState&&(G.flags|=2048,Ms(9,{destroy:void 0},As.bind(null,i,n),null)),[r,a,e]}function As(e,t){e.action=t}function js(e){var t=rs(),n=K;if(n!==null)return ks(t,n,e);rs(),t=t.memoizedState,n=rs();var r=n.queue.dispatch;return n.memoizedState=e,[t,r,!1]}function Ms(e,t,n,r){return e={tag:e,create:n,deps:r,inst:t,next:null},t=G.updateQueue,t===null&&(t=is(),G.updateQueue=t),n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e),e}function Ns(){return rs().memoizedState}function Ps(e,t,n,r){var i=ns();G.flags|=e,i.memoizedState=Ms(1|t,{destroy:void 0},n,r===void 0?null:r)}function Fs(e,t,n,r){var i=rs();r=r===void 0?null:r;var a=i.memoizedState.inst;K!==null&&r!==null&&Jo(r,K.memoizedState.deps)?i.memoizedState=Ms(t,a,n,r):(G.flags|=e,i.memoizedState=Ms(1|t,a,n,r))}function Is(e,t){Ps(8390656,8,e,t)}function Ls(e,t){Fs(2048,8,e,t)}function Rs(e){G.flags|=4;var t=G.updateQueue;if(t===null)t=is(),G.updateQueue=t,t.events=[e];else{var n=t.events;n===null?t.events=[e]:n.push(e)}}function zs(e){var t=rs().memoizedState;return Rs({ref:t,nextImpl:e}),function(){if(J&2)throw Error(i(440));return t.impl.apply(void 0,arguments)}}function Bs(e,t){return Fs(4,2,e,t)}function Vs(e,t){return Fs(4,4,e,t)}function Hs(e,t){if(typeof t==`function`){e=e();var n=t(e);return function(){typeof n==`function`?n():t(null)}}if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Us(e,t,n){n=n==null?null:n.concat([e]),Fs(4,4,Hs.bind(null,t,e),n)}function Ws(){}function Gs(e,t){var n=rs();t=t===void 0?null:t;var r=n.memoizedState;return t!==null&&Jo(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Ks(e,t){var n=rs();t=t===void 0?null:t;var r=n.memoizedState;if(t!==null&&Jo(t,r[1]))return r[0];if(r=e(),Ho){Je(!0);try{e()}finally{Je(!1)}}return n.memoizedState=[r,t],r}function qs(e,t,n){return n===void 0||Ro&1073741824&&!(X&261930)?e.memoizedState=t:(e.memoizedState=n,e=jd(),G.lanes|=e,ad|=e,n)}function Js(e,t,n,r){return Lr(n,t)?n:So.current===null?!(Ro&106)||Ro&1073741824&&!(X&261930)?(Ac=!0,e.memoizedState=n):(e=jd(),G.lanes|=e,ad|=e,t):(e=qs(e,n,r),Lr(e,t)||(Ac=!0),e)}function Ys(e,t,n,r,i){var a=R.p;R.p=a!==0&&8>a?a:8;var o=L.T,s={};s.types=o===null?null:o.types,L.T=s,sc(e,!1,t,n);try{var c=i(),l=L.S;l!==null&&l(s,c),typeof c==`object`&&c&&typeof c.then==`function`?oc(e,t,za(c,r),Ad(e)):oc(e,t,r,Ad(e))}catch(n){oc(e,t,{then:function(){},status:`rejected`,reason:n},Ad())}finally{R.p=a,o!==null&&s.types!==null&&(o.types=s.types),L.T=o}}function Xs(){}function Zs(e,t,n,r){if(e.tag!==5)throw Error(i(476));var a=Qs(e).queue;Ys(e,a,t,me,n===null?Xs:function(){return $s(e),n(r)})}function Qs(e){var t=e.memoizedState;if(t!==null)return t;t={memoizedState:me,baseState:me,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:cs,lastRenderedState:me},next:null};var n={};return t.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:cs,lastRenderedState:n},next:null},e.memoizedState=t,e=e.alternate,e!==null&&(e.memoizedState=t),t}function $s(e){var t=Qs(e);t.next===null&&(t=e.alternate.memoizedState),oc(e,t.next.queue,{},Ad())}function ec(){return xa(sh)}function tc(){return rs().memoizedState}function nc(){return rs().memoizedState}function rc(e){for(var t=e.return;t!==null;){switch(t.tag){case 24:case 3:var n=Ad();e=po(n);var r=mo(t,e,n);r!==null&&(Nd(r,t,n),ho(r,t,n)),t={cache:Oa()},e.payload=t;return}t=t.return}}function ic(e,t,n){var r=Ad();n={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},cc(e)?lc(t,n):(n=Si(e,t,n,r),n!==null&&(Nd(n,e,r),uc(n,t,r)))}function ac(e,t,n){oc(e,t,n,Ad())}function oc(e,t,n,r){var i={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(cc(e))lc(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,Lr(s,o))return xi(e,t,i,0),Qu===null&&bi(),!1}catch{}if(n=Si(e,t,i,r),n!==null)return Nd(n,e,r),uc(n,t,r),!0}return!1}function sc(e,t,n,r){if(r={lane:2,revertLane:Pf(),gesture:null,action:r,hasEagerState:!1,eagerState:null,next:null},cc(e)){if(t)throw Error(i(479))}else t=Si(e,n,r,2),t!==null&&Nd(t,e,2)}function cc(e){var t=e.alternate;return e===G||t!==null&&t===G}function lc(e,t){Vo=Bo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function uc(e,t,n){if(n&4194048){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,ft(e,n)}}var dc={readContext:xa,use:os,useCallback:qo,useContext:qo,useEffect:qo,useImperativeHandle:qo,useLayoutEffect:qo,useInsertionEffect:qo,useMemo:qo,useReducer:qo,useRef:qo,useState:qo,useDebugValue:qo,useDeferredValue:qo,useTransition:qo,useSyncExternalStore:qo,useId:qo,useHostTransitionStatus:qo,useFormState:qo,useActionState:qo,useOptimistic:qo,useMemoCache:qo,useCacheRefresh:qo,useEffectEvent:qo},fc={readContext:xa,use:os,useCallback:function(e,t){return ns().memoizedState=[e,t===void 0?null:t],e},useContext:xa,useEffect:Is,useImperativeHandle:function(e,t,n){n=n==null?null:n.concat([e]),Ps(4194308,4,Hs.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Ps(4194308,4,e,t)},useInsertionEffect:function(e,t){Ps(4,2,e,t)},useMemo:function(e,t){var n=ns();t=t===void 0?null:t;var r=e();if(Ho){Je(!0);try{e()}finally{Je(!1)}}return n.memoizedState=[r,t],r},useReducer:function(e,t,n){var r=ns();if(n!==void 0){var i=n(t);if(Ho){Je(!0);try{n(t)}finally{Je(!1)}}}else i=t;return r.memoizedState=r.baseState=i,e={pending:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:i},r.queue=e,e=e.dispatch=ic.bind(null,G,e),[r.memoizedState,e]},useRef:function(e){var t=ns();return e={current:e},t.memoizedState=e},useState:function(e){e=vs(e);var t=e.queue,n=ac.bind(null,G,t);return t.dispatch=n,[e.memoizedState,n]},useDebugValue:Ws,useDeferredValue:function(e,t){return qs(ns(),e,t)},useTransition:function(){var e=vs(!1);return e=Ys.bind(null,G,e.queue,!0,!1),ns().memoizedState=e,[!1,e]},useSyncExternalStore:function(e,t,n){var r=G,a=ns();if(W){if(n===void 0)throw Error(i(407));n=n()}else{if(n=t(),Qu===null)throw Error(i(349));X&127||ps(r,t,n)}a.memoizedState=n;var o={value:n,getSnapshot:t};return a.queue=o,Is(hs.bind(null,r,o,e),[e]),r.flags|=2048,Ms(9,{destroy:void 0},ms.bind(null,r,o,n,t),null),n},useId:function(){var e=ns(),t=Qu.identifierPrefix;if(W){var n=qi,r=Ki;n=(r&~(1<<32-Ye(r)-1)).toString(32)+n,t=`_`+t+`R_`+n,n=Uo++,0<n&&(t+=`H`+n.toString(32)),t+=`_`}else n=Ko++,t=`_`+t+`r_`+n.toString(32)+`_`;return e.memoizedState=t},useHostTransitionStatus:ec,useFormState:Ds,useActionState:Ds,useOptimistic:function(e){var t=ns();t.memoizedState=t.baseState=e;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return t.queue=n,t=sc.bind(null,G,!0,n),n.dispatch=t,[e,t]},useMemoCache:ss,useCacheRefresh:function(){return ns().memoizedState=rc.bind(null,G)},useEffectEvent:function(e){var t=ns(),n={impl:e};return t.memoizedState=n,function(){if(J&2)throw Error(i(440));return n.impl.apply(void 0,arguments)}}},pc={readContext:xa,use:os,useCallback:Gs,useContext:xa,useEffect:Ls,useImperativeHandle:Us,useInsertionEffect:Bs,useLayoutEffect:Vs,useMemo:Ks,useReducer:ls,useRef:Ns,useState:function(){return ls(cs)},useDebugValue:Ws,useDeferredValue:function(e,t){return Js(rs(),K.memoizedState,e,t)},useTransition:function(){var e=ls(cs)[0],t=rs().memoizedState;return[typeof e==`boolean`?e:as(e),t]},useSyncExternalStore:fs,useId:tc,useHostTransitionStatus:ec,useFormState:Os,useActionState:Os,useOptimistic:function(e,t){return ys(rs(),K,e,t)},useMemoCache:ss,useCacheRefresh:nc,useEffectEvent:zs},mc={readContext:xa,use:os,useCallback:Gs,useContext:xa,useEffect:Ls,useImperativeHandle:Us,useInsertionEffect:Bs,useLayoutEffect:Vs,useMemo:Ks,useReducer:ds,useRef:Ns,useState:function(){return ds(cs)},useDebugValue:Ws,useDeferredValue:function(e,t){var n=rs();return K===null?qs(n,e,t):Js(n,K.memoizedState,e,t)},useTransition:function(){var e=ds(cs)[0],t=rs().memoizedState;return[typeof e==`boolean`?e:as(e),t]},useSyncExternalStore:fs,useId:tc,useHostTransitionStatus:ec,useFormState:js,useActionState:js,useOptimistic:function(e,t){var n=rs();return K===null?(n.baseState=e,[e,n.queue.dispatch]):ys(n,K,e,t)},useMemoCache:ss,useCacheRefresh:nc,useEffectEvent:zs};function hc(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:D({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var gc={enqueueSetState:function(e,t,n){e=e._reactInternals;var r=Ad(),i=po(r);i.payload=t,n!=null&&(i.callback=n),t=mo(e,i,r),t!==null&&(Nd(t,e,r),ho(t,e,r))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=Ad(),i=po(r);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=mo(e,i,r),t!==null&&(Nd(t,e,r),ho(t,e,r))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=Ad(),r=po(n);r.tag=2,t!=null&&(r.callback=t),t=mo(e,r,n),t!==null&&(Nd(t,e,n),ho(t,e,n))}};function _c(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate==`function`?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Rr(n,r)||!Rr(i,a):!0}function vc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps==`function`&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps==`function`&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&gc.enqueueReplaceState(t,t.state,null)}function yc(e,t){var n=t;if(`ref`in t)for(var r in n={},t)r!==`ref`&&(n[r]=t[r]);if(e=e.defaultProps)for(var i in n===t&&(n=D({},n)),e)n[i]===void 0&&(n[i]=e[i]);return n}function bc(e){gi(e)}function xc(e){console.error(e)}function Sc(e){gi(e)}function Cc(e,t){try{var n=e.onUncaughtError;n(t.value,{componentStack:t.stack})}catch(e){setTimeout(function(){throw e})}}function wc(e,t,n){try{var r=e.onCaughtError;r(n.value,{componentStack:n.stack,errorBoundary:t.tag===1?t.stateNode:null})}catch(e){setTimeout(function(){throw e})}}function Tc(e,t,n){return n=po(n),n.tag=3,n.payload={element:null},n.callback=function(){Cc(e,t)},n}function Ec(e){return e=po(e),e.tag=3,e}function Dc(e,t,n,r){var i=n.type.getDerivedStateFromError;if(typeof i==`function`){var a=r.value;e.payload=function(){return i(a)},e.callback=function(){wc(t,n,r)}}var o=n.stateNode;o!==null&&typeof o.componentDidCatch==`function`&&(e.callback=function(){wc(t,n,r),typeof i!=`function`&&(_d===null?_d=new Set([this]):_d.add(this));var e=r.stack;this.componentDidCatch(r.value,{componentStack:e===null?``:e})})}function Oc(e,t,n,r,a){if(n.flags|=32768,typeof r==`object`&&r&&typeof r.then==`function`){if(t=n.alternate,t!==null&&va(t,n,a,!0),n=Do.current,n!==null){switch(n.tag){case 31:case 13:case 19:return Oo===null?Gd():n.alternate===null&&id===0&&(id=3),n.flags&=-257,n.flags|=65536,n.lanes=a,r===Ja?n.flags|=16384:(t=n.updateQueue,t===null?n.updateQueue=new Set([r]):t.add(r),mf(e,r,a)),!1;case 22:return n.flags|=65536,r===Ja?n.flags|=16384:(t=n.updateQueue,t===null?(t={transitions:null,markerInstances:null,retryQueue:new Set([r])},n.updateQueue=t):(n=t.retryQueue,n===null?t.retryQueue=new Set([r]):n.add(r)),mf(e,r,a)),!1}throw Error(i(435,n.tag))}return mf(e,r,a),Gd(),!1}if(W)return t=Do.current,t===null?(r!==ra&&(t=Error(i(423),{cause:r}),ua(Ri(t,n))),e=e.current.alternate,e.flags|=65536,a&=-a,e.lanes|=a,r=Ri(r,n),a=Tc(e.stateNode,r,a),go(e,a),id!==4&&(id=2)):(!(t.flags&65536)&&(t.flags|=256),t.flags|=65536,t.lanes=a,r!==ra&&(e=Error(i(422),{cause:r}),ua(Ri(e,n)))),!1;var o=Error(i(520),{cause:r});if(o=Ri(o,n),ud===null?ud=[o]:ud.push(o),id!==4&&(id=2),t===null)return!0;r=Ri(r,n),n=t;do{switch(n.tag){case 3:return n.flags|=65536,e=a&-a,n.lanes|=e,e=Tc(n.stateNode,r,e),go(n,e),!1;case 1:if(t=n.type,o=n.stateNode,!(n.flags&128)&&(typeof t.getDerivedStateFromError==`function`||o!==null&&typeof o.componentDidCatch==`function`&&(_d===null||!_d.has(o))))return n.flags|=65536,a&=-a,n.lanes|=a,a=Ec(a),Dc(a,e,n,r),go(n,a),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var kc=Error(i(461)),Ac=!1;function jc(e,t,n,r){t.child=e===null?co(t,null,n,r):so(t,e.child,n,r)}function Mc(e,t,n,r,i){n=n.render;var a=t.ref;if(`ref`in r){var o={};for(var s in r)s!==`ref`&&(o[s]=r[s])}else o=r;return ba(t),r=Yo(e,t,n,o,a,i),s=$o(),e!==null&&!Ac?(es(e,t,i),ol(e,t,i)):(W&&s&&Xi(t),t.flags|=1,jc(e,t,r,i),t.child)}function Nc(e,t,n,r,i){if(e===null){var a=n.type;return typeof a==`function`&&!ki(a)&&a.defaultProps===void 0&&n.compare===null?(t.tag=15,t.type=a,Pc(e,t,a,r,i)):(e=Mi(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!sl(e,i)){var o=a.memoizedProps;if(n=n.compare,n=n===null?Rr:n,n(o,r)&&e.ref===t.ref)return ol(e,t,i)}return t.flags|=1,e=Ai(a,r),e.ref=t.ref,e.return=t,t.child=e}function Pc(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Rr(a,r)&&e.ref===t.ref){if(Ac=!1,t.pendingProps=r=a,sl(e,i))e.flags&131072&&(Ac=!0);else return t.lanes=e.lanes,ol(e,t,i)}}return Hc(e,t,n,r,i)}function Fc(e,t,n,r){var i=r.children,a=e===null?null:e.memoizedState;if(e===null&&t.stateNode===null&&(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),r.mode===`hidden`){if(t.flags&128){if(a=a===null?n:a.baseLanes|n,e!==null){for(r=t.child=e.child,i=0;r!==null;)i=i|r.lanes|r.childLanes,r=r.sibling;r=i&~a}else r=0,t.child=null;return Lc(e,t,a,n,r)}if(n&536870912)t.memoizedState={baseLanes:0,cachePool:null},e!==null&&Ua(t,a===null?null:a.cachePool),a===null?To():wo(t,a),jo(t);else return r=t.lanes=536870912,Lc(e,t,a===null?n:a.baseLanes|n,n,r)}else a===null?(e!==null&&Ua(t,null),To(),Mo()):(Ua(t,a.cachePool),wo(t,a),Mo(),t.memoizedState=null);return jc(e,t,i,n),t.child}function Ic(e,t){return e!==null&&e.tag===22||t.stateNode!==null||(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),t.sibling}function Lc(e,t,n,r,i){var a=Ha();return a=a===null?null:{parent:Da._currentValue,pool:a},t.memoizedState={baseLanes:n,cachePool:a},e!==null&&Ua(t,null),To(),jo(t),e!==null&&va(e,t,r,!0),t.childLanes=i,null}function Rc(e,t){return t=Zc({mode:t.mode,children:t.children},e.mode),t.ref=e.ref,e.child=t,t.return=e,t}function zc(e,t,n){return so(t,e.child,null,n),e=Rc(t,t.pendingProps),e.flags|=2,No(t),t.memoizedState=null,e}function Bc(e,t,n){var r=t.pendingProps,a=!!(t.flags&128);if(t.flags&=-129,e===null){if(W){if(r.mode===`hidden`)return e=Rc(t,r),t.lanes=536870912,e.memoizedState={baseLanes:0,cachePool:null},Ic(null,e);if(Ao(t),(e=ea)?(e=am(e,na),e=e!==null&&e.data===`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Gi===null?null:{id:Ki,overflow:qi},retryLane:536870912,hydrationErrors:null},n=Fi(e),n.return=t,t.child=n,$i=t,ea=null)):e=null,e===null)throw ia(t);return t.lanes=536870912,null}return Rc(t,r)}var o=e.memoizedState;if(o!==null){var s=o.dehydrated;if(Ao(t),a){if(t.flags&256)t.flags&=-257,t=zc(e,t,n);else if(t.memoizedState!==null)t.child=e.child,t.flags|=128,t=null;else throw Error(i(558))}else if(Ac||va(e,t,n,!1),a=(n&e.childLanes)!==0,Ac||a){if(So.current===null){if(r=Qu,r!==null&&(s=pt(r,n),s!==0&&s!==o.retryLane))throw o.retryLane=s,Ci(e,s),Nd(r,e,s),kc;Gd()}t=zc(e,t,n)}else e=o.treeContext,ea=lm(s.nextSibling),$i=t,W=!0,ta=null,na=!1,e!==null&&Qi(t,e),t=Rc(t,r),t.flags|=134221824;return t}return e=Ai(e.child,{mode:r.mode,children:r.children}),e.ref=t.ref,t.child=e,e.return=t,e}function Vc(e,t){var n=t.ref;if(n===null)e!==null&&e.ref!==null&&(t.flags|=4194816);else{if(typeof n!=`function`&&typeof n!=`object`)throw Error(i(284));(e===null||e.ref!==n)&&(t.flags|=4194816)}}function Hc(e,t,n,r,i){return ba(t),n=Yo(e,t,n,r,void 0,i),r=$o(),e!==null&&!Ac?(es(e,t,i),ol(e,t,i)):(W&&r&&Xi(t),t.flags|=1,jc(e,t,n,i),t.child)}function Uc(e,t,n,r,i,a){return ba(t),t.updateQueue=null,n=Zo(t,r,n,i),Xo(e),r=$o(),e!==null&&!Ac?(es(e,t,a),ol(e,t,a)):(W&&r&&Xi(t),t.flags|=1,jc(e,t,n,a),t.child)}function Wc(e,t,n,r,i){if(ba(t),t.stateNode===null){var a=Ei,o=n.contextType;typeof o==`object`&&o&&(a=xa(o)),a=new n(r,a),t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,a.updater=gc,t.stateNode=a,a._reactInternals=t,a=t.stateNode,a.props=r,a.state=t.memoizedState,a.refs={},uo(t),o=n.contextType,a.context=typeof o==`object`&&o?xa(o):Ei,a.state=t.memoizedState,o=n.getDerivedStateFromProps,typeof o==`function`&&(hc(t,n,o,r),a.state=t.memoizedState),typeof n.getDerivedStateFromProps==`function`||typeof a.getSnapshotBeforeUpdate==`function`||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(o=a.state,typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount(),o!==a.state&&gc.enqueueReplaceState(a,a.state,null),yo(t,r,a,i),vo(),a.state=t.memoizedState),typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!0}else if(e===null){a=t.stateNode;var s=t.memoizedProps,c=yc(n,s);a.props=c;var l=a.context,u=n.contextType;o=Ei,typeof u==`object`&&u&&(o=xa(u));var d=n.getDerivedStateFromProps;u=typeof d==`function`||typeof a.getSnapshotBeforeUpdate==`function`,s=t.pendingProps!==s,u||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(s||l!==o)&&vc(t,a,r,o),lo=!1;var f=t.memoizedState;a.state=f,yo(t,r,a,i),vo(),l=t.memoizedState,s||f!==l||lo?(typeof d==`function`&&(hc(t,n,d,r),l=t.memoizedState),(c=lo||_c(t,n,c,r,f,l,o))?(u||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount==`function`&&(t.flags|=4194308)):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=o,r=c):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,fo(e,t),o=t.memoizedProps,u=yc(n,o),a.props=u,d=t.pendingProps,f=a.context,l=n.contextType,c=Ei,typeof l==`object`&&l&&(c=xa(l)),s=n.getDerivedStateFromProps,(l=typeof s==`function`||typeof a.getSnapshotBeforeUpdate==`function`)||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(o!==d||f!==c)&&vc(t,a,r,c),lo=!1,f=t.memoizedState,a.state=f,yo(t,r,a,i),vo();var p=t.memoizedState;o!==d||f!==p||lo||e!==null&&e.dependencies!==null&&ya(e.dependencies)?(typeof s==`function`&&(hc(t,n,s,r),p=t.memoizedState),(u=lo||_c(t,n,u,r,f,p,c)||e!==null&&e.dependencies!==null&&ya(e.dependencies))?(l||typeof a.UNSAFE_componentWillUpdate!=`function`&&typeof a.componentWillUpdate!=`function`||(typeof a.componentWillUpdate==`function`&&a.componentWillUpdate(r,p,c),typeof a.UNSAFE_componentWillUpdate==`function`&&a.UNSAFE_componentWillUpdate(r,p,c)),typeof a.componentDidUpdate==`function`&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate==`function`&&(t.flags|=1024)):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=p),a.props=r,a.state=p,a.context=c,r=u):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return a=r,Vc(e,t),r=!!(t.flags&128),a||r?(a=t.stateNode,n=r&&typeof n.getDerivedStateFromError!=`function`?null:a.render(),t.flags|=1,e!==null&&r?(t.child=so(t,e.child,null,i),t.child=so(t,null,n,i)):jc(e,t,n,i),t.memoizedState=a.state,e=t.child):e=ol(e,t,i),e}function Gc(e,t,n,r){return ca(),t.flags|=256,jc(e,t,n,r),t.child}var Kc={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function qc(e){return{baseLanes:e,cachePool:Wa()}}function Jc(e,t,n){return e=e===null?0:e.childLanes&~n,t&&(e|=cd),e}function Yc(e,t,n){var r=t.pendingProps,i=!1,a=!!(t.flags&128),o;if((o=a)||(o=e!==null&&e.memoizedState===null?!1:!!(Po.current&2)),o&&(i=!0,t.flags&=-129),o=!!(t.flags&32),t.flags&=-33,e===null){if(W){if(i?ko(t):Mo(),(e=ea)?(e=am(e,na),e=e!==null&&e.data!==`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:Gi===null?null:{id:Ki,overflow:qi},retryLane:536870912,hydrationErrors:null},n=Fi(e),n.return=t,t.child=n,$i=t,ea=null)):e=null,e===null)throw ia(t);return t.lanes=sm(e)?32:536870912,null}return a=r.children,r=r.fallback,i?(Mo(),i=t.mode,a=Zc({mode:`hidden`,children:a},i),r=Ni(r,i,n,null),a.return=t,r.return=t,a.sibling=r,t.child=a,r=t.child,r.memoizedState=qc(n),r.childLanes=Jc(e,o,n),t.memoizedState=Kc,Ic(null,r)):(ko(t),Xc(t,a))}var s=e.memoizedState;if(s!==null){var c=s.dehydrated;if(c!==null)return $c(e,t,a,o,r,c,s,n)}return i?(Mo(),i=r.fallback,a=t.mode,s=e.child,c=s.sibling,r=Ai(s,{mode:`hidden`,children:r.children}),r.subtreeFlags=s.subtreeFlags&1206910976,c===null?(i=Ni(i,a,n,null),i.flags|=2):i=Ai(c,i),i.return=t,r.return=t,r.sibling=i,t.child=r,Ic(null,r),r=t.child,i=e.child.memoizedState,i===null?i=qc(n):(a=i.cachePool,a===null?a=Wa():(s=Da._currentValue,a=a.parent===s?a:{parent:s,pool:s}),i={baseLanes:i.baseLanes|n,cachePool:a}),r.memoizedState=i,r.childLanes=Jc(e,o,n),t.memoizedState=Kc,Ic(e.child,r)):(ko(t),n=e.child,e=n.sibling,n=Ai(n,{mode:`visible`,children:r.children}),n.return=t,n.sibling=null,e!==null&&(o=t.deletions,o===null?(t.deletions=[e],t.flags|=16):o.push(e)),t.child=n,t.memoizedState=null,n)}function Xc(e,t){return t=Zc({mode:`visible`,children:t},e.mode),t.return=e,e.child=t}function Zc(e,t){return e=Oi(22,e,null,t),e.lanes=0,e}function Qc(e,t,n){return so(t,e.child,null,n),e=Xc(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function $c(e,t,n,r,a,o,s,c){if(n)return t.flags&256?(ko(t),t.flags&=-257,Qc(e,t,c)):t.memoizedState===null?(Mo(),o=a.fallback,s=t.mode,a=Zc({mode:`visible`,children:a.children},s),o=Ni(o,s,c,null),o.flags|=2,a.return=t,o.return=t,a.sibling=o,t.child=a,so(t,e.child,null,c),a=t.child,a.memoizedState=qc(c),a.childLanes=Jc(e,r,c),t.memoizedState=Kc,Ic(null,a)):(Mo(),t.child=e.child,t.flags|=128,null);if(ko(t),sm(o)){if(r=o.nextSibling&&o.nextSibling.dataset,r)var l=r.dgst;return r=l,r!==``&&(a=Error(i(419)),a.stack=``,a.digest=r,ua({value:a,source:null,stack:null})),Qc(e,t,c)}if(Ac||va(e,t,c,!1),r=(c&e.childLanes)!==0,Ac||r){if(So.current!==null)return Qc(e,t,c);if(r=Qu,r!==null&&(a=pt(r,c),a!==0&&a!==s.retryLane))throw s.retryLane=a,Ci(e,a),Nd(r,e,a),kc;return om(o)||Gd(),Qc(e,t,c)}return om(o)?(t.flags|=192,t.child=e.child,null):(e=s.treeContext,ea=lm(o.nextSibling),$i=t,W=!0,ta=null,na=!1,e!==null&&Qi(t,e),t=Xc(t,a.children),t.flags|=134221824,t)}function el(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),ga(e.return,t,n)}function tl(e){for(var t=null;e!==null;){var n=e.alternate;n!==null&&Lo(n)===null&&(t=e),e=e.sibling}return t}function nl(e,t,n,r,i,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i,treeForkCount:a}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i,o.treeForkCount=a)}function rl(e){var t=e.child;for(e.child=null;t!==null;){var n=t.sibling;t.sibling=e.child,e.child=t,t=n}}function il(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;r=r.children;var o=Po.current;if(t.flags&128)return Fo(t,o),null;var s=!!(o&2);if(s?(o=o&1|2,t.flags|=128):o&=1,Fo(t,o),i===`backwards`&&e!==null?(rl(e),jc(e,t,r,n),rl(e)):jc(e,t,r,n),r=W?Hi:0,!s&&e!==null&&e.flags&128)a:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&el(e,n,t);else if(e.tag===19)el(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break a;for(;e.sibling===null;){if(e.return===null||e.return===t)break a;e=e.return}e.sibling.return=e.return,e=e.sibling}switch(i){case`backwards`:n=tl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null,rl(t)),nl(t,!0,i,null,a,r);break;case`unstable_legacy-backwards`:for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Lo(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}nl(t,!0,n,null,a,r);break;case`together`:nl(t,!1,null,null,void 0,r);break;case`independent`:t.memoizedState=null;break;default:n=tl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),nl(t,!1,i,n,a,r)}return t.child}function al(e,t,n){var r=t.pendingProps;return ma(t,t.type,r.value),jc(e,t,r.children,n),t.child}function ol(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),ad|=t.lanes,(n&t.childLanes)===0){if(e!==null){if(va(e,t,n,!1),(n&t.childLanes)===0)return null}else return null}if(e!==null&&t.child!==e.child)throw Error(i(153));if(t.child!==null){for(e=t.child,n=Ai(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=Ai(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function sl(e,t){return(e.lanes&t)!==0||(e=e.dependencies,!!(e!==null&&ya(e)))}function cl(e,t,n){switch(t.tag){case 3:Ce(t,t.stateNode.containerInfo),ma(t,Da,e.memoizedState.cache),ca();break;case 27:case 5:Te(t);break;case 4:Ce(t,t.stateNode.containerInfo);break;case 10:ma(t,t.type,t.memoizedProps.value);break;case 31:if(t.memoizedState!==null)return t.flags|=128,Ao(t),null;break;case 13:var r=t.memoizedState;if(r!==null){if(r.dehydrated!==null)return ko(t),t.flags|=128,null;r=va(e,t,n,!1);var i=t.child.childLanes;return r||(n&i)!==0?Yc(e,t,n):(ko(t),e=ol(e,t,n),e===null?null:e.sibling)}ko(t);break;case 19:if(t.flags&128)return il(e,t,n);if(i=!!(e.flags&128),r=(n&t.childLanes)!==0,r||=(va(e,t,n,!1),(n&t.childLanes)!==0),i){if(r)return il(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),Fo(t,Po.current),r)break;return null;case 22:return t.lanes=0,Fc(e,t,n,t.pendingProps);case 24:ma(t,Da,e.memoizedState.cache)}return ol(e,t,n)}function ll(e,t,n){if(e!==null){if(e.memoizedProps!==t.pendingProps)Ac=!0;else{if(!sl(e,n)&&!(t.flags&128))return Ac=!1,cl(e,t,n);Ac=!!(e.flags&131072)}}else Ac=!1,W&&t.flags&1048576&&Yi(t,Hi,t.index);switch(t.lanes=0,t.tag){case 16:a:{var r=t.pendingProps;if(e=Za(t.elementType),t.type=e,typeof e==`function`)ki(e)?(r=yc(e,r),t.tag=1,t=Wc(null,t,e,r,n)):(t.tag=0,t=Hc(null,t,e,r,n));else{if(e!=null){var a=e.$$typeof;if(a===re){t.tag=11,t=Mc(null,t,e,r,n);break a}if(a===ae){t.tag=14,t=Nc(null,t,e,r,n);break a}if(a===M){t.tag=10,t.type=e,t=al(null,t,n);break a}}throw t=fe(e)||e,Error(i(306,t,``))}}return t;case 0:return Hc(e,t,t.type,t.pendingProps,n);case 1:return r=t.type,a=yc(r,t.pendingProps),Wc(e,t,r,a,n);case 3:a:{if(Ce(t,t.stateNode.containerInfo),e===null)throw Error(i(387));r=t.pendingProps;var o=t.memoizedState;a=o.element,fo(e,t),yo(t,r,null,n);var s=t.memoizedState;if(r=s.cache,ma(t,Da,r),r!==o.cache&&_a(t,[Da],n,!0),vo(),r=s.element,o.isDehydrated){if(o={element:r,isDehydrated:!1,cache:s.cache},t.updateQueue.baseState=o,t.memoizedState=o,t.flags&256){t=Gc(e,t,r,n);break a}if(r!==a){a=Ri(Error(i(424)),t),ua(a),t=Gc(e,t,r,n);break a}switch(e=t.stateNode.containerInfo,e.nodeType){case 9:e=e.body;break;default:e=e.nodeName===`HTML`?e.ownerDocument.body:e}for(ea=lm(e.firstChild),$i=t,W=!0,ta=null,na=!0,n=co(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling}else{if(ca(),r===a){t=ol(e,t,n);break a}jc(e,t,r,n)}t=t.child}return t;case 26:return Vc(e,t),e===null?(n=Nm(t.type,null,t.pendingProps,null))?t.memoizedState=n:W||(t.stateNode=fp(t.type,t.pendingProps,xe.current,t)):t.memoizedState=Nm(t.type,e.memoizedProps,t.pendingProps,e.memoizedState),null;case 27:return Te(t),e===null&&W&&(r=t.stateNode=hm(t.type,t.pendingProps,xe.current),$i=t,na=!0,a=ea,Sp(t.type)?(um=a,ea=lm(r.firstChild)):ea=a),jc(e,t,t.pendingProps.children,n),Vc(e,t),e===null&&(t.flags|=4194304),t.child;case 5:return e===null&&W&&((a=r=ea)&&(r=rm(r,t.type,t.pendingProps,na),r===null?a=!1:(t.stateNode=r,$i=t,ea=lm(r.firstChild),na=!1,a=!0)),a||ia(t)),Te(t),a=t.type,o=t.pendingProps,s=e===null?null:e.memoizedProps,r=o.children,pp(a,o)?r=null:s!==null&&pp(a,s)&&(t.flags|=32),t.memoizedState!==null&&(a=Yo(e,t,Qo,null,null,n),sh._currentValue=a),Vc(e,t),jc(e,t,r,n),t.child;case 6:return e===null&&W&&((e=n=ea)&&(n=im(n,t.pendingProps,na),n===null?e=!1:(t.stateNode=n,$i=t,ea=null,e=!0)),e||ia(t)),null;case 13:return Yc(e,t,n);case 4:return Ce(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=so(t,null,r,n):jc(e,t,r,n),t.child;case 11:return Mc(e,t,t.type,t.pendingProps,n);case 7:return r=t.pendingProps,Vc(e,t),jc(e,t,r,n),t.child;case 8:return jc(e,t,t.pendingProps.children,n),t.child;case 12:return jc(e,t,t.pendingProps.children,n),t.child;case 10:return al(e,t,n);case 9:return a=t.type._context,r=t.pendingProps.children,ba(t),a=xa(a),r=r(a),t.flags|=1,jc(e,t,r,n),t.child;case 14:return Nc(e,t,t.type,t.pendingProps,n);case 15:return Pc(e,t,t.type,t.pendingProps,n);case 19:return il(e,t,n);case 31:return Bc(e,t,n);case 22:return Fc(e,t,n,t.pendingProps);case 24:return ba(t),r=xa(Da),e===null?(a=Ha(),a===null&&(a=Qu,o=Oa(),a.pooledCache=o,o.refCount++,o!==null&&(a.pooledCacheLanes|=n),a=o),t.memoizedState={parent:r,cache:a},uo(t),ma(t,Da,a)):((e.lanes&n)!==0&&(fo(e,t),yo(t,null,null,n),vo()),a=e.memoizedState,o=t.memoizedState,a.parent===r?(r=o.cache,ma(t,Da,r),r!==a.cache&&_a(t,[Da],n,!0)):(a={parent:r,cache:r},t.memoizedState=a,t.lanes===0&&(t.memoizedState=t.updateQueue.baseState=a),ma(t,Da,r))),jc(e,t,t.pendingProps.children,n),t.child;case 30:return t.stateNode===null&&(t.stateNode={autoName:null,paired:null,clones:null,ref:null}),r=t.pendingProps,r.name!=null&&r.name!==`auto`?t.flags|=e===null?18882560:18874368:W&&Xi(t),e!==null&&e.memoizedProps.name!==r.name?t.flags|=4194816:Vc(e,t),jc(e,t,r.children,n),t.child;case 29:throw t.pendingProps}throw Error(i(156,t.tag))}function ul(e){e.flags|=4}function dl(e,t,n,r,i){var a;if((a=!!(e.mode&32))&&(a=n===null?Jm(t,r):Jm(t,r)&&(r.src!==n.src||r.srcSet!==n.srcSet)),a){if(e.flags|=16777216,(i&335544128)===i){if(e.stateNode.complete)e.flags|=8192;else if(Hd())e.flags|=8192;else throw Qa=Ja,Ka}}else e.flags&=-16777217}function fl(e,t){if(t.type!==`stylesheet`||t.state.loading&4)e.flags&=-16777217;else if(e.flags|=16777216,!Ym(t)){if(Hd())e.flags|=8192;else throw Qa=Ja,Ka}}function pl(e,t){t!==null&&(e.flags|=4),e.flags&16384&&(t=e.tag===22?536870912:st(),e.lanes|=t,ld|=t)}function ml(e,t){if(!W)switch(e.tailMode){case`visible`:break;case`collapsed`:for(var n=e.tail,r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null;break;default:for(t=e.tail,n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null}}function hl(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&1206910976,r|=i.flags&1206910976,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function gl(e,t,n){var r=t.pendingProps;switch(Zi(t),t.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return hl(t),null;case 1:return hl(t),null;case 3:return n=t.stateNode,r=null,e!==null&&(r=e.memoizedState.cache),t.memoizedState.cache!==r&&(t.flags|=2048),ha(Da),we(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(e===null||e.child===null)&&(sa(t)?ul(t):e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,la())),hl(t),null;case 26:var a=t.type,o=t.memoizedState;return e===null?(ul(t),o===null?(hl(t),dl(t,a,null,r,n)):(hl(t),fl(t,o))):o?o===e.memoizedState?(hl(t),t.flags&=-16777217):(ul(t),hl(t),fl(t,o)):(e=e.memoizedProps,e!==r&&ul(t),hl(t),dl(t,a,e,r,n)),null;case 27:if(Ee(t),n=xe.current,a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&ul(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return hl(t),t.subtreeFlags&=-33554433,null}e=ye.current,sa(t)?aa(t,e):(e=hm(a,r,n),t.stateNode=e,ul(t))}return hl(t),t.subtreeFlags&=-33554433,null;case 5:if(Ee(t),a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&ul(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return hl(t),t.subtreeFlags&=-33554433,null}if(o=ye.current,sa(t))aa(t,o);else{var s=lp(xe.current);switch(o){case 1:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case 2:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;default:switch(a){case`svg`:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case`math`:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;case`script`:o=s.createElement(`div`),o.innerHTML=`<script><\/script>`,o=o.removeChild(o.firstChild);break;case`select`:o=typeof r.is==`string`?s.createElement(`select`,{is:r.is}):s.createElement(`select`),r.multiple?o.multiple=!0:r.size&&(o.size=r.size);break;default:o=typeof r.is==`string`?s.createElement(a,{is:r.is}):s.createElement(a)}}o[yt]=t,o[bt]=r;a:for(s=t.child;s!==null;){if(s.tag===5||s.tag===6)o.appendChild(s.stateNode);else if(s.tag!==4&&s.tag!==27&&s.child!==null){s.child.return=s,s=s.child;continue}if(s===t)break a;for(;s.sibling===null;){if(s.return===null||s.return===t)break a;s=s.return}s.sibling.return=s.return,s=s.sibling}t.stateNode=o;a:switch(np(o,a,r),a){case`button`:case`input`:case`select`:case`textarea`:r=!!r.autoFocus;break a;case`img`:r=!0;break a;default:r=!1}r&&ul(t)}}return hl(t),t.subtreeFlags&=-33554433,dl(t,t.type,e===null?null:e.memoizedProps,t.pendingProps,n),null;case 6:if(e&&t.stateNode!=null)e.memoizedProps!==r&&ul(t);else{if(typeof r!=`string`&&t.stateNode===null)throw Error(i(166));if(e=xe.current,sa(t)){if(e=t.stateNode,n=t.memoizedProps,r=null,a=$i,a!==null)switch(a.tag){case 27:case 5:r=a.memoizedProps}e[yt]=t,e=!!(e.nodeValue===n||r!==null&&!0===r.suppressHydrationWarning||ep(e.nodeValue,n)),e||ia(t,!0)}else e=lp(e).createTextNode(r),e[yt]=t,t.stateNode=e}return hl(t),null;case 31:if(n=t.memoizedState,e===null||e.memoizedState!==null){if(r=sa(t),n!==null){if(e===null){if(!r)throw Error(i(318));if(e=t.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(557));e[yt]=t}else ca(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;hl(t),e=!1}else n=la(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=n),e=!0;if(!e)return t.flags&256?(No(t),t):(No(t),null);if(t.flags&128)throw Error(i(558))}return hl(t),null;case 13:if(r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(a=sa(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(i(318));if(a=t.memoizedState,a=a===null?null:a.dehydrated,!a)throw Error(i(317));a[yt]=t}else ca(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;hl(t),a=!1}else a=la(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=a),a=!0;if(!a)return t.flags&256?(No(t),t):(No(t),null)}return No(t),t.flags&128?(t.lanes=n,t):(n=r!==null,e=e!==null&&e.memoizedState!==null,n&&(r=t.child,a=null,r.alternate!==null&&r.alternate.memoizedState!==null&&r.alternate.memoizedState.cachePool!==null&&(a=r.alternate.memoizedState.cachePool.pool),o=null,r.memoizedState!==null&&r.memoizedState.cachePool!==null&&(o=r.memoizedState.cachePool.pool),o!==a&&(r.flags|=2048)),n!==e&&n&&(t.child.flags|=8192),pl(t,t.updateQueue),hl(t),null);case 4:return we(),e===null&&Wf(t.stateNode.containerInfo),t.flags|=67108864,hl(t),null;case 10:return ha(t.type),hl(t),null;case 19:if(Io(t),r=t.memoizedState,r===null)return hl(t),null;if(a=!!(t.flags&128),o=r.rendering,o===null){if(a)ml(r,!1);else{if(id!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=Lo(e),o!==null){for(t.flags|=128,ml(r,!1),e=o.updateQueue,t.updateQueue=e,pl(t,e),t.subtreeFlags=0,e=n,n=t.child;n!==null;)ji(n,e),n=n.sibling;return Fo(t,Po.current&1|2),W&&Ji(t,r.treeForkCount),t.child}e=e.sibling}r.tail!==null&&Le()>hd&&(t.flags|=128,a=!0,ml(r,!1),t.lanes=4194304)}}else{if(!a){if(e=Lo(o),e!==null){if(t.flags|=128,a=!0,e=e.updateQueue,t.updateQueue=e,pl(t,e),ml(r,!0),r.tail===null&&r.tailMode!==`collapsed`&&r.tailMode!==`visible`&&!o.alternate&&!W)return hl(t),null}else 2*Le()-r.renderingStartTime>hd&&n!==536870912&&(t.flags|=128,a=!0,ml(r,!1),t.lanes=4194304)}r.isBackwards?(o.sibling=t.child,t.child=o):(e=r.last,e===null?t.child=o:e.sibling=o,r.last=o)}if(r.tail!==null){e=r.tail;a:{for(n=e;n!==null;){if(n.alternate!==null){n=!1;break a}n=n.sibling}n=!0}return r.rendering=e,r.tail=e.sibling,r.renderingStartTime=Le(),e.sibling=null,o=Po.current,o=a?o&1|2:o&1,r.tailMode===`visible`||r.tailMode===`collapsed`||!n||W?Fo(t,o):(n=o,z(Do,t),z(Po,n),Oo===null&&(Oo=t)),W&&Ji(t,r.treeForkCount),e}return hl(t),null;case 22:case 23:return No(t),Eo(),r=t.memoizedState!==null,e===null?r&&(t.flags|=8192):e.memoizedState!==null!==r&&(t.flags|=8192),r?n&536870912&&!(t.flags&128)&&(hl(t),t.subtreeFlags&6&&(t.flags|=8192)):hl(t),n=t.updateQueue,n!==null&&pl(t,n.retryQueue),n=null,e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),r=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(r=t.memoizedState.cachePool.pool),r!==n&&(t.flags|=2048),e!==null&&ve(Va),null;case 24:return n=null,e!==null&&(n=e.memoizedState.cache),t.memoizedState.cache!==n&&(t.flags|=2048),ha(Da),hl(t),null;case 25:return null;case 30:return t.flags|=33554432,hl(t),null}throw Error(i(156,t.tag))}function _l(e,t){switch(Zi(t),t.tag){case 1:return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return ha(Da),we(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 26:case 27:case 5:return Ee(t),null;case 31:if(t.memoizedState!==null){if(No(t),t.alternate===null)throw Error(i(340));ca()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 13:if(No(t),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(i(340));ca()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return Io(t),e=t.flags,e&65536?(t.flags=e&-65537|128,e=t.memoizedState,e!==null&&(e.rendering=null,e.tail=null),t.flags|=4,t):null;case 4:return we(),null;case 10:return ha(t.type),null;case 22:case 23:return No(t),Eo(),e!==null&&ve(Va),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 24:return ha(Da),null;case 25:return null;default:return null}}function vl(e,t){switch(Zi(t),t.tag){case 3:ha(Da),we();break;case 26:case 27:case 5:Ee(t);break;case 4:we();break;case 31:t.memoizedState!==null&&No(t);break;case 13:No(t);break;case 19:Io(t);break;case 10:ha(t.type);break;case 22:case 23:No(t),Eo(),e!==null&&ve(Va);break;case 24:ha(Da)}}function yl(e,t){try{var n=t.updateQueue,r=n===null?null:n.lastEffect;if(r!==null){var i=r.next;n=i;do{if((n.tag&e)===e){r=void 0;var a=n.create,o=n.inst;r=a(),o.destroy=r}n=n.next}while(n!==i)}}catch(e){pf(t,t.return,e)}}function bl(e,t,n){try{var r=t.updateQueue,i=r===null?null:r.lastEffect;if(i!==null){var a=i.next;r=a;do{if((r.tag&e)===e){var o=r.inst,s=o.destroy;if(s!==void 0){o.destroy=void 0,i=t;var c=n,l=s;try{l()}catch(e){pf(i,c,e)}}}r=r.next}while(r!==a)}}catch(e){pf(t,t.return,e)}}function xl(e){var t=e.updateQueue;if(t!==null){var n=e.stateNode;try{xo(t,n)}catch(t){pf(e,e.return,t)}}}function Sl(e,t,n){n.props=yc(e.type,e.memoizedProps),n.state=e.memoizedState;try{n.componentWillUnmount()}catch(n){pf(e,t,n)}}function Cl(e,t){try{var n=e.ref;if(n!==null){switch(e.tag){case 26:case 27:case 5:var r=e.stateNode;break;case 30:var i=e.stateNode,a=pi(e.memoizedProps,i);(i.ref===null||i.ref.name!==a)&&(i.ref=Pp(a)),r=i.ref;break;case 7:if(e.stateNode===null){var o=new Fp(e);h(e.child,!1,Qp,o,void 0,void 0),e.stateNode=o}r=e.stateNode;break;default:r=e.stateNode}typeof n==`function`?e.refCleanup=n(r):n.current=r}}catch(n){pf(e,t,n)}}function wl(e,t){var n=e.ref,r=e.refCleanup;if(n!==null){if(typeof r==`function`)try{r()}catch(n){pf(e,t,n)}finally{e.refCleanup=null,e=e.alternate,e!=null&&(e.refCleanup=null)}else if(typeof n==`function`)try{n(null)}catch(n){pf(e,t,n)}else n.current=null}}function Tl(e,t){if((e.tag===5||e.tag===27||e.tag===6)&&e.alternate===null&&t!==null)for(var n=0;n<t.length;n++)em(e.stateNode,t[n])}function El(e){for(var t=e.return;t!==null&&(kl(t)&&em(e.stateNode,t.stateNode),!Ol(t));)t=t.return}function Dl(e){for(var t=e.return;t!==null&&(kl(t)&&tm(e.stateNode,t.stateNode),!Ol(t));)t=t.return}function Ol(e){return e.tag===5||e.tag===3||e.tag===27}function kl(e){return e&&e.tag===7&&e.stateNode!==null}function Al(e){var t=e.type,n=e.memoizedProps,r=e.stateNode;try{a:switch(t){case`button`:case`input`:case`select`:case`textarea`:n.autoFocus&&r.focus();break a;case`img`:n.src?r.src=n.src:n.srcSet&&(r.srcset=n.srcSet)}}catch(t){pf(e,e.return,t)}}function jl(e,t,n){try{var r=e.stateNode;ip(r,e.type,n,t),r[bt]=t}catch(t){pf(e,e.return,t)}}function Ml(e){return e.tag===5||e.tag===3||e.tag===26||e.tag===27&&Sp(e.type)||e.tag===4}function Nl(e){a:for(;;){for(;e.sibling===null;){if(e.return===null||Ml(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.tag===27&&Sp(e.type)||e.flags&2||e.child===null||e.tag===4)continue a;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function Pl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?(n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n).insertBefore(i,t):(t=n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n,t.appendChild(i),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=mn)),Tl(e,r),U=!0;else if(i!==4&&(i===27&&(Tl(e,r),r=null,Sp(e.type)&&(n=e.stateNode,t=null)),e=e.child,e!==null))for(Pl(e,t,n,r),e=e.sibling;e!==null;)Pl(e,t,n,r),e=e.sibling}function Fl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?n.insertBefore(i,t):n.appendChild(i),Tl(e,r),U=!0;else if(i!==4&&(i===27&&(Tl(e,r),r=null,Sp(e.type)&&(n=e.stateNode)),e=e.child,e!==null))for(Fl(e,t,n,r),e=e.sibling;e!==null;)Fl(e,t,n,r),e=e.sibling}function Il(e){var t=e.stateNode,n=e.memoizedProps;try{for(var r=e.type,i=t.attributes;i.length;)t.removeAttributeNode(i[0]);np(t,r,n),t[yt]=e,t[bt]=n}catch(t){pf(e,e.return,t)}}var Ll=!1,Rl=null;function zl(e){(e.tag===30||e.subtreeFlags&33554432)&&(Ll=!0)}var Bl=null;function Vl(){var e=Bl;return Bl=null,e}var Hl=0;function Ul(e,t,n,r,i){return Hl=0,Wl(e.child,t,n,r,i)}function Wl(e,t,n,r,i){for(var a=!1;e!==null;){if(e.tag===5){var o=e.stateNode;if(r!==null){var s=Op(o);r.push(s),s.view&&(a=!0)}else a||Op(o).view&&(a=!0);Ll=!0,Tp(o,Hl===0?t:t+`_`+Hl,n),Hl++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&i||Wl(e.child,t,n,r,i)&&(a=!0));e=e.sibling}return a}function Gl(e,t){for(;e!==null;)e.tag===5?Ep(e.stateNode,e.memoizedProps):(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&t||Gl(e.child,t)),e=e.sibling}function Kl(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if((e.tag!==22||e.memoizedState===null)&&(Kl(e),e.tag===30&&e.flags&18874368&&e.stateNode.paired)){var t=e.memoizedProps;if(t.name==null||t.name===`auto`)throw Error(i(544));var n=t.name;t=hi(t.default,t.share),t!==`none`&&(Ul(e,n,t,null,!1)||Gl(e.child,!1))}e=e.sibling}}function ql(e,t){if(e.tag===30){var n=e.stateNode,r=e.memoizedProps,i=pi(r,n),a=hi(r.default,n.paired?r.share:r.enter);a===`none`?Kl(e):Ul(e,i,a,null,!1)?(Kl(e),n.paired||t||Md(e,r.onEnter)):Gl(e.child,!1)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)ql(e,t),e=e.sibling;else Kl(e)}function Jl(e){if(Rl!==null&&Rl.size!==0){var t=Rl;if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var n=e.memoizedProps,r=n.name;if(r!=null&&r!==`auto`){var i=t.get(r);if(i!==void 0){var a=hi(n.default,n.share);if(a!==`none`&&(Ul(e,r,a,null,!1)?(a=e.stateNode,i.paired=a,a.paired=i,Md(e,n.onShare)):Gl(e.child,!1)),t.delete(r),t.size===0)break}}}Jl(e)}e=e.sibling}}}function Yl(e){if(e.tag===30){var t=e.memoizedProps,n=pi(t,e.stateNode),r=Rl===null?void 0:Rl.get(n),i=hi(t.default,r===void 0?t.exit:t.share);i!==`none`&&(Ul(e,n,i,null,!1)?r===void 0?Md(e,t.onExit):(i=e.stateNode,r.paired=i,i.paired=r,Rl.delete(n),Md(e,t.onShare)):Gl(e.child,!1)),Rl!==null&&Jl(e)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Yl(e),e=e.sibling;else Rl!==null&&Jl(e)}function Xl(e){for(e=e.child;e!==null;){if(e.tag===30){var t=e.memoizedProps,n=pi(t,e.stateNode);t=hi(t.default,t.update),e.flags&=-5,t!==`none`&&Ul(e,n,t,e.memoizedState=[],!1)}else e.subtreeFlags&33554432&&Xl(e);e=e.sibling}}function Zl(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var t=e.stateNode;t.paired!==null&&(t.paired=null,Gl(e.child,!1))}Zl(e)}e=e.sibling}}function Ql(e){if(e.tag===30)e.stateNode.paired=null,Gl(e.child,!1),Zl(e);else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Ql(e),e=e.sibling;else Zl(e)}function $l(e){for(e=e.child;e!==null;)e.tag===30?Gl(e.child,!1):e.subtreeFlags&33554432&&$l(e),e=e.sibling}function eu(e,t,n,r,i,a,o){for(var s=!1;t!==null;){if(t.tag===5){var c=t.stateNode;if(a!==null&&Hl<a.length){var l=a[Hl],u=Op(c);(l.view||u.view)&&(s=!0);var d;if(d=!(e.flags&4)){if(u.clip)d=!0;else{d=l.rect;var f=u.rect;d=d.y!==f.y||d.x!==f.x||d.height!==f.height||d.width!==f.width}}d&&(e.flags|=4),u.abs?u=!l.abs:(l=l.rect,u=u.rect,u=l.height!==u.height||l.width!==u.width),u&&(e.flags|=32)}else e.flags|=32;e.flags&4&&Tp(c,Hl===0?n:n+`_`+Hl,i),s&&e.flags&4||(Bl===null&&(Bl=[]),Bl.push(c,Hl===0?r:r+`_`+Hl,t.memoizedProps)),Hl++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&o?e.flags|=t.flags&32:eu(e,t.child,n,r,i,a,o)&&(s=!0));t=t.sibling}return s}function tu(e,t){for(e=e.child;e!==null;){if(e.tag===30){var n=e.memoizedProps,r=e.stateNode,i=pi(n,r),a=hi(n.default,n.update);if(t){r=r.clones;var o=r===null?null:r.map(kp)}else o=e.memoizedState,e.memoizedState=null;r=e;var s=e.child;Hl=0,i=eu(r,s,i,i,a,o,!1),e.flags&4&&i&&(t||Md(e,n.onUpdate))}else e.subtreeFlags&33554432&&tu(e,t);e=e.sibling}}var nu=!1,q=!1,ru=!1,iu=!1,au=typeof WeakSet==`function`?WeakSet:Set,ou=null,su=!1,cu=!1,lu=!1,uu=!1;function du(e,t,n){if(e=e.containerInfo,sp=gh,e=Ur(e),Wr(e)){if(`selectionStart`in e)var r={start:e.selectionStart,end:e.selectionEnd};else a:{r=(r=e.ownerDocument)&&r.defaultView||window;var i=r.getSelection&&r.getSelection();if(i&&i.rangeCount!==0){r=i.anchorNode;var a=i.anchorOffset,o=i.focusNode;i=i.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break a}var s=0,c=-1,l=-1,u=0,d=0,f=e,p=null;b:for(;;){for(var m;f!==r||a!==0&&f.nodeType!==3||(c=s+a),f!==o||i!==0&&f.nodeType!==3||(l=s+i),f.nodeType===3&&(s+=f.nodeValue.length),(m=f.firstChild)!==null;)p=f,f=m;for(;;){if(f===e)break b;if(p===r&&++u===a&&(c=s),p===o&&++d===i&&(l=s),(m=f.nextSibling)!==null)break;f=p,p=f.parentNode}f=m}r=c===-1||l===-1?null:{start:c,end:l}}else r=null}r||={start:0,end:0}}else r=null;for(cp={focusedElem:e,selectionRange:r},gh=!1,n=(n&335544064)===n,ou=t,t=n?9270:1024;ou!==null;){if(e=ou,n&&(r=e.deletions,r!==null))for(a=0;a<r.length;a++)n&&Yl(r[a]);if(e.alternate===null&&e.flags&2)n&&zl(e),fu(n);else{if(e.tag===22){if(r=e.alternate,e.memoizedState!==null){r!==null&&r.memoizedState===null&&n&&Yl(r),fu(n);continue}if(r!==null&&r.memoizedState!==null){n&&zl(e),fu(n);continue}}r=e.child,(e.subtreeFlags&t)!==0&&r!==null?(r.return=e,ou=r):(n&&Xl(e),fu(n))}}Rl=null}function fu(e){for(;ou!==null;){var t=ou,n=e,r=t.alternate,a=t.flags;switch(t.tag){case 0:case 11:case 15:break;case 1:if(a&1024&&r!==null){n=void 0,a=r.memoizedProps,r=r.memoizedState;var o=t.stateNode;try{var s=yc(t.type,a);n=o.getSnapshotBeforeUpdate(s,r),o.__reactInternalSnapshotBeforeUpdate=n}catch(e){pf(t,t.return,e)}}break;case 3:if(a&1024){if(r=t.stateNode.containerInfo,n=r.nodeType,n===9)nm(r);else if(n===1)switch(r.nodeName){case`HEAD`:case`HTML`:case`BODY`:nm(r);break;default:r.textContent=``}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&r!==null&&(n=pi(r.memoizedProps,r.stateNode),a=t.memoizedProps,a=hi(a.default,a.update),a!==`none`&&Ul(r,n,a,r.memoizedState=[],!0));break;default:if(a&1024)throw Error(i(163))}if(r=t.sibling,r!==null){r.return=t.return,ou=r;break}ou=t.return}}function pu(e,t,n){var r=n.flags;switch(n.tag){case 0:case 11:case 15:Nu(e,n),r&4&&yl(5,n);break;case 1:if(Nu(e,n),r&4){if(e=n.stateNode,t===null)try{e.componentDidMount()}catch(e){pf(n,n.return,e)}else{var i=yc(n.type,t.memoizedProps);t=t.memoizedState;try{e.componentDidUpdate(i,t,e.__reactInternalSnapshotBeforeUpdate)}catch(e){pf(n,n.return,e)}}}r&64&&xl(n),r&512&&Cl(n,n.return);break;case 3:if(Nu(e,n),r&64&&(e=n.updateQueue,e!==null)){if(t=null,n.child!==null)switch(n.child.tag){case 27:case 5:t=n.child.stateNode;break;case 1:t=n.child.stateNode}try{xo(e,t)}catch(e){pf(n,n.return,e)}}break;case 27:t===null&&r&4&&Il(n);case 26:case 5:Nu(e,n),t===null&&r&4&&Al(n),r&512&&Cl(n,n.return);break;case 12:Nu(e,n);break;case 31:Nu(e,n),r&4&&Su(e,n);break;case 13:Nu(e,n),r&4&&Cu(e,n),r&64&&(e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null&&(n=_f.bind(null,n),cm(e,n))));break;case 22:if(r=n.memoizedState!==null||nu,!r){var a=t!==null&&t.memoizedState!==null||q;t=nu,i=q,nu=r,(q=a)&&!i?(r=2,n.subtreeFlags&8772&&(r|=1),Fu(e,n,r)):Nu(e,n),nu=t,q=i}break;case 30:Nu(e,n),r&512&&Cl(n,n.return);break;case 7:r&512&&Cl(n,n.return);default:Nu(e,n)}}function mu(e,t){for(e=e.child;e!==null;)hu(e,t),e=e.sibling}function hu(e,t){switch(e.tag){case 5:case 26:try{var n=e.stateNode;if(t){var r=n.style;typeof r.setProperty==`function`?r.setProperty(`display`,`none`,`important`):r.display=`none`}else{var i=e.stateNode,a=e.memoizedProps.style,o=a!=null&&a.hasOwnProperty(`display`)?a.display:null;i.style.display=o==null||typeof o==`boolean`?``:(``+o).trim()}}catch(t){pf(e,e.return,t)}gu(e,t);break;case 6:try{e.stateNode.nodeValue=t?``:e.memoizedProps,U=!0}catch(t){pf(e,e.return,t)}break;case 18:try{var s=e.stateNode;t?wp(s,!0):wp(e.stateNode,!1)}catch(t){pf(e,e.return,t)}break;case 22:case 23:e.memoizedState===null&&mu(e,t);break;default:mu(e,t)}}function gu(e,t){if(e.subtreeFlags&67108864)for(e=e.child;e!==null;){a:{var n=e,r=t;switch(n.tag){case 4:hu(n,r);break a;case 22:n.memoizedState===null&&gu(n,r);break a;default:gu(n,r)}}e=e.sibling}}function _u(e){var t=e.alternate;t!==null&&(e.alternate=null,_u(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&Ot(t)),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}var vu=null,yu=!1;function bu(e,t,n){for(n=n.child;n!==null;)xu(e,t,n),n=n.sibling}function xu(e,t,n){if(qe&&typeof qe.onCommitFiberUnmount==`function`)try{qe.onCommitFiberUnmount(Ke,n)}catch{}switch(n.tag){case 26:q||wl(n,t),bu(e,t,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!q&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:q||wl(n,t),Dl(n);var r=vu,i=yu;Sp(n.type)&&(vu=n.stateNode,yu=!1),bu(e,t,n),gm(n.stateNode,n.type,n.memoizedProps),vu=r,yu=i;break;case 5:q||wl(n,t),Dl(n);case 6:if(n.tag===6&&Dl(n),r=vu,i=yu,vu=null,bu(e,t,n),vu=r,yu=i,vu!==null){if(yu)try{(vu.nodeType===9?vu.body:vu.nodeName===`HTML`?vu.ownerDocument.body:vu).removeChild(n.stateNode),U=!0}catch(e){pf(n,t,e)}else try{vu.removeChild(n.stateNode),U=!0}catch(e){pf(n,t,e)}}break;case 18:vu!==null&&(yu?(e=vu,Cp(e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,n.stateNode),Hh(e)):Cp(vu,n.stateNode));break;case 4:r=vu,i=yu,vu=n.stateNode.containerInfo,yu=!0,bu(e,t,n),vu=r,yu=i;break;case 0:case 11:case 14:case 15:bl(2,n,t),q||bl(4,n,t),bu(e,t,n);break;case 1:q||(wl(n,t),r=n.stateNode,typeof r.componentWillUnmount==`function`&&Sl(n,t,r)),bu(e,t,n);break;case 21:bu(e,t,n);break;case 22:q=(r=q)||n.memoizedState!==null,bu(e,t,n),q=r;break;case 30:wl(n,t),bu(e,t,n);break;case 7:q||wl(n,t),bu(e,t,n);break;default:bu(e,t,n)}}function Su(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null))){e=e.dehydrated;try{Hh(e)}catch(e){pf(t,t.return,e)}}}function Cu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null&&(e=e.dehydrated,e!==null))))try{Hh(e)}catch(e){pf(t,t.return,e)}}function wu(e){switch(e.tag){case 31:case 13:case 19:var t=e.stateNode;return t===null&&(t=e.stateNode=new au),t;case 22:return e=e.stateNode,t=e._retryCache,t===null&&(t=e._retryCache=new au),t;default:throw Error(i(435,e.tag))}}function Tu(e,t){var n=wu(e);t.forEach(function(t){if(!n.has(t)){n.add(t);var r=vf.bind(null,e,t);t.then(r,r)}})}function Eu(e,t,n){var r=t.deletions;if(r!==null)for(var a=0;a<r.length;a++){var o=r[a],s=e,c=t,l=c;a:for(;l!==null;){switch(l.tag){case 27:if(Sp(l.type)){vu=l.stateNode,yu=!1;break a}break;case 5:vu=l.stateNode,yu=!1;break a;case 3:case 4:vu=l.stateNode.containerInfo,yu=!0;break a}l=l.return}if(vu===null)throw Error(i(160));xu(s,c,o),vu=null,yu=!1,s=o.alternate,s!==null&&(s.return=null),o.return=null}if(t.subtreeFlags&13886)for(t=t.child;t!==null;)Ou(t,e,n),t=t.sibling}var Du=null;function Ou(e,t,n){var r=e.alternate,a=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(a&4&&(r=e.updateQueue,r=r===null?null:r.events,r!==null))for(var o=0;o<r.length;o++){var s=r[o];s.ref.impl=s.nextImpl}Eu(t,e,n),ku(e),a&4&&(bl(3,e,e.return),yl(3,e),bl(5,e,e.return));break;case 1:Eu(t,e,n),ku(e),a&512&&(q||r===null||wl(r,r.return)),a&64&&nu&&(e=e.updateQueue,e!==null&&(t=e.callbacks,t!==null&&(n=e.shared.hiddenCallbacks,e.shared.hiddenCallbacks=n===null?t:n.concat(t))));break;case 26:if(o=Du,Eu(t,e,n),ku(e),a&512&&(q||r===null||wl(r,r.return)),a&4){if(a=r===null?null:r.memoizedState,n=e.memoizedState,r===null){if(n===null){if(e.stateNode===null){if(nu)e.stateNode=fp(e.type,e.memoizedProps,t.containerInfo,e);else{a:{t=e.type,n=e.memoizedProps,a=o.ownerDocument||o;b:switch(t){case`title`:r=a.getElementsByTagName(`title`)[0],(!r||r[Et]||r[yt]||r.namespaceURI===`http://www.w3.org/2000/svg`||r.hasAttribute(`itemprop`))&&(r=a.createElement(t),a.head.insertBefore(r,a.querySelector(`head > title`))),np(r,t,n),r[yt]=e,Nt(r),t=r;break a;case`link`:if(o=Gm(`link`,`href`,a).get(t+(n.href||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`href`)===(n.href==null||n.href===``?null:n.href)&&r.getAttribute(`rel`)===(n.rel==null?null:n.rel)&&r.getAttribute(`title`)===(n.title==null?null:n.title)&&r.getAttribute(`crossorigin`)===(n.crossOrigin==null?null:n.crossOrigin)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;case`meta`:if(o=Gm(`meta`,`content`,a).get(t+(n.content||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`content`)===(n.content==null?null:``+n.content)&&r.getAttribute(`name`)===(n.name==null?null:n.name)&&r.getAttribute(`property`)===(n.property==null?null:n.property)&&r.getAttribute(`http-equiv`)===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute(`charset`)===(n.charSet==null?null:n.charSet)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;default:throw Error(i(468,t))}r[yt]=e,Nt(r),t=r}e.stateNode=t}}else nu||Km(o,e.type,e.stateNode)}else e.stateNode=Bm(o,n,e.memoizedProps)}else a===n?n===null&&e.stateNode!==null&&jl(e,e.memoizedProps,r.memoizedProps):(a===null?(t=r.stateNode,t===null||q||t.parentNode.removeChild(t)):a.count--,n===null?nu||Km(o,e.type,e.stateNode):Bm(o,n,e.memoizedProps))}break;case 27:Eu(t,e,n),ku(e),a&512&&(q||r===null||wl(r,r.return)),r!==null&&a&4&&jl(e,e.memoizedProps,r.memoizedProps);break;case 5:if(o=ru,ru=!1,Eu(t,e,n),ru=o,ku(e),a&512&&(q||r===null||wl(r,r.return)),e.flags&32){t=e.stateNode;try{on(t,``),U=!0}catch(t){pf(e,e.return,t)}}a&4&&e.stateNode!=null&&(t=e.memoizedProps,jl(e,t,r===null?t:r.memoizedProps)),a&1024&&(iu=!0);break;case 6:if(Eu(t,e,n),ku(e),a&4){if(e.stateNode===null)throw Error(i(162));t=e.memoizedProps,n=e.stateNode;try{n.nodeValue=t,U=!0}catch(t){pf(e,e.return,t)}}break;case 3:if(U=!1,Wm=null,o=Du,Du=bm(t.containerInfo),Eu(t,e,n),Du=o,ku(e),a&4&&r!==null&&r.memoizedState.isDehydrated)try{Hh(t.containerInfo)}catch(t){pf(e,e.return,t)}iu&&(iu=!1,Au(e)),U=!1;break;case 4:a=ru,ru=nu,r=Ht(),o=Du,Du=bm(e.stateNode.containerInfo),Eu(t,e,n),ku(e),Du=o,U&&cu&&(lu=!0),U=r,ru=a;break;case 12:Eu(t,e,n),ku(e);break;case 31:Eu(t,e,n),ku(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Tu(e,t)));break;case 13:Eu(t,e,n),ku(e),e.child.flags&8192&&e.memoizedState!==null!=(r!==null&&r.memoizedState!==null)&&(pd=Le()),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Tu(e,t)));break;case 22:o=e.memoizedState!==null,s=r!==null&&r.memoizedState!==null;var c=nu,l=q,u=ru;nu=c||o,ru=u||o,q=l||s,Eu(t,e,n),q=l,ru=u,nu=c,ku(e),a&8192&&(t=e.stateNode,t._visibility=o?t._visibility&-2:t._visibility|1,!o||r===null||s||nu||q||(t=s||q,n=nu,r=q,nu=o||nu,q=t,Pu(e,2),nu=n,q=r),!o&&ru||mu(e,o)),a&4&&(t=e.updateQueue,t!==null&&(n=t.retryQueue,n!==null&&(t.retryQueue=null,Tu(e,n))));break;case 19:Eu(t,e,n),ku(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,Tu(e,t)));break;case 30:a&512&&(q||r===null||wl(r,r.return)),a=Ht(),o=cu,s=(n&335544064)===n,c=e.memoizedProps,cu=s&&hi(c.default,c.update)!==`none`,Eu(t,e,n),ku(e),s&&r!==null&&U&&(e.flags|=4),cu=o,U=a;break;case 21:break;case 7:a&512&&(q||r===null||wl(r,r.return)),r&&r.stateNode!==null&&(r.stateNode._fragmentFiber=e);default:Eu(t,e,n),ku(e)}}function ku(e){var t=e.flags;if(t&2){try{for(var n,r=e.return;r!==null;){if(Ml(r)){n=r;break}r=r.return}r=null;for(var a=e.return;a!==null;){if(kl(a)){var o=a.stateNode;r===null?r=[o]:r.push(o)}if(Ol(a))break;a=a.return}var s=r;if(n==null)throw Error(i(160));switch(n.tag){case 27:var c=n.stateNode;Fl(e,Nl(e),c,s);break;case 5:var l=n.stateNode;n.flags&32&&(on(l,``),n.flags&=-33),Fl(e,Nl(e),l,s);break;case 3:case 4:var u=n.stateNode.containerInfo;Pl(e,Nl(e),u,s);break;default:throw Error(i(161))}}catch(t){pf(e,e.return,t)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Au(e){if(e.subtreeFlags&1024)for(e=e.child;e!==null;){var t=e;Au(t),t.tag===5&&t.flags&1024&&(t=t.stateNode,gh=!0,t.reset(),gh=!1),e=e.sibling}}function ju(e,t){if(t.subtreeFlags&9270)for(t=t.child;t!==null;)Mu(t,e),t=t.sibling;else tu(t,!1)}function Mu(e,t){var n=e.alternate;if(n===null)ql(e,!1);else switch(e.tag){case 3:if(uu=su=!1,Vl(),ju(t,e),!su&&!lu){if(e=Bl,e!==null)for(var r=0;r<e.length;r+=3){n=e[r];var i=e[r+1];Ep(n,e[r+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(`+i+`)`})}e=t.containerInfo,e=e.nodeType===9?e.documentElement:e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===``&&(e.style.viewTransitionName=`none`,e.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(root)`}),e.animate({width:[0,0],height:[0,0]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition`})),uu=!0}Bl=null;break;case 5:ju(t,e);break;case 4:r=su,su=!1,ju(t,e),su&&(lu=!0),su=r;break;case 22:e.memoizedState===null&&(n.memoizedState===null?ju(t,e):ql(e,!1));break;case 30:r=su,i=Vl(),su=!1,ju(t,e),su&&(e.flags|=4);var a=e.memoizedProps,o=e.stateNode;t=pi(a,o),o=pi(n.memoizedProps,o);var s=hi(a.default,a.update);s===`none`?t=!1:(a=n.memoizedState,n.memoizedState=null,n=e.child,Hl=0,t=eu(e,n,t,o,s,a,!0),Hl!==(a===null?0:a.length)&&(e.flags|=32)),e.flags&4&&t?(Md(e,e.memoizedProps.onUpdate),Bl=i):i!==null&&(i.push.apply(i,Bl),Bl=i),su=e.flags&32?!0:r;break;default:ju(t,e)}}function Nu(e,t){if(t.subtreeFlags&8772)for(t=t.child;t!==null;)pu(e,t.alternate,t),t=t.sibling}function Pu(e,t){for(e=e.child;e!==null;){var n=e,r=t;switch(n.tag){case 0:case 11:case 14:case 15:bl(4,n,n.return),Pu(n,r);break;case 1:wl(n,n.return);var i=n.stateNode;typeof i.componentWillUnmount==`function`&&Sl(n,n.return,i),Pu(n,r);break;case 27:r&2&&gm(n.stateNode,n.type,n.memoizedProps);case 5:wl(n,n.return),n.tag!==5&&n.tag!==27||Dl(n),Pu(n,r);break;case 6:Dl(n);break;case 26:wl(n,n.return),i=n.stateNode,n.memoizedState!==null||i===null||q||i.parentNode.removeChild(i),Pu(n,r);break;case 22:n.memoizedState===null&&Pu(n,r);break;case 30:wl(n,n.return),Pu(n,r);break;case 7:wl(n,n.return);default:Pu(n,r)}e=e.sibling}}function Fu(e,t,n){for(n=t.subtreeFlags&8772?n:n&-2,t=t.child;t!==null;){var r=t.alternate,i=e,a=t,o=a.flags,s=!!(n&1);switch(a.tag){case 0:case 11:case 15:Fu(i,a,n),yl(4,a);break;case 1:if(Fu(i,a,n),r=a,i=r.stateNode,typeof i.componentDidMount==`function`)try{i.componentDidMount()}catch(e){pf(r,r.return,e)}if(r=a,i=r.updateQueue,i!==null){var c=r.stateNode;try{var l=i.shared.hiddenCallbacks;if(l!==null)for(i.shared.hiddenCallbacks=null,i=0;i<l.length;i++)bo(l[i],c)}catch(e){pf(r,r.return,e)}}s&&o&64&&xl(a),Cl(a,a.return);break;case 27:n&2&&Il(a);case 5:a.tag!==5&&a.tag!==27||El(a),Fu(i,a,n),s&&r===null&&o&4&&Al(a),Cl(a,a.return);break;case 6:El(a);break;case 26:c=a.stateNode,a.memoizedState!==null||c===null||nu||Km(bm(c.ownerDocument),a.type,c),Fu(i,a,n),s&&r===null&&o&4&&Al(a),Cl(a,a.return);break;case 12:Fu(i,a,n);break;case 31:Fu(i,a,n),s&&o&4&&Su(i,a);break;case 13:Fu(i,a,n),s&&o&4&&Cu(i,a);break;case 22:a.memoizedState===null&&Fu(i,a,n),Cl(a,a.return);break;case 30:Fu(i,a,n),Cl(a,a.return);break;case 7:Cl(a,a.return);default:Fu(i,a,n)}t=t.sibling}}function Iu(e,t){var n=null;e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),e=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(e=t.memoizedState.cachePool.pool),e!==n&&(e!=null&&e.refCount++,n!=null&&ka(n))}function Lu(e,t){e=null,t.alternate!==null&&(e=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==e&&(t.refCount++,e!=null&&ka(e))}function Ru(e,t,n,r){var i=(n&335544064)===n;if(t.subtreeFlags&(i?10262:10256))for(t=t.child;t!==null;)zu(e,t,n,r),t=t.sibling;else i&&$l(t)}function zu(e,t,n,r){var i=(n&335544064)===n;i&&t.alternate===null&&t.return!==null&&t.return.alternate!==null&&Ql(t);var a=t.flags;switch(t.tag){case 0:case 11:case 15:Ru(e,t,n,r),a&2048&&yl(9,t);break;case 1:Ru(e,t,n,r);break;case 3:Ru(e,t,n,r),i&&uu&&(e=e.containerInfo,e=e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,e.style.viewTransitionName===`root`&&(e.style.viewTransitionName=``),e=e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===`none`&&(e.style.viewTransitionName=``)),a&2048&&(a=null,t.alternate!==null&&(a=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==a&&(t.refCount++,a!=null&&ka(a)));break;case 12:if(a&2048){Ru(e,t,n,r),a=t.stateNode;try{var o=t.memoizedProps,s=o.id,c=o.onPostCommit;typeof c==`function`&&c(s,t.alternate===null?`mount`:`update`,a.passiveEffectDuration,-0)}catch(e){pf(t,t.return,e)}}else Ru(e,t,n,r);break;case 31:Ru(e,t,n,r);break;case 13:Ru(e,t,n,r);break;case 23:break;case 22:o=t.stateNode,s=t.alternate,t.memoizedState===null?(i&&s!==null&&s.memoizedState!==null&&Ql(t),o._visibility&2?Ru(e,t,n,r):(o._visibility|=2,Bu(e,t,n,r,!!(t.subtreeFlags&10256)||!1))):(i&&s!==null&&s.memoizedState===null&&Ql(s),o._visibility&2?Ru(e,t,n,r):Vu(e,t)),a&2048&&Iu(s,t);break;case 24:Ru(e,t,n,r),a&2048&&Lu(t.alternate,t);break;case 30:i&&(a=t.alternate,a!==null&&(Gl(a.child,!0),Gl(t.child,!0))),Ru(e,t,n,r);break;default:Ru(e,t,n,r)}}function Bu(e,t,n,r,i){for(i&&=!!(t.subtreeFlags&10256)||!1,t=t.child;t!==null;){var a=e,o=t,s=n,c=r,l=o.flags;switch(o.tag){case 0:case 11:case 15:Bu(a,o,s,c,i),yl(8,o);break;case 23:break;case 22:var u=o.stateNode;o.memoizedState===null?(u._visibility|=2,Bu(a,o,s,c,i)):u._visibility&2?Bu(a,o,s,c,i):Vu(a,o),i&&l&2048&&Iu(o.alternate,o);break;case 24:Bu(a,o,s,c,i),i&&l&2048&&Lu(o.alternate,o);break;default:Bu(a,o,s,c,i)}t=t.sibling}}function Vu(e,t){if(t.subtreeFlags&10256)for(t=t.child;t!==null;){var n=e,r=t,i=r.flags;switch(r.tag){case 22:Vu(n,r),i&2048&&Iu(r.alternate,r);break;case 24:Vu(n,r),i&2048&&Lu(r.alternate,r);break;default:Vu(n,r)}t=t.sibling}}var Hu=8192;function Uu(e,t,n){if(e.subtreeFlags&Hu)for(e=e.child;e!==null;)Wu(e,t,n),e=e.sibling}function Wu(e,t,n){switch(e.tag){case 26:Uu(e,t,n),e.flags&Hu&&(e.memoizedState===null?(e=e.stateNode,(t&335544128)===t&&Zm(n,e)):Qm(n,Du,e.memoizedState,e.memoizedProps));break;case 5:Uu(e,t,n),e.flags&Hu&&(e=e.stateNode,(t&335544128)===t&&Zm(n,e));break;case 3:case 4:var r=Du;Du=bm(e.stateNode.containerInfo),Uu(e,t,n),Du=r;break;case 22:e.memoizedState===null&&(r=e.alternate,r!==null&&r.memoizedState!==null?(r=Hu,Hu=16777216,Uu(e,t,n),Hu=r):Uu(e,t,n));break;case 30:if((e.flags&Hu)!==0&&(r=e.memoizedProps.name,r!=null&&r!==`auto`)){var i=e.stateNode;i.paired=null,Rl===null&&(Rl=new Map),Rl.set(r,i)}Uu(e,t,n);break;default:Uu(e,t,n)}}function Gu(e){var t=e.alternate;if(t!==null&&(e=t.child,e!==null)){t.child=null;do t=e.sibling,e.sibling=null,e=t;while(e!==null)}}function Ku(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];ou=r,Yu(r,e)}Gu(e)}if(e.subtreeFlags&10256)for(e=e.child;e!==null;)qu(e),e=e.sibling}function qu(e){switch(e.tag){case 0:case 11:case 15:Ku(e),e.flags&2048&&bl(9,e,e.return);break;case 3:Ku(e);break;case 12:Ku(e);break;case 22:var t=e.stateNode;e.memoizedState!==null&&t._visibility&2&&(e.return===null||e.return.tag!==13)?(t._visibility&=-3,Ju(e)):Ku(e);break;default:Ku(e)}}function Ju(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];ou=r,Yu(r,e)}Gu(e)}for(e=e.child;e!==null;){switch(t=e,t.tag){case 0:case 11:case 15:bl(8,t,t.return),Ju(t);break;case 22:n=t.stateNode,n._visibility&2&&(n._visibility&=-3,Ju(t));break;default:Ju(t)}e=e.sibling}}function Yu(e,t){for(;ou!==null;){var n=ou;switch(n.tag){case 0:case 11:case 15:bl(8,n,t);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var r=n.memoizedState.cachePool.pool;r!=null&&r.refCount++}break;case 24:ka(n.memoizedState.cache)}if(r=n.child,r!==null)r.return=n,ou=r;else a:for(n=e;ou!==null;){r=ou;var i=r.sibling,a=r.return;if(_u(r),r===n){ou=null;break a}if(i!==null){i.return=a,ou=i;break a}ou=a}}}var Xu={getCacheForType:function(e){var t=xa(Da),n=t.data.get(e);return n===void 0&&(n=e(),t.data.set(e,n)),n},cacheSignal:function(){return xa(Da).controller.signal}},Zu=typeof WeakMap==`function`?WeakMap:Map,J=0,Qu=null,Y=null,X=0,Z=0,$u=null,ed=!1,td=!1,nd=!1,rd=0,id=0,ad=0,od=0,sd=0,cd=0,ld=0,ud=null,dd=null,fd=!1,pd=0,md=0,hd=1/0,gd=null,_d=null,vd=0,yd=null,bd=null,xd=0,Sd=0,Cd=null,wd=null,Td=null,Ed=null,Dd=null,Od=0,kd=null;function Ad(){return J&2&&X!==0?X&-X:L.T===null?gt():Pf()}function jd(){if(cd===0){if(!(X&536870912)||W){var e=et;et<<=1,!(et&3932160)&&(et=262144),cd=e}else cd=536870912}return e=Do.current,e!==null&&(e.flags|=32),cd}function Md(e,t){if(t!=null){var n=e.stateNode,r=n.ref;r===null&&(r=n.ref=Pp(pi(e.memoizedProps,n))),Ed===null&&(Ed=[]),Ed.push(t.bind(null,r))}}function Nd(e,t,n){(e===Qu&&(Z===2||Z===9)||e.cancelPendingCommit!==null)&&(Bd(e,0),Ld(e,X,cd,!1)),lt(e,n),(!(J&2)||e!==Qu)&&(e===Qu&&(!(J&2)&&(od|=n),id===4&&Ld(e,X,cd,!1)),Ef(e))}function Pd(e,t,n){if(J&6)throw Error(i(327));var r=!n&&!(t&127)&&(t&e.expiredLanes)===0||it(e,t),a=r?Jd(e,t):Kd(e,t,!0),o=r;do{if(a===0){td&&!r&&Ld(e,t,0,!1);break}if(n=e.current.alternate,o&&!Id(n)){a=Kd(e,t,!1),o=!1;continue}if(a===2){if(o=t,e.errorRecoveryDisabledLanes&o)var s=0;else s=e.pendingLanes&-536870913,s=s===0?s&536870912?536870912:0:s;if(s!==0){t=s;a:{var c=e;a=ud;var l=c.current.memoizedState.isDehydrated;if(l&&(Bd(c,s).flags|=256),s=Kd(c,s,!1),s!==2&&s!==6){if(nd&&!l){c.errorRecoveryDisabledLanes|=o,od|=o,a=4;break a}o=dd,dd=a,o!==null&&(dd===null?dd=o:dd.push.apply(dd,o))}a=s}if(o=!1,a!==2)continue}}if(a===1){Bd(e,0),Ld(e,t,0,!0);break}a:{switch(r=e,o=a,o){case 0:case 1:throw Error(i(345));case 4:if((t&4194048)!==t&&(t&62914560)!==t)break;case 6:Ld(r,t,cd,!ed);break a;case 2:dd=null;break;case 3:case 5:break;default:throw Error(i(329))}if((t&62914560)===t&&(a=pd+300-Le(),10<a)){if(Ld(r,t,cd,!ed),rt(r,0,!0)!==0)break a;xd=t,r.timeoutHandle=gp(Fd.bind(null,r,n,dd,gd,fd,t,cd,od,ld,ed,o,`Throttled`,-0,0),a);break a}Fd(r,n,dd,gd,fd,t,cd,od,ld,ed,o,null,-0,0)}break}while(1);Ef(e)}function Fd(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.timeoutHandle=-1;var m=t.subtreeFlags,h=(a&335544064)===a;if(d=null,(h||m&8192||(m&16785408)==16785408)&&(d={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:mn},Rl=null,Wu(t,a,d),h&&(m=d,h=e.containerInfo,h=(h.nodeType===9?h:h.ownerDocument).__reactViewTransition,h!=null&&(m.count++,m.waitingForViewTransition=!0,m=nh.bind(m),h.finished.then(m,m))),m=(a&62914560)===a?pd-Le():(a&4194048)===a?md-Le():0,m=eh(d,m),m!==null)){xd=a,e.cancelPendingCommit=m(tf.bind(null,e,t,a,n,r,i,o,s,c,l,u,d,null,f,p)),Ld(e,a,o,!l);return}tf(e,t,a,n,r,i,o,s,c,l,u,d)}function Id(e){for(var t=e;;){var n=t.tag;if((n===0||n===11||n===15)&&t.flags&16384&&(n=t.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!Lr(a(),i))return!1}catch{return!1}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Ld(e,t,n,r){t=at(e,t),t&=~sd,t&=~od,e.suspendedLanes|=t,e.pingedLanes&=~t,r&&(e.warmLanes|=t),r=e.expirationTimes;for(var i=t;0<i;){var a=31-Ye(i),o=1<<a;r[a]=-1,i&=~o}n!==0&&dt(e,n,t)}function Rd(){return J&6?!0:(Df(0,!1),!1)}function zd(){if(Y!==null){if(Z===0)var e=Y.return;else e=Y,pa=fa=null,ts(e),to=null,no=0,e=Y;for(;e!==null;)vl(e.alternate,e),e=e.return;Y=null}}function Bd(e,t){var n=e.timeoutHandle;return n!==-1&&(e.timeoutHandle=-1,_p(n)),n=e.cancelPendingCommit,n!==null&&(e.cancelPendingCommit=null,n()),xd=0,zd(),Qu=e,Y=n=Ai(e.current,null),X=t,Z=0,$u=null,ed=!1,td=it(e,t),nd=!1,ld=cd=sd=od=ad=id=0,dd=ud=null,fd=!1,rd=at(e,t),bi(),n}function Vd(e,t){G=null,L.H=dc,t===Ga||t===qa?(t=$a(),Z=3):t===Ka?(t=$a(),Z=4):Z=t===kc?8:typeof t==`object`&&t&&typeof t.then==`function`?6:1,$u=t,Y===null&&(id=1,Cc(e,Ri(t,e.current)))}function Hd(){var e=Do.current;return e===null?!0:(X&4194048)===X?Oo===null:(X&62914560)===X||X&536870912?e===Oo:!1}function Ud(){var e=L.H;return L.H=dc,e===null?dc:e}function Wd(){var e=L.A;return L.A=Xu,e}function Gd(){id=4,ed||(X&4194048)!==X&&Do.current!==null||(td=!0),!(ad&134217727)&&!(od&134217727)||Qu===null||Ld(Qu,X,cd,!1)}function Kd(e,t,n){var r=J;J|=2;var i=Ud(),a=Wd();(Qu!==e||X!==t)&&(gd=null,Bd(e,t)),t=!1;var o=id;a:do try{if(Z!==0&&Y!==null){var s=Y,c=$u;switch(Z){case 8:zd(),o=6;break a;case 3:case 2:case 9:case 6:Do.current===null&&(t=!0);var l=Z;if(Z=0,$u=null,Qd(e,s,c,l),n&&td){o=0;break a}break;default:l=Z,Z=0,$u=null,Qd(e,s,c,l)}}qd(),o=id;break}catch(t){Vd(e,t)}while(1);return t&&e.shellSuspendCounter++,pa=fa=null,J=r,L.H=i,L.A=a,Y===null&&(Qu=null,X=0,bi()),o}function qd(){for(;Y!==null;)Xd(Y)}function Jd(e,t){var n=J;J|=2;var r=Ud(),a=Wd();Qu!==e||X!==t?(gd=null,hd=Le()+500,Bd(e,t)):td=it(e,t);a:do try{if(Z!==0&&Y!==null){t=Y;var o=$u;b:switch(Z){case 1:Z=0,$u=null,Qd(e,t,o,1);break;case 2:case 9:if(Ya(o)){Z=0,$u=null,Zd(t);break}t=function(){Z!==2&&Z!==9||Qu!==e||(Z=7),Ef(e)},o.then(t,t);break a;case 3:Z=7;break a;case 4:Z=5;break a;case 7:Ya(o)?(Z=0,$u=null,Zd(t)):(Z=0,$u=null,Qd(e,t,o,7));break;case 5:var s=null;switch(Y.tag){case 26:s=Y.memoizedState;case 5:case 27:var c=Y;if(s?Ym(s):c.stateNode.complete){Z=0,$u=null;var l=c.sibling;if(l!==null)Y=l;else{var u=c.return;u===null?Y=null:(Y=u,$d(u))}break b}}Z=0,$u=null,Qd(e,t,o,5);break;case 6:Z=0,$u=null,Qd(e,t,o,6);break;case 8:zd(),id=6;break a;default:throw Error(i(462))}}Yd();break}catch(t){Vd(e,t)}while(1);return pa=fa=null,L.H=r,L.A=a,J=n,Y===null?(Qu=null,X=0,bi(),id):0}function Yd(){for(;Y!==null&&!Fe();)Xd(Y)}function Xd(e){var t=ll(e.alternate,e,rd);e.memoizedProps=e.pendingProps,t===null?$d(e):Y=t}function Zd(e){var t=e,n=t.alternate;switch(t.tag){case 15:case 0:t=Uc(n,t,t.pendingProps,t.type,void 0,X);break;case 11:t=Uc(n,t,t.pendingProps,t.type.render,t.ref,X);break;case 5:ts(t);var r=t;r===$i&&(W?(oa(r),r.tag===5&&r.stateNode!=null&&(ea=r.stateNode)):(oa(r),W=!0));default:vl(n,t),t=Y=ji(t,rd),t=ll(n,t,rd)}e.memoizedProps=e.pendingProps,t===null?$d(e):Y=t}function Qd(e,t,n,r){pa=fa=null,ts(t),to=null,no=0;var i=t.return;try{if(Oc(e,i,t,n,X)){id=1,Cc(e,Ri(n,e.current)),Y=null;return}}catch(t){if(i!==null)throw Y=i,t;id=1,Cc(e,Ri(n,e.current)),Y=null;return}t.flags&32768?(W||r===1?e=!0:td||X&536870912?e=!1:(ed=e=!0,(r===2||r===9||r===3||r===6)&&(r=Do.current,r!==null&&r.tag===13&&(r.flags|=16384))),ef(t,e)):$d(t)}function $d(e){var t=e;do{if(t.flags&32768){ef(t,ed);return}e=t.return;var n=gl(t.alternate,t,rd);if(n!==null){Y=n;return}if(t=t.sibling,t!==null){Y=t;return}Y=t=e}while(t!==null);id===0&&(id=5)}function ef(e,t){do{var n=_l(e.alternate,e);if(n!==null){n.flags&=32767,Y=n;return}if(n=e.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!t&&(e=e.sibling,e!==null)){Y=e;return}Y=e=n}while(e!==null);id=6,Y=null}function tf(e,t,n,r,a,o,s,c,l,u,d,f){e.cancelPendingCommit=null;do uf();while(vd!==0);if(J&6)throw Error(i(327));if(t!==null){if(t===e.current)throw Error(i(177));e===Qu&&(Y=Qu=null,X=0),bd=t,yd=e,xd=n,Cd=a,wd=r,nf(e,t,n,s,c,l,f)}}function nf(e,t,n,r,i,a,o){var s=t.lanes|t.childLanes;if(Sd=s,s|=yi,ut(e,n,s,r,i,a),Ed=null,(n&335544064)===n?(Dd=Ma(e),r=10262):(Dd=null,r=10256),(t.subtreeFlags&r)!==0||(t.flags&r)!==0?(e.callbackNode=null,e.callbackPriority=0,yf(Ve,function(){return df(),null})):(e.callbackNode=null,e.callbackPriority=0),Ll=!1,r=!!(t.flags&13878),t.subtreeFlags&13878||r){r=L.T,L.T=null,i=R.p,R.p=2,a=J,J|=4;try{du(e,t,n)}finally{J=a,R.p=i,L.T=r}}vd=1,Ll?Td=Mp(o,e.containerInfo,Dd,of,sf,af,cf,df,rf,null,null):(of(),sf(),cf())}function rf(e){if(vd!==0){var t=yd.onRecoverableError;t(e,{componentStack:null})}}function af(){vd===3&&(vd=0,Mu(bd,yd),vd=4)}function of(){if(vd===1){vd=0;var e=yd,t=bd,n=xd,r=!!(t.flags&13878);if(t.subtreeFlags&13878||r){r=L.T,L.T=null;var i=R.p;R.p=2;var a=J;J|=4;try{cu=lu=!1,Ou(t,e,n),n=cp;var o=Ur(e.containerInfo),s=n.focusedElem,c=n.selectionRange;if(o!==s&&s&&s.ownerDocument&&Hr(s.ownerDocument.documentElement,s)){if(c!==null&&Wr(s)){var l=c.start,u=c.end;if(u===void 0&&(u=l),`selectionStart`in s)s.selectionStart=l,s.selectionEnd=Math.min(u,s.value.length);else{var d=s.ownerDocument||document,f=d&&d.defaultView||window;if(f.getSelection){var p=f.getSelection(),m=s.textContent.length,h=Math.min(c.start,m),g=c.end===void 0?h:Math.min(c.end,m);!p.extend&&h>g&&(o=g,g=h,h=o);var _=Vr(s,h),v=Vr(s,g);if(_&&v&&(p.rangeCount!==1||p.anchorNode!==_.node||p.anchorOffset!==_.offset||p.focusNode!==v.node||p.focusOffset!==v.offset)){var y=d.createRange();y.setStart(_.node,_.offset),p.removeAllRanges(),h>g?(p.addRange(y),p.extend(v.node,v.offset)):(y.setEnd(v.node,v.offset),p.addRange(y))}}}}for(d=[],p=s;p=p.parentNode;)p.nodeType===1&&d.push({element:p,left:p.scrollLeft,top:p.scrollTop});for(typeof s.focus==`function`&&s.focus(),s=0;s<d.length;s++){var b=d[s];b.element.scrollLeft=b.left,b.element.scrollTop=b.top}}gh=!!sp,cp=sp=null}finally{J=a,R.p=i,L.T=r}}e.current=t,vd=2}}function sf(){if(vd===2){vd=0;var e=yd,t=bd,n=!!(t.flags&8772);if(t.subtreeFlags&8772||n){n=L.T,L.T=null;var r=R.p;R.p=2;var i=J;J|=4;try{pu(e,t.alternate,t)}finally{J=i,R.p=r,L.T=n}}vd=3}}function cf(){if(vd===4||vd===3){vd=0;var e=Td;Td=null,Ie();var t=yd,n=bd,r=xd,i=wd,a=(r&335544064)===r?10262:10256;if((n.subtreeFlags&a)!==0||(n.flags&a)!==0?vd=5:(vd=0,bd=yd=null,lf(t,t.pendingLanes)),a=t.pendingLanes,a===0&&(_d=null),ht(r),n=n.stateNode,qe&&typeof qe.onCommitFiberRoot==`function`)try{qe.onCommitFiberRoot(Ke,n,void 0,(n.current.flags&128)==128)}catch{}if(i!==null){n=L.T,a=R.p,R.p=2,L.T=null;try{for(var o=t.onRecoverableError,s=0;s<i.length;s++){var c=i[s];o(c.value,{componentStack:c.stack})}}finally{L.T=n,R.p=a}}if(i=Ed,o=Dd,Dd=null,i!==null&&(Ed=null,o===null&&(o=[]),e!==null))for(c=0;c<i.length;c++)n=(0,i[c])(o),n!==void 0&&e.finished.finally(n);xd&3&&uf(),Ef(t),a=t.pendingLanes,r&261930&&a&42?t===kd?Od++:(Od=0,kd=t):(Od=0,kd=null),Df(0,!1)}}function lf(e,t){(e.pooledCacheLanes&=t)===0&&(t=e.pooledCache,t!=null&&(e.pooledCache=null,ka(t)))}function uf(){return Td!==null&&(Td.skipTransition(),Td=null),of(),sf(),cf(),df()}function df(){if(vd!==5)return!1;var e=yd,t=Sd;Sd=0;var n=ht(xd),r=L.T,a=R.p;try{R.p=32>n?32:n,L.T=null,n=Cd,Cd=null;var o=yd,s=xd;if(vd=0,bd=yd=null,xd=0,J&6)throw Error(i(331));var c=J;if(J|=4,qu(o.current),zu(o,o.current,s,n),J=c,Df(0,!1),qe&&typeof qe.onPostCommitFiberRoot==`function`)try{qe.onPostCommitFiberRoot(Ke,o)}catch{}return!0}finally{R.p=a,L.T=r,lf(e,t)}}function ff(e,t,n){t=Ri(n,t),t=Tc(e.stateNode,t,2),e=mo(e,t,2),e!==null&&(lt(e,2),Ef(e))}function pf(e,t,n){if(e.tag===3)ff(e,e,n);else for(;t!==null;){if(t.tag===3){ff(t,e,n);break}if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError==`function`||typeof r.componentDidCatch==`function`&&(_d===null||!_d.has(r))){e=Ri(n,e),n=Ec(2),r=mo(t,n,2),r!==null&&(Dc(n,r,t,e),lt(r,2),Ef(r));break}}t=t.return}}function mf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new Zu;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(nd=!0,i.add(n),e=hf.bind(null,e,t,n),t.then(e,e))}function hf(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),e.pingedLanes|=e.suspendedLanes&n,e.warmLanes&=~n,Qu===e&&(X&n)===n&&(id===4||id===3&&(X&62914560)===X&&300>Le()-pd?J&2?sd|=n:Bd(e,0):sd|=n,ld===X&&(ld=0)),Ef(e)}function gf(e,t){t===0&&(t=st()),e=Ci(e,t),e!==null&&(lt(e,t),Ef(e))}function _f(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),gf(e,n)}function vf(e,t){var n=0;switch(e.tag){case 31:case 13:var r=e.stateNode,a=e.memoizedState;a!==null&&(n=a.retryLane);break;case 19:r=e.stateNode;break;case 22:r=e.stateNode._retryCache;break;default:throw Error(i(314))}r!==null&&r.delete(t),gf(e,n)}function yf(e,t){return Ne(e,t)}var bf=null,xf=null,Sf=!1,Cf=!1,wf=!1,Tf=0;function Ef(e){e!==xf&&e.next===null&&(xf===null?bf=xf=e:xf=xf.next=e),Cf=!0,Sf||(Sf=!0,Nf())}function Df(e,t){if(!wf&&Cf){wf=!0;do for(var n=!1,r=bf;r!==null;){if(!t){if(e!==0){var i=r.pendingLanes;if(i===0)var a=0;else{var o=r.suspendedLanes,s=r.pingedLanes;a=(1<<31-Ye(42|e)+1)-1,a&=i&~(o&~s),a=a&201326741?a&201326741|1:a?a|2:0}a!==0&&(n=!0,Mf(r,a))}else a=X,a=rt(r,r===Qu?a:0,r.cancelPendingCommit!==null||r.timeoutHandle!==-1),!(a&3)||it(r,a)||(n=!0,Mf(r,a))}r=r.next}while(n);wf=!1}}function Of(){kf()}function kf(){Cf=Sf=!1;var e=0;Tf!==0&&hp()&&(e=Tf);for(var t=Le(),n=null,r=bf;r!==null;){var i=r.next,a=Af(r,t);a===0?(r.next=null,n===null?bf=i:n.next=i,i===null&&(xf=n)):(n=r,(e!==0||a&3)&&(Cf=!0)),r=i}vd!==0&&vd!==5||Df(e,!1),Tf!==0&&(Tf=0)}function Af(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes&-62914561;0<a;){var o=31-Ye(a),s=1<<o,c=i[o];c===-1?((s&n)===0||(s&r)!==0)&&(i[o]=ot(s,t)):c<=t&&(e.expiredLanes|=s),a&=~s}if(t=Qu,n=X,n=rt(e,e===t?n:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r=e.callbackNode,n===0||e===t&&(Z===2||Z===9)||e.cancelPendingCommit!==null)return r!==null&&r!==null&&Pe(r),e.callbackNode=null,e.callbackPriority=0;if(!(n&3)||it(e,n)){if(t=n&-n,t===e.callbackPriority)return t;switch(r!==null&&Pe(r),ht(n)){case 2:case 8:n=Be;break;case 32:n=Ve;break;case 268435456:n=Ue;break;default:n=Ve}return r=jf.bind(null,e),n=Ne(n,r),e.callbackPriority=t,e.callbackNode=n,t}return r!==null&&r!==null&&Pe(r),e.callbackPriority=2,e.callbackNode=null,2}function jf(e,t){if(vd!==0&&vd!==5)return e.callbackNode=null,e.callbackPriority=0,null;var n=e.callbackNode;if(uf()&&e.callbackNode!==n)return null;var r=X;return r=rt(e,e===Qu?r:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r===0?null:(Pd(e,r,t),Af(e,Le()),e.callbackNode!=null&&e.callbackNode===n?jf.bind(null,e):null)}function Mf(e,t){if(uf())return null;Pd(e,t,!0)}function Nf(){bp(function(){J&6?Ne(ze,Of):kf()})}function Pf(){if(Tf===0){var e=Fa;e===0&&(e=$e,$e<<=1,!($e&261888)&&($e=256)),Tf=e}return Tf}function Ff(e){return e==null||typeof e==`symbol`||typeof e==`boolean`?null:typeof e==`function`?e:pn(e)}function If(e,t,n,r,i){if(t===`submit`&&n&&n.stateNode===i){var a=Ff((i[bt]||null).action),o=r.submitter;o&&(t=(t=o[bt]||null)?Ff(t.formAction):o.getAttribute(`formAction`),t!==null&&(a=t,o=null));var s=new Fn(`action`,`action`,null,r,i);e.push({event:s,listeners:[{instance:null,listener:function(){if(r.defaultPrevented){if(Tf!==0){var e=new FormData(i,o);Zs(n,{pending:!0,data:e,method:i.method,action:a},null,e)}}else typeof a==`function`&&(s.preventDefault(),e=new FormData(i,o),Zs(n,{pending:!0,data:e,method:i.method,action:a},a,e))},currentTarget:i}]})}}for(var Lf=0;Lf<ui.length;Lf++){var Rf=ui[Lf];di(Rf.toLowerCase(),`on`+(Rf[0].toUpperCase()+Rf.slice(1)))}di(ni,`onAnimationEnd`),di(ri,`onAnimationIteration`),di(ii,`onAnimationStart`),di(`dblclick`,`onDoubleClick`),di(`focusin`,`onFocus`),di(`focusout`,`onBlur`),di(ai,`onTransitionRun`),di(oi,`onTransitionStart`),di(si,`onTransitionCancel`),di(ci,`onTransitionEnd`),Rt(`onMouseEnter`,[`mouseout`,`mouseover`]),Rt(`onMouseLeave`,[`mouseout`,`mouseover`]),Rt(`onPointerEnter`,[`pointerout`,`pointerover`]),Rt(`onPointerLeave`,[`pointerout`,`pointerover`]),Lt(`onChange`,`change click focusin focusout input keydown keyup selectionchange`.split(` `)),Lt(`onSelect`,`focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange`.split(` `)),Lt(`onBeforeInput`,[`compositionend`,`keypress`,`textInput`,`paste`]),Lt(`onCompositionEnd`,`compositionend focusout keydown keypress keyup mousedown`.split(` `)),Lt(`onCompositionStart`,`compositionstart focusout keydown keypress keyup mousedown`.split(` `)),Lt(`onCompositionUpdate`,`compositionupdate focusout keydown keypress keyup mousedown`.split(` `));var zf=`abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting`.split(` `),Bf=new Set(`beforetoggle cancel close invalid load scroll scrollend toggle`.split(` `).concat(zf));function Vf(e,t){t=!!(t&4);for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;a:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],c=s.instance,l=s.currentTarget;if(s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){gi(e)}i.currentTarget=null,a=c}else for(o=0;o<r.length;o++){if(s=r[o],c=s.instance,l=s.currentTarget,s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){gi(e)}i.currentTarget=null,a=c}}}}function Q(e,t){var n=t[St];n===void 0&&(n=t[St]=new Set);var r=e+`__bubble`;n.has(r)||(Gf(t,e,2,!1),n.add(r))}function Hf(e,t,n){var r=0;t&&(r|=4),Gf(n,e,r,t)}var Uf=`_reactListening`+Math.random().toString(36).slice(2);function Wf(e){if(!e[Uf]){e[Uf]=!0,Ft.forEach(function(t){t!==`selectionchange`&&(Bf.has(t)||Hf(t,!1,e),Hf(t,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Uf]||(t[Uf]=!0,Hf(`selectionchange`,!1,t))}}function Gf(e,t,n,r){switch(Ch(t)){case 2:var i=_h;break;case 8:i=vh;break;default:i=yh}n=i.bind(null,t,n,e),i=void 0,!wn||t!==`touchstart`&&t!==`touchmove`&&t!==`wheel`||(i=!0),r?i===void 0?e.addEventListener(t,n,!0):e.addEventListener(t,n,{capture:!0,passive:i}):i===void 0?e.addEventListener(t,n,!1):e.addEventListener(t,n,{passive:i})}function Kf(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)a:for(;;){if(r===null)return;var s=r.tag;if(s===3||s===4){var c=r.stateNode.containerInfo;if(c===i)break;if(s===4)for(s=r.return;s!==null;){var l=s.tag;if((l===3||l===4)&&s.stateNode.containerInfo===i)return;s=s.return}for(;c!==null;){if(s=kt(c),s===null)return;if(l=s.tag,l===5||l===6||l===26||l===27){r=a=s;continue a}c=c.parentNode}}r=r.return}xn(function(){var r=a,i=gn(n),s=[];a:{var c=li.get(e);if(c!==void 0){var l=Fn,u=e;switch(e){case`keypress`:if(An(n)===0)break a;case`keydown`:case`keyup`:l=$n;break;case`focusin`:u=`focus`,l=Wn;break;case`focusout`:u=`blur`,l=Wn;break;case`beforeblur`:case`afterblur`:l=Wn;break;case`click`:if(n.button===2)break a;case`auxclick`:case`dblclick`:case`mousedown`:case`mousemove`:case`mouseup`:case`mouseout`:case`mouseover`:case`contextmenu`:l=Hn;break;case`drag`:case`dragend`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`dragstart`:case`drop`:l=Un;break;case`touchcancel`:case`touchend`:case`touchmove`:case`touchstart`:l=nr;break;case ni:case ri:case ii:l=Gn;break;case ci:l=rr;break;case`scroll`:case`scrollend`:l=Ln;break;case`wheel`:l=ir;break;case`copy`:case`cut`:case`paste`:l=Kn;break;case`gotpointercapture`:case`lostpointercapture`:case`pointercancel`:case`pointerdown`:case`pointermove`:case`pointerout`:case`pointerover`:case`pointerup`:l=er;break;case`submit`:l=tr;break;case`toggle`:case`beforetoggle`:l=ar}var d=!!(t&4),f=!d&&(e===`scroll`||e===`scrollend`),p=d?c===null?null:c+`Capture`:c;d=[];for(var m=r,h;m!==null;){var g=m;if(h=g.stateNode,g=g.tag,g!==5&&g!==26&&g!==27||h===null||p===null||(g=Sn(m,p),g!=null&&d.push(qf(m,g,h))),f)break;m=m.return}0<d.length&&(c=new l(c,u,null,n,i),s.push({event:c,listeners:d}))}}if(!(t&7)){a:{if(l=e===`mouseover`||e===`pointerover`,c=e===`mouseout`||e===`pointerout`,l&&n!==hn&&(u=n.relatedTarget||n.fromElement)&&(kt(u)||u[xt]))break a;(c||l)&&(u=i.window===i?i:(l=i.ownerDocument)?l.defaultView||l.parentWindow:window,c?(l=n.relatedTarget||n.toElement,c=r,l=l?kt(l):null,l!==null&&(f=o(l),d=l.tag,l!==f||d!==5&&d!==27&&d!==6)&&(l=null)):(c=null,l=r),c!==l&&(d=Hn,g=`onMouseLeave`,p=`onMouseEnter`,m=`mouse`,(e===`pointerout`||e===`pointerover`)&&(d=er,g=`onPointerLeave`,p=`onPointerEnter`,m=`pointer`),f=c==null?u:jt(c),h=l==null?u:jt(l),u=new d(g,m+`leave`,c,n,i),u.target=f,u.relatedTarget=h,g=null,kt(i)===r&&(d=new d(p,m+`enter`,l,n,i),d.target=h,d.relatedTarget=f,g=d),f=g,d=c&&l?E(c,l,Yf):null,c!==null&&Xf(s,u,c,d,!1),l!==null&&f!==null&&Xf(s,f,l,d,!0)))}a:{if(c=r?jt(r):window,l=c.nodeName&&c.nodeName.toLowerCase(),l===`select`||l===`input`&&c.type===`file`)var _=Tr;else if(yr(c)){if(Er)_=Fr;else{_=Nr;var v=Mr}}else l=c.nodeName,!l||l.toLowerCase()!==`input`||c.type!==`checkbox`&&c.type!==`radio`?r&&un(r.elementType)&&(_=Tr):_=Pr;if(_&&=_(e,r)){br(s,_,n,i);break a}v&&v(e,c,r)}switch(v=r?jt(r):window,e){case`focusin`:(yr(v)||v.contentEditable===`true`)&&(Kr=v,qr=r,Jr=null);break;case`focusout`:Jr=qr=Kr=null;break;case`mousedown`:Yr=!0;break;case`contextmenu`:case`mouseup`:case`dragend`:Yr=!1,Xr(s,n,i);break;case`selectionchange`:if(Gr)break;case`keydown`:case`keyup`:Xr(s,n,i)}var y;if(sr)b:{switch(e){case`compositionstart`:var b=`onCompositionStart`;break b;case`compositionend`:b=`onCompositionEnd`;break b;case`compositionupdate`:b=`onCompositionUpdate`;break b}b=void 0}else hr?pr(e,n)&&(b=`onCompositionEnd`):e===`keydown`&&n.keyCode===229&&(b=`onCompositionStart`);b&&(ur&&n.locale!==`ko`&&(hr||b!==`onCompositionStart`?b===`onCompositionEnd`&&hr&&(y=kn()):(En=i,Dn=`value`in En?En.value:En.textContent,hr=!0)),v=Jf(r,b),0<v.length&&(b=new qn(b,e,null,n,i),s.push({event:b,listeners:v}),y?b.data=y:(y=mr(n),y!==null&&(b.data=y)))),(y=lr?gr(e,n):_r(e,n))&&(b=Jf(r,`onBeforeInput`),0<b.length&&(v=new qn(`onBeforeInput`,`beforeinput`,null,n,i),s.push({event:v,listeners:b}),v.data=y)),If(s,e,r,n,i)}Vf(s,t)})}function qf(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Jf(e,t){for(var n=t+`Capture`,r=[];e!==null;){var i=e,a=i.stateNode;if(i=i.tag,i!==5&&i!==26&&i!==27||a===null||(i=Sn(e,n),i!=null&&r.unshift(qf(e,i,a)),i=Sn(e,t),i!=null&&r.push(qf(e,i,a))),e.tag===3)return r;e=e.return}return[]}function Yf(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5&&e.tag!==27);return e||null}function Xf(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,c=s.alternate,l=s.stateNode;if(s=s.tag,c!==null&&c===r)break;s!==5&&s!==26&&s!==27||l===null||(c=l,i?(l=Sn(n,a),l!=null&&o.unshift(qf(n,l,c))):i||(l=Sn(n,a),l!=null&&o.push(qf(n,l,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Zf=/\r\n?/g,Qf=/\u0000|\uFFFD/g;function $f(e){return(typeof e==`string`?e:``+e).replace(Zf,`
`).replace(Qf,``)}function ep(e,t){return t=$f(t),$f(e)===t}function $(e,t,n,r,a,o){switch(n){case`children`:if(typeof r==`string`)t===`body`||t===`textarea`&&r===``||on(e,r);else if(typeof r==`number`||typeof r==`bigint`)t!==`body`&&on(e,``+r);else return;break;case`className`:Wt(e,`class`,r);break;case`tabIndex`:Wt(e,`tabindex`,r);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:Wt(e,n,r);break;case`style`:ln(e,r,o);return;case`data`:if(t!==`object`){Wt(e,`data`,r);break}case`src`:case`href`:if(r===``&&(t!==`a`||n!==`href`)){e.removeAttribute(n);break}if(r==null||typeof r==`function`||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=pn(r),e.setAttribute(n,r);break;case`action`:case`formAction`:if(typeof r==`function`){e.setAttribute(n,`javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')`);break}if(typeof o==`function`&&(n===`formAction`?(t!==`input`&&$(e,t,`name`,a.name,a,null),$(e,t,`formEncType`,a.formEncType,a,null),$(e,t,`formMethod`,a.formMethod,a,null),$(e,t,`formTarget`,a.formTarget,a,null)):($(e,t,`encType`,a.encType,a,null),$(e,t,`method`,a.method,a,null),$(e,t,`target`,a.target,a,null))),r==null||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=pn(r),e.setAttribute(n,r);break;case`onClick`:r!=null&&(e.onclick=mn);return;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`multiple`:e.multiple=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`muted`:e.muted=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`ref`:break;case`autoFocus`:break;case`xlinkHref`:if(r==null||typeof r==`function`||typeof r==`boolean`||typeof r==`symbol`){e.removeAttribute(`xlink:href`);break}n=pn(r),e.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,n);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`credentialless`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:r&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,``):e.removeAttribute(n);break;case`capture`:case`download`:!0===r?e.setAttribute(n,``):!1!==r&&r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`cols`:case`rows`:case`size`:case`span`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`&&!isNaN(r)&&1<=r?e.setAttribute(n,r):e.removeAttribute(n);break;case`rowSpan`:case`start`:r==null||typeof r==`function`||typeof r==`symbol`||isNaN(r)?e.removeAttribute(n):e.setAttribute(n,r);break;case`popover`:Q(`beforetoggle`,e),Q(`toggle`,e),Ut(e,`popover`,r);break;case`xlinkActuate`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:actuate`,r);break;case`xlinkArcrole`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:arcrole`,r);break;case`xlinkRole`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:role`,r);break;case`xlinkShow`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:show`,r);break;case`xlinkTitle`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:title`,r);break;case`xlinkType`:Gt(e,`http://www.w3.org/1999/xlink`,`xlink:type`,r);break;case`xmlBase`:Gt(e,`http://www.w3.org/XML/1998/namespace`,`xml:base`,r);break;case`xmlLang`:Gt(e,`http://www.w3.org/XML/1998/namespace`,`xml:lang`,r);break;case`xmlSpace`:Gt(e,`http://www.w3.org/XML/1998/namespace`,`xml:space`,r);break;case`is`:Ut(e,`is`,r);break;case`innerText`:case`textContent`:return;default:if(!(2<n.length)||n[0]!==`o`&&n[0]!==`O`||n[1]!==`n`&&n[1]!==`N`)n=dn.get(n)||n,Ut(e,n,r);else return}U=!0}function tp(e,t,n,r,a,o){switch(n){case`style`:ln(e,r,o);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`children`:if(typeof r==`string`)on(e,r);else if(typeof r==`number`||typeof r==`bigint`)on(e,``+r);else return;break;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`onClick`:r!=null&&(e.onclick=mn);return;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`innerHTML`:case`ref`:return;case`innerText`:case`textContent`:return;default:if(!It.hasOwnProperty(n))a:{if(n[0]===`o`&&n[1]===`n`&&(a=n.endsWith(`Capture`),o=n.slice(2,a?n.length-7:void 0),t=e[bt]||null,t=t==null?null:t[n],typeof t==`function`&&e.removeEventListener(o,t,a),typeof r==`function`)){typeof t!=`function`&&t!==null&&(n in e?e[n]=null:e.hasAttribute(n)&&e.removeAttribute(n)),e.addEventListener(o,r,a);break a}U=!0,n in e?e[n]=r:!0===r?e.setAttribute(n,``):Ut(e,n,r)}return}U=!0}function np(e,t,n){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`img`:Q(`error`,e),Q(`load`,e);var r=!1,a=!1,o;for(o in n)if(n.hasOwnProperty(o)){var s=n[o];if(s!=null)switch(o){case`src`:r=!0;break;case`srcSet`:a=!0;break;case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,o,s,n,null)}}a&&$(e,t,`srcSet`,n.srcSet,n,null),r&&$(e,t,`src`,n.src,n,null);return;case`input`:Q(`invalid`,e);var c=o=s=a=null,l=null,u=null;for(r in n)if(n.hasOwnProperty(r)){var d=n[r];if(d!=null)switch(r){case`name`:a=d;break;case`type`:s=d;break;case`checked`:l=d;break;case`defaultChecked`:u=d;break;case`value`:o=d;break;case`defaultValue`:c=d;break;case`children`:case`dangerouslySetInnerHTML`:if(d!=null)throw Error(i(137,t));break;default:$(e,t,r,d,n,null)}}en(e,o,c,l,u,s,a,!1);return;case`select`:for(a in Q(`invalid`,e),r=s=o=null,n)if(n.hasOwnProperty(a)&&(c=n[a],c!=null))switch(a){case`value`:o=c;break;case`defaultValue`:s=c;break;case`multiple`:r=c;default:$(e,t,a,c,n,null)}t=o,n=s,e.multiple=!!r,t==null?n!=null&&nn(e,!!r,n,!0):nn(e,!!r,t,!1);return;case`textarea`:for(s in Q(`invalid`,e),o=a=r=null,n)if(n.hasOwnProperty(s)&&(c=n[s],c!=null))switch(s){case`value`:r=c;break;case`defaultValue`:a=c;break;case`children`:o=c;break;case`dangerouslySetInnerHTML`:if(c!=null)throw Error(i(91));break;default:$(e,t,s,c,n,null)}an(e,r,a,o);return;case`option`:for(l in n)if(n.hasOwnProperty(l)&&(r=n[l],r!=null))switch(l){case`selected`:e.selected=r&&typeof r!=`function`&&typeof r!=`symbol`;break;default:$(e,t,l,r,n,null)}return;case`dialog`:Q(`beforetoggle`,e),Q(`toggle`,e),Q(`cancel`,e),Q(`close`,e);break;case`iframe`:case`object`:Q(`load`,e);break;case`video`:case`audio`:for(r=0;r<zf.length;r++)Q(zf[r],e);break;case`image`:Q(`error`,e),Q(`load`,e);break;case`details`:Q(`toggle`,e);break;case`embed`:case`source`:case`link`:Q(`error`,e),Q(`load`,e);case`area`:case`base`:case`br`:case`col`:case`hr`:case`keygen`:case`meta`:case`param`:case`track`:case`wbr`:case`menuitem`:for(u in n)if(n.hasOwnProperty(u)&&(r=n[u],r!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,u,r,n,null)}return;default:if(un(t)){for(d in n)n.hasOwnProperty(d)&&(r=n[d],r!==void 0&&tp(e,t,d,r,n,void 0));return}}for(c in n)n.hasOwnProperty(c)&&(r=n[c],r!=null&&$(e,t,c,r,n,null))}var rp={};function ip(e,t,n,r){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`input`:var a=null,o=null,s=null,c=null,l=null,u=null,d=null;for(m in n){var f=n[m];if(n.hasOwnProperty(m)&&f!=null)switch(m){case`checked`:break;case`value`:break;case`defaultValue`:l=f;default:r.hasOwnProperty(m)||$(e,t,m,null,r,f)}}for(var p in r){var m=r[p];if(f=n[p],r.hasOwnProperty(p)&&(m!=null||f!=null))switch(p){case`type`:m!==f&&(U=!0),o=m;break;case`name`:m!==f&&(U=!0),a=m;break;case`checked`:m!==f&&(U=!0),u=m;break;case`defaultChecked`:m!==f&&(U=!0),d=m;break;case`value`:m!==f&&(U=!0),s=m;break;case`defaultValue`:m!==f&&(U=!0),c=m;break;case`children`:case`dangerouslySetInnerHTML`:if(m!=null)throw Error(i(137,t));break;default:m!==f&&$(e,t,p,m,r,f)}}$t(e,s,c,l,u,d,o,a);return;case`select`:for(o in m=s=c=p=null,n)if(l=n[o],n.hasOwnProperty(o)&&l!=null)switch(o){case`value`:break;case`multiple`:m=l;default:r.hasOwnProperty(o)||$(e,t,o,null,r,l)}for(a in r)if(o=r[a],l=n[a],r.hasOwnProperty(a)&&(o!=null||l!=null))switch(a){case`value`:o!==l&&(U=!0),p=o;break;case`defaultValue`:o!==l&&(U=!0),c=o;break;case`multiple`:o!==l&&(U=!0),s=o;default:o!==l&&$(e,t,a,o,r,l)}t=c,n=s,r=m,p==null?!!r!=!!n&&(t==null?nn(e,!!n,n?[]:``,!1):nn(e,!!n,t,!0)):nn(e,!!n,p,!1);return;case`textarea`:for(c in m=p=null,n)if(a=n[c],n.hasOwnProperty(c)&&a!=null&&!r.hasOwnProperty(c))switch(c){case`value`:break;case`children`:break;default:$(e,t,c,null,r,a)}for(s in r)if(a=r[s],o=n[s],r.hasOwnProperty(s)&&(a!=null||o!=null))switch(s){case`value`:a!==o&&(U=!0),p=a;break;case`defaultValue`:a!==o&&(U=!0),m=a;break;case`children`:break;case`dangerouslySetInnerHTML`:if(a!=null)throw Error(i(91));break;default:a!==o&&$(e,t,s,a,r,o)}rn(e,p,m);return;case`option`:for(var h in n)if(p=n[h],n.hasOwnProperty(h)&&p!=null&&!r.hasOwnProperty(h))switch(h){case`selected`:e.selected=!1;break;default:$(e,t,h,null,r,p)}for(l in r)if(p=r[l],m=n[l],r.hasOwnProperty(l)&&p!==m&&(p!=null||m!=null))switch(l){case`selected`:p!==m&&(U=!0),e.selected=p&&typeof p!=`function`&&typeof p!=`symbol`;break;default:$(e,t,l,p,r,m)}return;case`img`:case`link`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:case`menuitem`:for(var g in n)p=n[g],n.hasOwnProperty(g)&&p!=null&&!r.hasOwnProperty(g)&&$(e,t,g,null,r,p);for(u in r)if(p=r[u],m=n[u],r.hasOwnProperty(u)&&p!==m&&(p!=null||m!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:if(p!=null)throw Error(i(137,t));break;default:$(e,t,u,p,r,m)}return;default:if(un(t)){for(var _ in n)p=n[_],n.hasOwnProperty(_)&&p!==void 0&&!r.hasOwnProperty(_)&&tp(e,t,_,void 0,r,p);for(d in r)p=r[d],m=n[d],!r.hasOwnProperty(d)||p===m||p===void 0&&m===void 0||tp(e,t,d,p,r,m);return}}for(var v in n)p=n[v],n.hasOwnProperty(v)&&p!=null&&!r.hasOwnProperty(v)&&$(e,t,v,null,r,p);for(f in r)p=r[f],m=n[f],!r.hasOwnProperty(f)||p===m||p==null&&m==null||$(e,t,f,p,r,m)}function ap(e){switch(e){case`css`:case`script`:case`font`:case`img`:case`image`:case`input`:case`link`:return!0;default:return!1}}function op(){if(typeof performance.getEntriesByType==`function`){for(var e=0,t=0,n=performance.getEntriesByType(`resource`),r=0;r<n.length;r++){var i=n[r],a=i.transferSize,o=i.initiatorType,s=i.duration;if(a&&s&&ap(o)){for(o=0,s=i.responseEnd,r+=1;r<n.length;r++){var c=n[r],l=c.startTime;if(l>s)break;var u=c.transferSize,d=c.initiatorType;u&&ap(d)&&(c=c.responseEnd,o+=u*(c<s?1:(s-l)/(c-l)))}if(--r,t+=8*(a+o)/(i.duration/1e3),e++,10<e)break}}if(0<e)return t/e/1e6}return navigator.connection&&(e=navigator.connection.downlink,typeof e==`number`)?e:5}var sp=null,cp=null;function lp(e){return e.nodeType===9?e:e.ownerDocument}function up(e){switch(e){case`http://www.w3.org/2000/svg`:return 1;case`http://www.w3.org/1998/Math/MathML`:return 2;default:return 0}}function dp(e,t){if(e===0)switch(t){case`svg`:return 1;case`math`:return 2;default:return 0}return e===1&&t===`foreignObject`?0:e}function fp(e,t,n,r){return n=lp(n).createElement(e),n[yt]=r,n[bt]=t,np(n,e,t),Nt(n),n}function pp(e,t){return e===`textarea`||e===`noscript`||typeof t.children==`string`||typeof t.children==`number`||typeof t.children==`bigint`||typeof t.dangerouslySetInnerHTML==`object`&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var mp=null;function hp(){var e=window.event;return e&&e.type===`popstate`?e!==mp&&(mp=e,!0):(mp=null,!1)}var gp=typeof setTimeout==`function`?setTimeout:void 0,_p=typeof clearTimeout==`function`?clearTimeout:void 0,vp=typeof Promise==`function`?Promise:void 0,yp=typeof requestAnimationFrame==`function`?requestAnimationFrame:gp,bp=typeof queueMicrotask==`function`?queueMicrotask:vp===void 0?gp:function(e){return vp.resolve(null).then(e).catch(xp)};function xp(e){setTimeout(function(){throw e})}function Sp(e){return e===`head`}function Cp(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8){if(n=i.data,n===`/$`||n===`/&`){if(r===0){e.removeChild(i),Hh(t);return}r--}else if(n===`$`||n===`$?`||n===`$~`||n===`$!`||n===`&`)r++;else if(n===`html`)_m(e.ownerDocument.documentElement);else if(n===`head`){n=e.ownerDocument.head,_m(n);for(var a=n.firstChild;a;){var o=a.nextSibling,s=a.nodeName;a[Et]||s===`SCRIPT`||s===`STYLE`||s===`LINK`&&a.rel.toLowerCase()===`stylesheet`||n.removeChild(a),a=o}}else n===`body`&&_m(e.ownerDocument.body)}n=i}while(n);Hh(t)}function wp(e,t){var n=e;e=0;do{var r=n.nextSibling;if(n.nodeType===1?t?(n._stashedDisplay=n.style.display,n.style.display=`none`):(n.style.display=n._stashedDisplay||``,n.getAttribute(`style`)===``&&n.removeAttribute(`style`)):n.nodeType===3&&(t?(n._stashedText=n.nodeValue,n.nodeValue=``):n.nodeValue=n._stashedText||``),r&&r.nodeType===8){if(n=r.data,n===`/$`){if(e===0)break;e--}else n!==`$`&&n!==`$?`&&n!==`$~`&&n!==`$!`||e++}n=r}while(n)}function Tp(e,t,n){if(t=CSS.escape(t)===t?t:`r-`+btoa(t).replace(/=/g,``),e.style.viewTransitionName=t,n!=null&&(e.style.viewTransitionClass=n),n=getComputedStyle(e),n.display===`inline`){if(t=e.getClientRects(),t.length===1)var r=1;else for(var i=r=0;i<t.length;i++){var a=t[i];0<a.width&&0<a.height&&r++}r===1&&(e=e.style,e.display=t.length===1?`inline-block`:`block`,e.marginTop=`-`+n.paddingTop,e.marginBottom=`-`+n.paddingBottom)}}function Ep(e,t){e=e.style,t=t.style;var n=t==null?null:t.hasOwnProperty(`viewTransitionName`)?t.viewTransitionName:t.hasOwnProperty(`view-transition-name`)?t[`view-transition-name`]:null;e.viewTransitionName=n==null||typeof n==`boolean`?``:(``+n).trim(),n=t==null?null:t.hasOwnProperty(`viewTransitionClass`)?t.viewTransitionClass:t.hasOwnProperty(`view-transition-class`)?t[`view-transition-class`]:null,e.viewTransitionClass=n==null||typeof n==`boolean`?``:(``+n).trim(),e.display===`inline-block`&&(t==null?e.display=e.margin=``:(n=t.display,e.display=n==null||typeof n==`boolean`?``:n,n=t.margin,n==null?(n=t.hasOwnProperty(`marginTop`)?t.marginTop:t[`margin-top`],e.marginTop=n==null||typeof n==`boolean`?``:n,t=t.hasOwnProperty(`marginBottom`)?t.marginBottom:t[`margin-bottom`],e.marginBottom=t==null||typeof t==`boolean`?``:t):e.margin=n))}function Dp(e,t,n){return n=n.ownerDocument.defaultView,{rect:e,abs:t.position===`absolute`||t.position===`fixed`,clip:t.clipPath!==`none`||t.overflow!==`visible`||t.filter!==`none`||t.mask!==`none`||t.mask!==`none`||t.borderRadius!==`0px`,view:0<=e.bottom&&0<=e.right&&e.top<=n.innerHeight&&e.left<=n.innerWidth}}function Op(e){return Dp(e.getBoundingClientRect(),getComputedStyle(e),e)}function kp(e){var t=e.getBoundingClientRect();t=new DOMRect(t.x+2e4,t.y+2e4,t.width,t.height);var n=getComputedStyle(e);return Dp(t,n,e)}function Ap(e){return e.documentElement.clientHeight}function jp(e){this.addEventListener(`load`,e),this.addEventListener(`error`,e)}function Mp(e,t,n,r,i,a,o,s,c){var l=t.nodeType===9?t:t.ownerDocument;try{var u=l.startViewTransition({update:function(){var t=l.defaultView,n=t.navigation&&t.navigation.transition,o=l.fonts.status;r();var s=[];if(o===`loaded`&&(Ap(l),l.fonts.status===`loading`&&s.push(l.fonts.ready)),o=s.length,e!==null)for(var c=e.suspenseyImages,u=0,d=0;d<c.length;d++){var f=c[d];if(!f.complete){var p=f.getBoundingClientRect();if(0<p.bottom&&0<p.right&&p.top<t.innerHeight&&p.left<t.innerWidth){if(u+=Xm(f),u>$m){s.length=o;break}f=new Promise(jp.bind(f)),s.push(f)}}}if(0<s.length)return t=Promise.race([Promise.all(s),new Promise(function(e){return setTimeout(e,500)})]).then(i,i),(n?Promise.allSettled([n.finished,t]):t).then(a,a);if(i(),n)return n.finished.then(a,a);a()},types:n});l.__reactViewTransition=u;var d=[];return u.ready.then(function(){for(var e=l.documentElement.getAnimations({subtree:!0}),t=0;t<e.length;t++){var n=e[t],r=n.effect,i=r.pseudoElement;if(i!=null&&i.startsWith(`::view-transition`)){d.push(n),n=r.getKeyframes();for(var a=i=void 0,s=!0,c=0;c<n.length;c++){var u=n[c],f=u.width;if(i===void 0)i=f;else if(i!==f){s=!1;break}if(f=u.height,a===void 0)a=f;else if(a!==f){s=!1;break}delete u.width,delete u.height,u.transform===`none`&&delete u.transform}s&&i!==void 0&&a!==void 0&&(r.setKeyframes(n),s=getComputedStyle(r.target,r.pseudoElement),s.width!==i||s.height!==a)&&(s=n[0],s.width=i,s.height=a,s=n[n.length-1],s.width=i,s.height=a,r.setKeyframes(n))}}o()},function(e){l.__reactViewTransition===u&&(l.__reactViewTransition=null);try{if(typeof e==`object`&&e)switch(e.name){case`InvalidStateError`:(e.message===`View transition was skipped because document visibility state is hidden.`||e.message===`Skipping view transition because document visibility state has become hidden.`||e.message===`Skipping view transition because viewport size changed.`||e.message===`Transition was aborted because of invalid state`)&&(e=null)}e!==null&&c(e)}finally{r(),i(),o()}}),u.finished.finally(function(){for(var e=0;e<d.length;e++)d[e].cancel();l.__reactViewTransition===u&&(l.__reactViewTransition=null),s()}),u}catch{return r(),i(),o(),null}}function Np(e,t){this._scope=document.documentElement,this._selector=`::view-transition-`+e+`(`+t+`)`}Np.prototype.animate=function(e,t){return t=typeof t==`number`?{duration:t}:D({},t),t.pseudoElement=this._selector,this._scope.animate(e,t)},Np.prototype.getAnimations=function(){for(var e=this._scope,t=this._selector,n=e.getAnimations({subtree:!0}),r=[],i=0;i<n.length;i++){var a=n[i].effect;a!==null&&a.target===e&&a.pseudoElement===t&&r.push(n[i])}return r},Np.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function Pp(e){return{name:e,group:new Np(`group`,e),imagePair:new Np(`image-pair`,e),old:new Np(`old`,e),new:new Np(`new`,e)}}function Fp(e){this._fragmentFiber=e,this._observers=this._eventListeners=null}Fp.prototype.addEventListener=function(e,t,n){var r=null,i=null;if(!(n!=null&&typeof n!=`boolean`&&(r=n.signal||null,r!==null&&r.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var a=this._eventListeners;if(Bp(a,e,t,n)===-1){var o=this,s=t;n!=null&&typeof n!=`boolean`&&!0===n.once&&(s=function(r){o.removeEventListener(e,t,n),typeof t==`function`?t.call(this,r):t.handleEvent(r)}),r!==null&&(i=o.removeEventListener.bind(o,e,t,n),r.addEventListener(`abort`,i,{once:!0}),i=r.removeEventListener.bind(r,`abort`,i)),r=Rp(n),a.push({type:e,listener:t,optionsOrUseCapture:n,attachedListener:s,cleanup:i}),h(this._fragmentFiber.child,!1,Ip,e,s,r)}this._eventListeners=a}};function Ip(e,t,n,r){return b(e).addEventListener(t,n,r),!1}Fp.prototype.removeEventListener=function(e,t,n){var r=this._eventListeners;if(r!==null&&(t=Bp(r,e,t,n),t!==-1)){var i=r[t];n=i.attachedListener;var a=i.cleanup;i=Rp(i.optionsOrUseCapture),h(this._fragmentFiber.child,!1,Lp,e,n,i),r.splice(t,1),a!==null&&a()}};function Lp(e,t,n,r){return b(e).removeEventListener(t,n,r),!1}function Rp(e){return e!=null&&typeof e!=`boolean`&&(!0===e.once||e.signal instanceof AbortSignal)?{capture:e.capture,passive:e.passive}:e}function zp(e){return e==null?`c=0`:typeof e==`boolean`?`c=`+(e?`1`:`0`):`c=`+(e.capture?`1`:`0`)}function Bp(e,t,n,r){if(e.length===0)return-1;r=zp(r);for(var i=0;i<e.length;i++){var a=e[i];if(a.type===t&&a.listener===n&&zp(a.optionsOrUseCapture)===r)return i}return-1}Fp.prototype.dispatchEvent=function(e){var t=g(this._fragmentFiber);if(t===null)return!0;t=b(t);var n=this._eventListeners;if(n!==null&&0<n.length||!e.bubbles){var r=t.nodeType===9?t.createComment(``):document.createTextNode(``);if(n)for(var i=0;i<n.length;i++){var a=n[i];r.addEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture))}if(t.appendChild(r),e=r.dispatchEvent(e),n)for(i=0;i<n.length;i++)a=n[i],r.removeEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture));return t.removeChild(r),e}return t.dispatchEvent(e)},Fp.prototype.focus=function(e){h(this._fragmentFiber.child,!0,Vp,e,void 0,void 0)};function Vp(e,t){return e.tag!==6&&(e=b(e),pm(e,t))}Fp.prototype.focusLast=function(e){var t=[];h(this._fragmentFiber.child,!0,Hp,t,void 0,void 0);for(var n=t.length-1;0<=n&&!Vp(t[n],e);n--);};function Hp(e,t){return t.push(e),!1}Fp.prototype.blur=function(){var e=g(this._fragmentFiber);e!==null&&(e=b(e),e=lp(e).activeElement,e!==null&&h(this._fragmentFiber.child,!1,Up,e,void 0,void 0))};function Up(e,t){return e.tag!==6&&(e=b(e),e===t||e.contains(t)?(t.blur(),!0):!1)}Fp.prototype.observeUsing=function(e){this._observers===null&&(this._observers=new Set),this._observers.add(e),h(this._fragmentFiber.child,!1,Wp,e,void 0,void 0)};function Wp(e,t){return e.tag!==6&&(e=b(e),t.observe(e),!1)}Fp.prototype.unobserveUsing=function(e){var t=this._observers;if(t!==null&&t.has(e)){t.delete(e),h(this._fragmentFiber.child,!1,Gp,e,void 0,void 0);for(var n=t=0;n<Kp.length;n++){var r=Kp[n];r.fragmentInstance===this&&r.observer===e?e.unobserve(r.instance):Kp[t++]=r}Kp.length=t}};function Gp(e,t){return e.tag!==6&&(e=b(e),t.unobserve(e),!1)}var Kp=[],qp=!1;function Jp(e,t,n){Kp.push({fragmentInstance:e,observer:t,instance:n}),qp||(qp=!0,mm(function(){qp=!1;var e=Kp;Kp=[];for(var t=0;t<e.length;t++){var n=e[t];n.observer.unobserve(n.instance)}}))}Fp.prototype.getClientRects=function(){var e=[];return h(this._fragmentFiber.child,!1,Yp,e,void 0,void 0),e};function Yp(e,t){if(e.tag===6){e=e.stateNode;var n=e.ownerDocument.createRange();n.selectNodeContents(e),t.push.apply(t,n.getClientRects())}else e=b(e),t.push.apply(t,e.getClientRects());return!1}Fp.prototype.getRootNode=function(e){var t=g(this._fragmentFiber);return t===null?this:b(t).getRootNode(e)},Fp.prototype.compareDocumentPosition=function(e){var t=g(this._fragmentFiber);if(t===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];h(this._fragmentFiber.child,!1,Hp,n,void 0,void 0);var r=b(t);if(n.length===0){if(n=r,_(this._fragmentFiber)){a:{for(t=this._fragmentFiber.return;t!==null;){if(t.tag===4){t=t.stateNode.containerInfo;break a}if(t.tag===3||t.tag===5||t.tag===27)break;t=t.return}t=null}t!=null&&(n=t)}t=this._fragmentFiber;var i=r=n.compareDocumentPosition(e);return n===e?i=Node.DOCUMENT_POSITION_CONTAINS:r&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=v(t)[1],n===null?i=Node.DOCUMENT_POSITION_PRECEDING:(e=b(n).compareDocumentPosition(e),i=e===0||e&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),i|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}t=b(n[0]),i=b(n[n.length-1]);var a=_(this._fragmentFiber)?t.parentElement:r;if(a==null)return Node.DOCUMENT_POSITION_DISCONNECTED;r=a.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_CONTAINED_BY,a=a.compareDocumentPosition(i)&Node.DOCUMENT_POSITION_CONTAINED_BY;var o=t.compareDocumentPosition(e),s=i.compareDocumentPosition(e),c=o&Node.DOCUMENT_POSITION_CONTAINED_BY||s&Node.DOCUMENT_POSITION_CONTAINED_BY;return s=r&&a&&o&Node.DOCUMENT_POSITION_FOLLOWING&&s&Node.DOCUMENT_POSITION_PRECEDING,t=r&&t===e||a&&i===e||c||s?Node.DOCUMENT_POSITION_CONTAINED_BY:!r&&t===e||!a&&i===e?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:o,t&Node.DOCUMENT_POSITION_DISCONNECTED||t&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Xp(t,this._fragmentFiber,n[0],n[n.length-1],e)?t:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Xp(e,t,n,r,i){var a=kt(i);if(e&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!a)a:{for(;a!==null;){if(a.tag===7&&(a===t||a.alternate===t)){n=!0;break a}a=a.return}n=!1}return n}if(e&Node.DOCUMENT_POSITION_CONTAINS){if(a===null)return a=i.ownerDocument,i===a||i===a.documentElement||i===a.body;a:{for(a=t,t=g(t);a!==null;){if(!(a.tag!==5&&a.tag!==3&&a.tag!==27||a!==t&&a.alternate!==t)){a=!0;break a}a=a.return}a=!1}return a}return e&Node.DOCUMENT_POSITION_PRECEDING?((t=!!a)&&!(t=a===n)&&(t=E(n,a,T),t===null?t=!1:(h(t,!0,C,a,n),a=x,x=null,t=a!==null)),t):e&Node.DOCUMENT_POSITION_FOLLOWING?((t=!!a)&&!(t=a===r)&&(t=E(r,a,T),t===null?t=!1:(h(t,!0,w,a,r),a=x,S=x=null,t=a!==null)),t):!1}function Zp(e,t){var n=e.ownerDocument.createRange();n.selectNodeContents(e),e=n.getBoundingClientRect(),window.scrollTo(window.scrollX+e.left,t?window.scrollY+e.top:window.scrollY+e.bottom-window.innerHeight)}Fp.prototype.scrollIntoView=function(e){if(typeof e==`object`)throw Error(i(566));var t=[];h(this._fragmentFiber.child,!1,Hp,t,void 0,void 0);var n=!1!==e;if(t.length===0){var r=v(this._fragmentFiber);if(r=n?r[1]||r[0]||g(this._fragmentFiber):r[0]||r[1],r===null)return;if(r.tag===6){e=b(r),Zp(e,n);return}if(r=b(r),r.nodeType!==9){if(r.nodeType===11){n=`host`in r?r.host:null,n!==null&&n.scrollIntoView(e);return}r.scrollIntoView(e)}}for(r=n?t.length-1:0;r!==(n?-1:t.length);){var a=t[r];a.tag===6?(a=b(a),Zp(a,n)):b(a).scrollIntoView(e),r+=n?-1:1}};function Qp(e,t){return e=b(e),$p(e,t),!1}function $p(e,t){e.reactFragments??=new Set,e.reactFragments.add(t)}function em(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.addEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){for(var r=0,i=0;i<Kp.length;i++){var a=Kp[i];(a.fragmentInstance!==t||a.observer!==n||a.instance!==e)&&(Kp[r++]=a)}Kp.length=r,n.observe(e)}),$p(e,t))}function tm(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.removeEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){typeof n.rootMargin==`string`?Jp(t,n,e):n.unobserve(e)}),e.reactFragments!=null&&e.reactFragments.delete(t))}function nm(e){var t=e.firstChild;for(t&&t.nodeType===10&&(t=t.nextSibling);t;){var n=t;switch(t=t.nextSibling,n.nodeName){case`HTML`:case`HEAD`:case`BODY`:nm(n),Ot(n);continue;case`SCRIPT`:case`STYLE`:continue;case`LINK`:if(n.rel.toLowerCase()===`stylesheet`)continue}e.removeChild(n)}}function rm(e,t,n,r){for(;e.nodeType===1;){var i=n;if(e.nodeName.toLowerCase()!==t.toLowerCase()){if(!r&&(e.nodeName!==`INPUT`||e.type!==`hidden`))break}else if(!r){if(t===`input`&&e.type===`hidden`){var a=i.name==null?null:``+i.name;if(i.type===`hidden`&&e.getAttribute(`name`)===a)return e}else return e}else if(!e[Et])switch(t){case`meta`:if(!e.hasAttribute(`itemprop`))break;return e;case`link`:if(a=e.getAttribute(`rel`),a===`stylesheet`&&e.hasAttribute(`data-precedence`)||a!==i.rel||e.getAttribute(`href`)!==(i.href==null||i.href===``?null:i.href)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin)||e.getAttribute(`title`)!==(i.title==null?null:i.title))break;return e;case`style`:if(e.hasAttribute(`data-precedence`))break;return e;case`script`:if(a=e.getAttribute(`src`),(a!==(i.src==null?null:i.src)||e.getAttribute(`type`)!==(i.type==null?null:i.type)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin))&&a&&e.hasAttribute(`async`)&&!e.hasAttribute(`itemprop`))break;return e;default:return e}if(e=lm(e.nextSibling),e===null)break}return null}function im(e,t,n){if(t===``)return null;for(;e.nodeType!==3;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!n||(e=lm(e.nextSibling),e===null))return null;return e}function am(e,t){for(;e.nodeType!==8;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!t||(e=lm(e.nextSibling),e===null))return null;return e}function om(e){return e.data===`$?`||e.data===`$~`}function sm(e){return e.data===`$!`||e.data===`$?`&&e.ownerDocument.readyState!==`loading`}function cm(e,t){var n=e.ownerDocument;if(e.data===`$~`)e._reactRetry=t;else if(e.data!==`$?`||n.readyState!==`loading`)t();else{var r=function(){t(),n.removeEventListener(`DOMContentLoaded`,r)};n.addEventListener(`DOMContentLoaded`,r),e._reactRetry=r}}function lm(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t===`$`||t===`$!`||t===`$?`||t===`$~`||t===`&`||t===`F!`||t===`F`)break;if(t===`/$`||t===`/&`)return null}}return e}var um=null;function dm(e){e=e.nextSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`/$`||n===`/&`){if(t===0)return lm(e.nextSibling);t--}else n!==`$`&&n!==`$!`&&n!==`$?`&&n!==`$~`&&n!==`&`||t++}e=e.nextSibling}return null}function fm(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`$`||n===`$!`||n===`$?`||n===`$~`||n===`&`){if(t===0)return e;t--}else n!==`/$`&&n!==`/&`||t++}e=e.previousSibling}return null}function pm(e,t){function n(){r=!0}if(e.ownerDocument.activeElement===e)return!0;var r=!1;try{e.ownerDocument.addEventListener(`focus`,n,!0),(e.focus||HTMLElement.prototype.focus).call(e,t)}finally{e.ownerDocument.removeEventListener(`focus`,n,!0)}return r}function mm(e){yp(function(){yp(function(t){return e(t)})})}function hm(e,t,n){switch(t=lp(n),e){case`html`:if(e=t.documentElement,!e)throw Error(i(452));return e;case`head`:if(e=t.head,!e)throw Error(i(453));return e;case`body`:if(e=t.body,!e)throw Error(i(454));return e;default:throw Error(i(451))}}function gm(e,t,n){for(var r in n){var i=n[r];n.hasOwnProperty(r)&&i!=null&&$(e,t,r,null,rp,i)}n.dangerouslySetInnerHTML!=null&&(e.textContent=``),e.onclick===mn&&(e.onclick=null),Ot(e)}function _m(e){for(var t=e.attributes;t.length;)e.removeAttributeNode(t[0]);Ot(e)}var vm=new Map,ym=new Set;function bm(e){if(typeof e.getRootNode==`function`){var t=e.getRootNode();if(t.nodeType===9||t.nodeType===11)return t}return e.nodeType===9?e:e.ownerDocument}var xm=R.d;R.d={f:Sm,r:Cm,D:Em,C:Dm,L:Om,m:km,X:jm,S:Am,M:Mm};function Sm(){var e=xm.f(),t=Rd();return e||t}function Cm(e){var t=At(e);t!==null&&t.tag===5&&t.type===`form`?$s(t):xm.r(e)}var wm=typeof document>`u`?null:document;function Tm(e,t,n){var r=wm;if(r&&typeof t==`string`&&t){var i=Qt(t);i=`link[rel="`+e+`"][href="`+i+`"]`,typeof n==`string`&&(i+=`[crossorigin="`+n+`"]`),ym.has(i)||(ym.add(i),e={rel:e,crossOrigin:n,href:t},r.querySelector(i)===null&&(t=r.createElement(`link`),np(t,`link`,e),Nt(t),r.head.appendChild(t)))}}function Em(e){xm.D(e),Tm(`dns-prefetch`,e,null)}function Dm(e,t){xm.C(e,t),Tm(`preconnect`,e,t)}function Om(e,t,n){xm.L(e,t,n);var r=wm;if(r&&e&&t){var i=`link[rel="preload"][as="`+Qt(t)+`"]`;t===`image`&&n&&n.imageSrcSet?(i+=`[imagesrcset="`+Qt(n.imageSrcSet)+`"]`,typeof n.imageSizes==`string`&&(i+=`[imagesizes="`+Qt(n.imageSizes)+`"]`)):i+=`[href="`+Qt(e)+`"]`;var a=i;switch(t){case`style`:a=Pm(e);break;case`script`:a=Rm(e)}if(!(vm.has(a)||(e=D({rel:`preload`,href:t===`image`&&n&&n.imageSrcSet?void 0:e,as:t},n),vm.set(a,e),r.querySelector(i)!==null||t===`style`&&r.querySelector(Fm(a))||t===`script`&&r.querySelector(zm(a))))){var o=r.createElement(`link`);np(o,`link`,e),t===`style`&&(o[Dt]=!0,o.onload=o.onerror=function(){Pt(o)}),Nt(o),r.head.appendChild(o)}}}function km(e,t){xm.m(e,t);var n=wm;if(n&&e){var r=t&&typeof t.as==`string`?t.as:`script`,i=`link[rel="modulepreload"][as="`+Qt(r)+`"][href="`+Qt(e)+`"]`,a=i;switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:a=Rm(e)}if(!vm.has(a)&&(e=D({rel:`modulepreload`,href:e},t),vm.set(a,e),n.querySelector(i)===null)){switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:if(n.querySelector(zm(a)))return}r=n.createElement(`link`),np(r,`link`,e),Nt(r),n.head.appendChild(r)}}}function Am(e,t,n){xm.S(e,t,n);var r=wm;if(r&&e){var i=Mt(r).hoistableStyles,a=Pm(e);t||=`default`;var o=i.get(a);if(!o){var s={loading:0,preload:null};if(o=r.querySelector(Fm(a)))s.loading=5;else{e=D({rel:`stylesheet`,href:e,"data-precedence":t},n),(n=vm.get(a))&&Hm(e,n);var c=o=r.createElement(`link`);Nt(c),np(c,`link`,e),c._p=new Promise(function(e,t){c.onload=e,c.onerror=t}),c.addEventListener(`load`,function(){s.loading|=1}),c.addEventListener(`error`,function(){s.loading|=2}),s.loading|=4,Vm(o,t,r)}o={type:`stylesheet`,instance:o,count:1,state:s},i.set(a,o)}}}function jm(e,t){xm.X(e,t);var n=wm;if(n&&e){var r=Mt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=D({src:e,async:!0},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),Nt(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Mm(e,t){xm.M(e,t);var n=wm;if(n&&e){var r=Mt(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=D({src:e,async:!0,type:`module`},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),Nt(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Nm(e,t,n,r){var a=(a=xe.current)?bm(a):null;if(!a)throw Error(i(446));switch(e){case`meta`:case`title`:return null;case`style`:return typeof n.precedence==`string`&&typeof n.href==`string`?(n=Pm(n.href),t=Mt(a).hoistableStyles,r=t.get(n),r||(r={type:`style`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};case`link`:if(n.rel===`stylesheet`&&typeof n.href==`string`&&typeof n.precedence==`string`){e=Pm(n.href);var o=Mt(a).hoistableStyles,s=o.get(e);if(s||(a=a.ownerDocument||a,s={type:`stylesheet`,instance:null,count:0,state:{loading:0,preload:null}},o.set(e,s),(o=a.querySelector(Fm(e)))?o._p||(s.instance=o,s.state.loading=5):(o=vm.get(e),o||(o={rel:`preload`,as:`style`,href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},vm.set(e,o)),Lm(a,e,o,s.state))),t&&r===null)throw Error(i(528,``));return s}if(t&&r!==null)throw Error(i(529,``));return null;case`script`:return t=n.async,n=n.src,typeof n==`string`&&t&&typeof t!=`function`&&typeof t!=`symbol`?(n=Rm(n),t=Mt(a).hoistableScripts,r=t.get(n),r||(r={type:`script`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};default:throw Error(i(444,e))}}function Pm(e){return`href="`+Qt(e)+`"`}function Fm(e){return`link[rel="stylesheet"][`+e+`]`}function Im(e){return D({},e,{"data-precedence":e.precedence,precedence:null})}function Lm(e,t,n,r){if(t=e.querySelector(`link[rel="preload"][as="style"][`+t+`]`)){if(!0!==t[Dt]){r.loading=1;return}}else t=e.createElement(`link`),t[Dt]=!0,t.onload=t.onerror=Pt.bind(null,t),np(t,`link`,n),Nt(t),e.head.appendChild(t);r.preload=t,t.addEventListener(`load`,function(){return r.loading|=1}),t.addEventListener(`error`,function(){return r.loading|=2})}function Rm(e){return`[src="`+Qt(e)+`"]`}function zm(e){return`script[async]`+e}function Bm(e,t,n){if(t.count++,t.instance===null)switch(t.type){case`style`:var r=e.querySelector(`style[data-href~="`+Qt(n.href)+`"]`);if(r)return t.instance=r,Nt(r),r;var a=D({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return r=(e.ownerDocument||e).createElement(`style`),Nt(r),np(r,`style`,a),Vm(r,n.precedence,e),t.instance=r;case`stylesheet`:a=Pm(n.href);var o=e.querySelector(Fm(a));if(o)return t.state.loading|=4,t.instance=o,Nt(o),o;r=Im(n),(a=vm.get(a))&&Hm(r,a),o=(e.ownerDocument||e).createElement(`link`),Nt(o);var s=o;return s._p=new Promise(function(e,t){s.onload=e,s.onerror=t}),np(o,`link`,r),t.state.loading|=4,Vm(o,n.precedence,e),t.instance=o;case`script`:return o=Rm(n.src),(a=e.querySelector(zm(o)))?(t.instance=a,Nt(a),a):(r=n,(a=vm.get(o))&&(r=D({},n),Um(r,a)),e=e.ownerDocument||e,a=e.createElement(`script`),Nt(a),np(a,`link`,r),e.head.appendChild(a),t.instance=a);case`void`:return null;default:throw Error(i(443,t.type))}else t.type===`stylesheet`&&!(t.state.loading&4)&&(r=t.instance,t.state.loading|=4,Vm(r,n.precedence,e));return t.instance}function Vm(e,t,n){for(var r=n.querySelectorAll(`link[rel="stylesheet"][data-precedence],style[data-precedence]`),i=r.length?r[r.length-1]:null,a=i,o=0;o<r.length;o++){var s=r[o];if(s.dataset.precedence===t)a=s;else if(a!==i)break}a?a.parentNode.insertBefore(e,a.nextSibling):(t=n.nodeType===9?n.head:n,t.insertBefore(e,t.firstChild))}function Hm(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.title??=t.title}function Um(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.integrity??=t.integrity}var Wm=null;function Gm(e,t,n){if(Wm===null){var r=new Map,i=Wm=new Map;i.set(n,r)}else i=Wm,r=i.get(n),r||(r=new Map,i.set(n,r));if(r.has(e))return r;for(r.set(e,null),n=n.getElementsByTagName(e),i=0;i<n.length;i++){var a=n[i];if(!(a[Et]||a[yt]||e===`link`&&a.getAttribute(`rel`)===`stylesheet`)&&a.namespaceURI!==`http://www.w3.org/2000/svg`){var o=a.getAttribute(t)||``;o=e+o;var s=r.get(o);s?s.push(a):r.set(o,[a])}}return r}function Km(e,t,n){e=e.ownerDocument||e,e.head.insertBefore(n,t===`title`?e.querySelector(`head > title`):null)}function qm(e,t,n){if(n===1||t.itemProp!=null)return!1;switch(e){case`meta`:case`title`:return!0;case`style`:if(typeof t.precedence!=`string`||typeof t.href!=`string`||t.href===``)break;return!0;case`link`:if(typeof t.rel!=`string`||typeof t.href!=`string`||t.href===``||t.onLoad||t.onError)break;switch(t.rel){case`stylesheet`:return e=t.disabled,typeof t.precedence==`string`&&e==null;default:return!0}case`script`:if(t.async&&typeof t.async!=`function`&&typeof t.async!=`symbol`&&!t.onLoad&&!t.onError&&t.src&&typeof t.src==`string`)return!0}return!1}function Jm(e,t){return e===`img`&&t.src!=null&&t.src!==``&&t.onLoad==null&&t.loading!==`lazy`}function Ym(e){return!(e.type===`stylesheet`&&!(e.state.loading&3))}function Xm(e){return(e.width||100)*(e.height||100)*(typeof devicePixelRatio==`number`?devicePixelRatio:1)*.25}function Zm(e,t){typeof t.decode==`function`&&(e.imgCount++,t.complete||(e.imgBytes+=Xm(t),e.suspenseyImages.push(t)),e=rh.bind(e),t.decode().then(e,e))}function Qm(e,t,n,r){if(n.type===`stylesheet`&&(typeof r.media!=`string`||!1!==matchMedia(r.media).matches)&&!(n.state.loading&4)){if(n.instance===null){var i=Pm(r.href),a=t.querySelector(Fm(i));if(a){t=a._p,typeof t==`object`&&t&&typeof t.then==`function`&&(e.count++,e=nh.bind(e),t.then(e,e)),n.state.loading|=4,n.instance=a,Nt(a);return}a=t.ownerDocument||t,r=Im(r),(i=vm.get(i))&&Hm(r,i),a=a.createElement(`link`),Nt(a);var o=a;o._p=new Promise(function(e,t){o.onload=e,o.onerror=t}),np(a,`link`,r),n.instance=a}e.stylesheets===null&&(e.stylesheets=new Map),e.stylesheets.set(n,t),(t=n.state.preload)&&!(n.state.loading&3)&&(e.count++,n=nh.bind(e),t.addEventListener(`load`,n),t.addEventListener(`error`,n))}}var $m=0;function eh(e,t){return e.stylesheets&&e.count===0&&ah(e,e.stylesheets),0<e.count||0<e.imgCount?function(n){var r=setTimeout(function(){if(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}},6e4+t);0<e.imgBytes&&$m===0&&($m=62500*op());var i=setTimeout(function(){if(e.waitingForImages=!1,e.count===0&&(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend)){var t=e.unsuspend;e.unsuspend=null,t()}},(e.imgBytes>$m?50:800)+t);return e.unsuspend=n,function(){e.unsuspend=null,clearTimeout(r),clearTimeout(i)}}:null}function th(e){if(e.count===0&&(e.imgCount===0||!e.waitingForImages)){if(e.stylesheets)ah(e,e.stylesheets);else if(e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}}}function nh(){this.count--,th(this)}function rh(){this.imgCount--,th(this)}var ih=null;function ah(e,t){e.stylesheets=null,e.unsuspend!==null&&(e.count++,ih=new Map,t.forEach(oh,e),ih=null,nh.call(e))}function oh(e,t){if(!(t.state.loading&4)){var n=ih.get(e);if(n)var r=n.get(null);else{n=new Map,ih.set(e,n);for(var i=e.querySelectorAll(`link[data-precedence],style[data-precedence]`),a=0;a<i.length;a++){var o=i[a];(o.nodeName===`LINK`||o.getAttribute(`media`)!==`not all`)&&(n.set(o.dataset.precedence,o),r=o)}r&&n.set(null,r)}i=t.instance,o=i.getAttribute(`data-precedence`),a=n.get(o)||r,a===r&&n.set(null,i),n.set(o,i),this.count++,r=nh.bind(this),i.addEventListener(`load`,r),i.addEventListener(`error`,r),a?a.parentNode.insertBefore(i,a.nextSibling):(e=e.nodeType===9?e.head:e,e.insertBefore(i,e.firstChild)),t.state.loading|=4}}var sh={$$typeof:M,Provider:null,Consumer:null,_currentValue:me,_currentValue2:me,_threadCount:0};function ch(e,t,n,r,i,a,o,s,c){this.tag=1,this.containerInfo=e,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=ct(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=ct(0),this.hiddenUpdates=ct(null),this.identifierPrefix=r,this.onUncaughtError=i,this.onCaughtError=a,this.onRecoverableError=o,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=c,this.transitionTypes=null,this.incompleteTransitions=new Map}function lh(e,t,n,r,i,a,o,s,c,l,u,d){return e=new ch(e,t,n,o,c,l,u,d,s),t=1,!0===a&&(t|=24),a=Oi(3,null,null,t),e.current=a,a.stateNode=e,t=Oa(),t.refCount++,e.pooledCache=t,t.refCount++,a.memoizedState={element:r,isDehydrated:n,cache:t},uo(a),e}function uh(e){return e?(e=Ei,e):Ei}function dh(e,t,n,r,i,a){i=uh(i),r.context===null?r.context=i:r.pendingContext=i,r=po(t),r.payload={element:n},a=a===void 0?null:a,a!==null&&(r.callback=a),n=mo(e,r,t),n!==null&&(Nd(n,e,t),ho(n,e,t))}function fh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ph(e,t){fh(e,t),(e=e.alternate)&&fh(e,t)}function mh(e){if(e.tag===13||e.tag===31){var t=Ci(e,67108864);t!==null&&Nd(t,e,67108864),ph(e,67108864)}}function hh(e){if(e.tag===13||e.tag===31){var t=Ad();t=mt(t);var n=Ci(e,t);n!==null&&Nd(n,e,t),ph(e,t)}}var gh=!0;function _h(e,t,n,r){var i=L.T;L.T=null;var a=R.p;try{R.p=2,yh(e,t,n,r)}finally{R.p=a,L.T=i}}function vh(e,t,n,r){var i=L.T;L.T=null;var a=R.p;try{R.p=8,yh(e,t,n,r)}finally{R.p=a,L.T=i}}function yh(e,t,n,r){if(gh){var i=bh(r);if(i===null)Kf(e,t,r,xh,n),Mh(e,r);else if(Ph(i,e,t,n,r))r.stopPropagation();else if(Mh(e,r),t&4&&-1<jh.indexOf(e)){for(;i!==null;){var a=At(i);if(a!==null)switch(a.tag){case 3:if(a=a.stateNode,a.current.memoizedState.isDehydrated){var o=nt(a.pendingLanes);if(o!==0){var s=a;for(s.pendingLanes|=2,s.entangledLanes|=2;o;){var c=1<<31-Ye(o);s.entanglements[1]|=c,o&=~c}Ef(a),!(J&6)&&(hd=Le()+500,Df(0,!1))}}break;case 31:case 13:s=Ci(a,2),s!==null&&Nd(s,a,2),Rd(),ph(a,2)}if(a=bh(r),a===null&&Kf(e,t,r,xh,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Kf(e,t,r,null,n)}}function bh(e){return e=gn(e),Sh(e)}var xh=null;function Sh(e){if(xh=null,e=kt(e),e!==null){var t=o(e);if(t===null)e=null;else{var n=t.tag;if(n===13){if(e=s(t),e!==null)return e;e=null}else if(n===31){if(e=c(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null)}}return xh=e,null}function Ch(e){switch(e){case`beforetoggle`:case`cancel`:case`click`:case`close`:case`contextmenu`:case`copy`:case`cut`:case`auxclick`:case`dblclick`:case`dragend`:case`dragstart`:case`drop`:case`focusin`:case`focusout`:case`input`:case`invalid`:case`keydown`:case`keypress`:case`keyup`:case`mousedown`:case`mouseup`:case`paste`:case`pause`:case`play`:case`pointercancel`:case`pointerdown`:case`pointerup`:case`ratechange`:case`reset`:case`seeked`:case`submit`:case`toggle`:case`touchcancel`:case`touchend`:case`touchstart`:case`volumechange`:case`change`:case`selectionchange`:case`textInput`:case`compositionstart`:case`compositionend`:case`compositionupdate`:case`beforeblur`:case`afterblur`:case`beforeinput`:case`blur`:case`fullscreenchange`:case`fullscreenerror`:case`focus`:case`hashchange`:case`popstate`:case`select`:case`selectstart`:return 2;case`drag`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`mousemove`:case`mouseout`:case`mouseover`:case`pointermove`:case`pointerout`:case`pointerover`:case`resize`:case`scroll`:case`touchmove`:case`wheel`:case`mouseenter`:case`mouseleave`:case`pointerenter`:case`pointerleave`:return 8;case`message`:switch(Re()){case ze:return 2;case Be:return 8;case Ve:case He:return 32;case Ue:return 268435456;default:return 32}default:return 32}}var wh=!1,Th=null,Eh=null,Dh=null,Oh=new Map,kh=new Map,Ah=[],jh=`mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset`.split(` `);function Mh(e,t){switch(e){case`focusin`:case`focusout`:Th=null;break;case`dragenter`:case`dragleave`:Eh=null;break;case`mouseover`:case`mouseout`:Dh=null;break;case`pointerover`:case`pointerout`:Oh.delete(t.pointerId);break;case`gotpointercapture`:case`lostpointercapture`:kh.delete(t.pointerId)}}function Nh(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=At(t),t!==null&&mh(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Ph(e,t,n,r,i){switch(t){case`focusin`:return Th=Nh(Th,e,t,n,r,i),!0;case`dragenter`:return Eh=Nh(Eh,e,t,n,r,i),!0;case`mouseover`:return Dh=Nh(Dh,e,t,n,r,i),!0;case`pointerover`:var a=i.pointerId;return Oh.set(a,Nh(Oh.get(a)||null,e,t,n,r,i)),!0;case`gotpointercapture`:return a=i.pointerId,kh.set(a,Nh(kh.get(a)||null,e,t,n,r,i)),!0}return!1}function Fh(e){var t=kt(e.target);if(t!==null){var n=o(t);if(n!==null){if(t=n.tag,t===13){if(t=s(n),t!==null){e.blockedOn=t,_t(e.priority,function(){hh(n)});return}}else if(t===31){if(t=c(n),t!==null){e.blockedOn=t,_t(e.priority,function(){hh(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ih(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=bh(e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);hn=r,n.target.dispatchEvent(r),hn=null}else return t=At(n),t!==null&&mh(t),e.blockedOn=n,!1;t.shift()}return!0}function Lh(e,t,n){Ih(e)&&n.delete(t)}function Rh(){wh=!1,Th!==null&&Ih(Th)&&(Th=null),Eh!==null&&Ih(Eh)&&(Eh=null),Dh!==null&&Ih(Dh)&&(Dh=null),Oh.forEach(Lh),kh.forEach(Lh)}function zh(e,n){e.blockedOn===n&&(e.blockedOn=null,wh||(wh=!0,t.unstable_scheduleCallback(t.unstable_NormalPriority,Rh)))}var Bh=null;function Vh(e){Bh!==e&&(Bh=e,t.unstable_scheduleCallback(t.unstable_NormalPriority,function(){Bh===e&&(Bh=null);for(var t=0;t<e.length;t+=3){var n=e[t],r=e[t+1],i=e[t+2];if(typeof r!=`function`){if(Sh(r||n)===null)continue;break}var a=At(n);a!==null&&(e.splice(t,3),t-=3,Zs(a,{pending:!0,data:i,method:n.method,action:r},r,i))}}))}function Hh(e){function t(t){return zh(t,e)}Th!==null&&zh(Th,e),Eh!==null&&zh(Eh,e),Dh!==null&&zh(Dh,e),Oh.forEach(t),kh.forEach(t);for(var n=0;n<Ah.length;n++){var r=Ah[n];r.blockedOn===e&&(r.blockedOn=null)}for(;0<Ah.length&&(n=Ah[0],n.blockedOn===null);)Fh(n),n.blockedOn===null&&Ah.shift();if(n=(e.ownerDocument||e).$$reactFormReplay,n!=null)for(r=0;r<n.length;r+=3){var i=n[r],a=n[r+1],o=i[bt]||null;if(typeof a==`function`)o||Vh(n);else if(o){var s=null;if(a&&a.hasAttribute(`formAction`)){if(i=a,o=a[bt]||null)s=o.formAction;else if(Sh(i)!==null)continue}else s=o.action;typeof s==`function`?n[r+1]=s:(n.splice(r,3),r-=3),Vh(n)}}}function Uh(){function e(e){e.canIntercept&&e.info===`react-transition`&&e.intercept({handler:function(){return new Promise(function(e){return i=e})},focusReset:`manual`,scroll:`manual`})}function t(){i!==null&&(i(),i=null),r||setTimeout(n,20)}function n(){if(!r&&!navigation.transition){var e=navigation.currentEntry;e&&e.url!=null&&navigation.navigate(e.url,{state:e.getState(),info:`react-transition`,history:`replace`})}}if(typeof navigation==`object`){var r=!1,i=null;return navigation.addEventListener(`navigate`,e),navigation.addEventListener(`navigatesuccess`,t),navigation.addEventListener(`navigateerror`,t),setTimeout(n,100),function(){r=!0,navigation.removeEventListener(`navigate`,e),navigation.removeEventListener(`navigatesuccess`,t),navigation.removeEventListener(`navigateerror`,t),i!==null&&(i(),i=null)}}}function Wh(e){this._internalRoot=e}Gh.prototype.render=Wh.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(i(409));var n=t.current;dh(n,Ad(),e,t,null,null)},Gh.prototype.unmount=Wh.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;dh(e.current,2,null,e,null,null),Rd(),t[xt]=null}};function Gh(e){this._internalRoot=e}Gh.prototype.unstable_scheduleHydration=function(e){if(e){var t=gt();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Ah.length&&t!==0&&t<Ah[n].priority;n++);Ah.splice(n,0,e),n===0&&Fh(e)}};var Kh=n.version;if(Kh!==`19.3.0`)throw Error(i(527,Kh,`19.3.0`));R.findDOMNode=function(e){var t=e._reactInternals;if(t===void 0)throw typeof e.render==`function`?Error(i(188)):(e=Object.keys(e).join(`,`),Error(i(268,e)));return e=d(t),e=e===null?null:p(e),e=e===null?null:e.stateNode,e};var qh={bundleType:0,version:`19.3.0`,rendererPackageName:`react-dom`,currentDispatcherRef:L,reconcilerVersion:`19.3.0`};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`){var Jh=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Jh.isDisabled&&Jh.supportsFiber)try{Ke=Jh.inject(qh),qe=Jh}catch{}}e.createRoot=function(e,t){if(!a(e))throw Error(i(299));var n=!1,r=``,o=bc,s=xc,c=Sc;return t!=null&&(!0===t.unstable_strictMode&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onUncaughtError!==void 0&&(o=t.onUncaughtError),t.onCaughtError!==void 0&&(s=t.onCaughtError),t.onRecoverableError!==void 0&&(c=t.onRecoverableError)),t=lh(e,1,!1,null,null,n,r,null,o,s,c,Uh),e[xt]=t.current,Wf(e),new Wh(t)}})),g=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=h()})),_=c(u(),1),v=g(),y=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.fragment`);function r(e,n,r){var i=null;if(r!==void 0&&(i=``+r),n.key!==void 0&&(i=``+n.key),`key`in n)for(var a in r={},n)a!==`key`&&(r[a]=n[a]);else r=n;return n=r.ref,{$$typeof:t,type:e,key:i,ref:n===void 0?null:n,props:r}}e.Fragment=n,e.jsx=r,e.jsxs=r})),b=o(((e,t)=>{t.exports=y()})),x=b(),S={models:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h4A1.5 1.5 0 0 1 12 6.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 10.5zm8 0A1.5 1.5 0 0 1 14.5 5h3A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-3a1.5 1.5 0 0 1-1.5-1.5zm-8 8A1.5 1.5 0 0 1 6.5 13h4a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 17.5z`}),chat:(0,x.jsx)(`path`,{d:`M5.5 7.5A2.5 2.5 0 0 1 8 5h8a2.5 2.5 0 0 1 2.5 2.5v5A2.5 2.5 0 0 1 16 15h-3.7l-3.6 3.2a.7.7 0 0 1-1.2-.52V15A2.5 2.5 0 0 1 5.5 12.5z`}),activity:(0,x.jsx)(`path`,{d:`M4.5 13h3l1.8-5.5 3.2 9 2.1-6.5h4.9`}),settings:(0,x.jsx)(`path`,{d:`M12 8.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6Zm0-3.2v2m0 10v2m7-7h-2M7 12H5m11.95-4.95-1.42 1.42M8.47 15.53l-1.42 1.42m9.9 0-1.42-1.42M8.47 8.47 7.05 7.05`}),gallery:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h11A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 17.5zm3 2h8m-8 3h8m-8 3h5`}),command:(0,x.jsx)(`path`,{d:`M8 8.5A2.5 2.5 0 1 1 5.5 6 2.5 2.5 0 0 1 8 8.5Zm0 7A2.5 2.5 0 1 1 5.5 13 2.5 2.5 0 0 1 8 15.5Zm8-7A2.5 2.5 0 1 1 13.5 6 2.5 2.5 0 0 1 16 8.5Zm0 7A2.5 2.5 0 1 1 13.5 13 2.5 2.5 0 0 1 16 15.5Z`}),help:(0,x.jsx)(`path`,{d:`M9.5 9a2.5 2.5 0 1 1 4.4 1.62c-.68.72-1.9 1.18-1.9 2.38v.25m0 3.25h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z`}),menu:(0,x.jsx)(`path`,{d:`M5 7h14M5 12h14M5 17h14`}),close:(0,x.jsx)(`path`,{d:`m7 7 10 10M17 7 7 17`}),search:(0,x.jsx)(`path`,{d:`m16.5 16.5 3 3M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14Z`}),warning:(0,x.jsx)(`path`,{d:`M12 4.8 21 19H3zm0 5.2v4m0 2.5h.01`}),key:(0,x.jsx)(`path`,{d:`M14.5 9.5A4.5 4.5 0 1 0 11 13.9l2.1 2.1H16v2h2v2h2v-2.9l-4.6-4.6a4.5 4.5 0 0 0-.9-3Z`}),schema:(0,x.jsx)(`path`,{d:`M6 5h7l5 5v9H6zm7 0v5h5M9 13h6M9 16h6M9 10h2`}),list:(0,x.jsx)(`path`,{d:`M9 6.5h10M9 12h10M9 17.5h10M5 6.5h.01M5 12h.01M5 17.5h.01`}),attach:(0,x.jsx)(`path`,{d:`M16.5 7.5v8a4.5 4.5 0 0 1-9 0V6.75a3 3 0 0 1 6 0V15a1.5 1.5 0 0 1-3 0V7.5`}),copy:(0,x.jsx)(`path`,{d:`M9.5 9h8A1.5 1.5 0 0 1 19 10.5v8a1.5 1.5 0 0 1-1.5 1.5h-8A1.5 1.5 0 0 1 8 18.5v-8A1.5 1.5 0 0 1 9.5 9ZM16 9V6.5A1.5 1.5 0 0 0 14.5 5h-8A1.5 1.5 0 0 0 5 6.5v8A1.5 1.5 0 0 0 6.5 16H8`}),edit:(0,x.jsx)(`path`,{d:`M5 19h4L18.5 9.5a2.83 2.83 0 0 0-4-4L5 15zm8.5-12.5 4 4`}),retry:(0,x.jsx)(`path`,{d:`M5 12a7 7 0 0 1 12-4.9L19 9m0-4.5V9h-4.5M19 12a7 7 0 0 1-12 4.9L5 15m0 4.5V15h4.5`}),trash:(0,x.jsx)(`path`,{d:`M5 7h14M10 11v5.5m4-5.5v5.5M6.5 7l.75 11.1A2 2 0 0 0 9.24 20h5.52a2 2 0 0 0 2-1.9L17.5 7M9.5 7V5.5A1.5 1.5 0 0 1 11 4h2a1.5 1.5 0 0 1 1.5 1.5V7`}),info:(0,x.jsx)(`path`,{d:`M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-10v5.5m0-8.5h.01`}),details:(0,x.jsx)(`path`,{d:`M6 19v-7m6 7V5m6 14v-4`})};function C(e){return(0,x.jsx)(`svg`,{className:e.className??`ds-icon`,"aria-hidden":`true`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`1.8`,strokeLinecap:`round`,strokeLinejoin:`round`,children:S[e.name]})}var w=(0,_.forwardRef)(function({children:e,onClick:t,onDoubleClick:n,onMouseDown:r,onMouseEnter:i,onKeyDown:a,type:o=`button`,disabled:s=!1,variant:c=`secondary`,size:l=`medium`,shape:u=`default`,fullWidth:d=!1,icon:f,iconPosition:p=`left`,iconOnly:m=!1,inline:h=!1,loading:g=!1,active:v=!1,ariaLabel:y,title:b,role:S,id:C,tabIndex:w,className:T=``,style:E,"aria-pressed":D,"aria-checked":O,"aria-selected":k,"aria-expanded":A,"aria-haspopup":j,"aria-controls":ee,"aria-describedby":te,"aria-busy":ne,"data-testid":M},re){let ie=(0,_.useCallback)(e=>{!s&&!g&&t&&t(e)},[s,g,t]),N=[`button`,`button--${c}`,`button--${l}`,u!=="default"&&`button--${u}`,d&&`button--full-width`,m&&`button--icon-only`,h&&`button--inline`,g&&`button--loading`,v&&`button--active`,T].filter(Boolean).join(` `),ae=f??(m?e:void 0),oe=!m&&e,se=ae&&!g;return(0,x.jsxs)(`button`,{ref:re,type:o,onClick:ie,onDoubleClick:n,onMouseDown:r,onMouseEnter:i,onKeyDown:a,disabled:s||g,className:N,"aria-label":y??(m&&typeof e==`string`?e:void 0),title:b,role:S,id:C,tabIndex:w,style:E,"aria-pressed":D,"aria-checked":O,"aria-selected":k,"aria-expanded":A,"aria-haspopup":j,"aria-controls":ee,"aria-describedby":te,"aria-busy":ne,"data-testid":M,children:[g&&(0,x.jsx)(`span`,{className:`button__loading-spinner`,"aria-hidden":`true`}),se&&p===`left`&&(0,x.jsx)(`span`,{className:`button__icon`,"aria-hidden":`true`,children:ae}),oe&&(h?e:(0,x.jsx)(`span`,{className:`button__text`,children:e})),se&&p===`right`&&(0,x.jsx)(`span`,{className:`button__icon`,"aria-hidden":`true`,children:ae})]})});function T({children:e,variant:t=`default`,size:n=`small`,className:r=``}){let i=[`badge`,`badge--${t}`,`badge--${n}`,r].filter(Boolean).join(` `);return(0,x.jsx)(`span`,{className:i,children:e})}function E(e){switch(e){case`running`:return`success`;case`preparing`:return`info`;case`idle`:return`default`;case`busy`:return`primary`;case`stopping`:return`warning`;case`terminated`:return`default`;case`error`:return`danger`}}function D(e){return e===`preparing`||e===`stopping`||e===`busy`}function O({state:e,label:t,size:n=`small`,pulse:r,className:i=``}){let a=r??D(e),o=[`status-tag`,`status-tag--${e}`,i].filter(Boolean).join(` `);return(0,x.jsxs)(T,{variant:E(e),size:n,className:o,children:[a&&(0,x.jsx)(`span`,{className:`status-tag__indicator`,"aria-hidden":`true`,"data-testid":`status-tag-indicator`}),(0,x.jsx)(`span`,{className:`status-tag__label`,children:t})]})}var k=(0,_.memo)(O);function A({value:e,variant:t=`primary`,size:n=`md`,showLabel:r=!1,label:i,className:a=``,animated:o=!0,ariaLabel:s}){let c=e===null,l=e===null?0:Math.max(0,Math.min(100,e)),u=[`progress-bar`,`progress-bar--${t}`,`progress-bar--${n}`,c?`progress-bar--indeterminate`:``,o?`progress-bar--animated`:``,a].filter(Boolean).join(` `),d=i??(r&&!c?`${l.toFixed(0)}%`:null);return(0,x.jsxs)(`div`,{className:`progress-bar-wrapper`,children:[(0,x.jsx)(`div`,{className:u,role:`progressbar`,"aria-valuenow":c?void 0:l,"aria-valuemin":0,"aria-valuemax":100,"aria-label":s,"aria-busy":c,children:(0,x.jsx)(`div`,{className:`progress-bar__fill`,style:c?void 0:{width:`${String(l)}%`}})}),d&&(0,x.jsx)(`span`,{className:`progress-bar__label`,children:d})]})}function j({illustration:e,title:t,description:n,primaryAction:r,secondaryAction:i,children:a,className:o=``,showIllustration:s=!0}){let c=r?.onClick,l=i?.onClick,u=(0,_.useCallback)(()=>{c?.()},[c]),d=(0,_.useCallback)(()=>{l?.()},[l]),f=(0,_.useMemo)(()=>[`empty-state`,o].filter(Boolean).join(` `),[o]);return(0,x.jsxs)(`div`,{className:f,role:`status`,"aria-live":`polite`,children:[s&&e&&(0,x.jsx)(`div`,{className:`empty-state__illustration`,children:e}),(0,x.jsxs)(`div`,{className:`empty-state__content`,children:[(0,x.jsx)(`h3`,{className:`empty-state__title`,children:t}),(0,x.jsx)(`p`,{className:`empty-state__description`,children:n}),a]}),(r||i)&&(0,x.jsxs)(`div`,{className:`empty-state__actions`,children:[r&&(0,x.jsx)(w,{variant:`primary`,onClick:u,className:`empty-state__action-btn empty-state__action-btn--primary`,children:r.label}),i&&(0,x.jsx)(x.Fragment,{children:i.href?(0,x.jsx)(`a`,{href:i.href,className:`empty-state__action-link`,onClick:d,children:i.label}):(0,x.jsx)(w,{variant:`ghost`,onClick:d,className:`empty-state__action-btn empty-state__action-btn--secondary`,children:i.label})})]})]})}var ee=(0,_.memo)(j);function te({tabs:e,groups:t=[],defaultTab:n,activeTab:r,onTabChange:i,className:a=``,panelClassName:o=``,ariaLabel:s,showOverflowControls:c=!0,showGroupLabels:l=!0,overflowMode:u,variant:d=`underlined`,fillContainer:f=!1,renderPanel:p=!0,moreTabsLabel:m=`More tabs`,selectTabLabel:h=`Select tab`,scrollLeftLabel:g=`Scroll left`,scrollRightLabel:v=`Scroll right`}){let y=u??(t.length>0?`menu`:`dropdown`),[b,S]=(0,_.useState)(n||e[0]?.id||``),[C,w]=(0,_.useState)(!1),[T,E]=(0,_.useState)(!1),[D,O]=(0,_.useState)(!1),[k,A]=(0,_.useState)(!1),[j,ee]=(0,_.useState)(!1),[te,ne]=(0,_.useState)(-1),M=(0,_.useRef)(null),re=(0,_.useRef)(new Map),ie=(0,_.useRef)(null),N=(0,_.useRef)(null),ae=(0,_.useRef)(null),oe=(0,_.useRef)([]),se=(0,_.useRef)(null),ce=(0,_.useRef)(0),P=(0,_.useRef)(0),F=r??b,le=(0,_.useMemo)(()=>t.length>0?t.map(t=>({group:t,tabs:e.filter(e=>e.groupId===t.id)})):[{group:null,tabs:e}],[t,e]),ue=(0,_.useMemo)(()=>le.flatMap(({tabs:e})=>e),[le]),de=(0,_.useCallback)(e=>{r||S(e),i?.(e),ee(!1)},[r,i]),I=(0,_.useCallback)((t,n)=>{let r;switch(t.key){case`ArrowRight`:t.preventDefault(),r=n===e.length-1?0:n+1;break;case`ArrowLeft`:t.preventDefault(),r=n===0?e.length-1:n-1;break;case`Home`:t.preventDefault(),r=0;break;case`End`:t.preventDefault(),r=e.length-1;break;default:return}let i=e[r];i&&(de(i.id),re.current.get(i.id)?.focus())},[e,de]),fe=(0,_.useCallback)((e,t)=>{t?re.current.set(e,t):re.current.delete(e)},[]),pe=(0,_.useCallback)(()=>{let e=M.current;if(!e||!c){w(!1),E(!1);return}let{scrollLeft:t,scrollWidth:n,clientWidth:r}=e;w(t>0),E(t+r<n-1)},[c]),L=(0,_.useCallback)(e=>{let t=M.current;if(!t)return;let n=t.clientWidth*.75;t.scrollBy({left:e===`left`?-n:n,behavior:`smooth`}),se.current!==null&&clearTimeout(se.current),se.current=setTimeout(pe,300)},[pe]),R=(0,_.useCallback)(()=>{if(!M.current||D)return;let e=re.current.get(F);if(!e)return;let t=M.current,n=e.getBoundingClientRect(),r=t.getBoundingClientRect();n.left<r.left?t.scrollLeft-=r.left-n.left+8:n.right>r.right&&(t.scrollLeft+=n.right-r.right+8)},[F,D]),me=(0,_.useCallback)(()=>{typeof window>`u`||y!==`dropdown`||O(window.innerWidth<480)},[y]),he=(0,_.useCallback)(()=>{A(e=>!e)},[]),ge=(0,_.useCallback)(e=>{e.key===`Escape`&&k&&(e.preventDefault(),A(!1))},[k]),_e=(0,_.useCallback)(e=>{e.touches[0]&&(ce.current=e.touches[0].clientX)},[]),ve=(0,_.useCallback)(e=>{e.touches[0]&&(P.current=e.touches[0].clientX)},[]),z=(0,_.useCallback)(()=>{if(!ce.current||!P.current)return;let t=ce.current-P.current;if(Math.abs(t)>50){let n=e.findIndex(e=>e.id===F),r=e[n+1],i=e[n-1];t>0&&r?de(r.id):t<0&&i&&de(i.id)}ce.current=0,P.current=0},[F,e,de]),ye=(0,_.useCallback)(e=>{let t=ue.length;switch(e.key){case`Escape`:e.preventDefault(),ee(!1),ne(-1),ae.current?.focus();break;case`ArrowDown`:e.preventDefault(),ne(e=>{let n=e>=t-1?0:e+1;return oe.current[n]?.focus(),n});break;case`ArrowUp`:e.preventDefault(),ne(e=>{let n=e<=0?t-1:e-1;return oe.current[n]?.focus(),n});break;case`Home`:e.preventDefault(),ne(0),oe.current[0]?.focus();break;case`End`:e.preventDefault(),ne(t-1),oe.current[t-1]?.focus()}},[ue.length]),be=(0,_.useCallback)(e=>{e.key===`Escape`&&j&&(e.preventDefault(),ee(!1),ne(-1))},[j]);(0,_.useEffect)(()=>{if(!k)return;let e=e=>{ie.current&&!ie.current.contains(e.target)&&A(!1)};return document.addEventListener(`mousedown`,e),()=>{document.removeEventListener(`mousedown`,e)}},[k]),(0,_.useEffect)(()=>{if(!j)return;let e=e=>{N.current&&!N.current.contains(e.target)&&(ee(!1),ne(-1))},t=e=>{e.key===`Escape`&&(ee(!1),ne(-1),ae.current?.focus())};return document.addEventListener(`mousedown`,e),document.addEventListener(`keydown`,t),()=>{document.removeEventListener(`mousedown`,e),document.removeEventListener(`keydown`,t)}},[j]),(0,_.useEffect)(()=>{if(j&&oe.current.length>0){let e=ue.findIndex(e=>e.id===F),t=e>=0?e:0;ne(t),requestAnimationFrame(()=>{oe.current[t]?.focus()})}},[j,ue,F]),(0,_.useEffect)(()=>{if(!e.find(e=>e.id===F)){let t=e[0];t&&de(t.id)}},[e,F,de]),(0,_.useEffect)(()=>{me(),pe();let e=()=>{me(),pe()};return window.addEventListener(`resize`,e),()=>{window.removeEventListener(`resize`,e)}},[me,pe]),(0,_.useEffect)(()=>{R()},[F,R]),(0,_.useEffect)(()=>{pe()},[e,pe]),(0,_.useEffect)(()=>()=>{se.current!==null&&clearTimeout(se.current)},[]);let xe=e.find(e=>e.id===F)?.content,Se=e.find(e=>e.id===F)?.label,Ce=t.length>0;oe.current=[];let we=(e,t)=>{let n=e.id===F;return(0,x.jsxs)(`button`,{ref:t=>{fe(e.id,t)},role:`tab`,"aria-selected":n,"aria-controls":p?`tabpanel-${e.id}`:void 0,id:`tab-${e.id}`,tabIndex:n?0:-1,className:`tabs__tab${n?` tabs__tab--active`:``}`,onClick:()=>{de(e.id)},onKeyDown:e=>{I(e,t),(e.key===`ArrowLeft`||e.key===`ArrowRight`||e.key===`Home`||e.key===`End`)&&e.stopPropagation()},children:[e.labelPrefix,(0,x.jsx)(`span`,{className:`tabs__tab-label`,children:e.label}),e.labelExtra]},e.id)},Te=()=>Ce?le.map(({group:t,tabs:n},r)=>(0,x.jsxs)(`div`,{className:`tabs__group`,children:[t&&(0,x.jsxs)(x.Fragment,{children:[r>0&&(0,x.jsx)(`div`,{className:`tabs__separator`,role:`separator`,"aria-hidden":`true`}),l&&(0,x.jsx)(`span`,{className:`tabs__group-label`,children:t.label})]}),(0,x.jsx)(`div`,{className:`tabs__group-tabs`,children:n.map(t=>{let n=e.findIndex(e=>e.id===t.id);return we(t,n)})})]},t?.id||`default`)):e.map((e,t)=>we(e,t)),Ee=()=>y!==`menu`||!c?null:(0,x.jsxs)(`div`,{className:`tabs__overflow`,ref:N,children:[(0,x.jsx)(`button`,{ref:ae,type:`button`,className:`tabs__overflow-btn`,onClick:()=>{ee(!j)},onKeyDown:be,"aria-label":m,"aria-expanded":j,"aria-haspopup":`menu`,children:(0,x.jsxs)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`,children:[(0,x.jsx)(`circle`,{cx:`12`,cy:`12`,r:`1`}),(0,x.jsx)(`circle`,{cx:`12`,cy:`5`,r:`1`}),(0,x.jsx)(`circle`,{cx:`12`,cy:`19`,r:`1`})]})}),j&&(0,x.jsx)(`div`,{className:`tabs__overflow-menu`,role:`menu`,onKeyDown:ye,children:le.map(({group:e,tabs:t})=>(0,x.jsxs)(`div`,{className:`tabs__overflow-group`,children:[e&&l&&(0,x.jsx)(`div`,{className:`tabs__overflow-group-label`,children:e.label}),t.map(e=>{let t=e.id===F,n=ue.findIndex(t=>t.id===e.id);return(0,x.jsxs)(`button`,{ref:e=>{e&&(oe.current[n]=e)},role:`menuitem`,tabIndex:te===n?0:-1,className:`tabs__overflow-item ${t?`tabs__overflow-item--active`:``}`,onClick:()=>{de(e.id)},children:[e.labelPrefix,(0,x.jsx)(`span`,{children:e.label}),e.labelExtra]},e.id)})]},e?.id||`default`))})]}),De=d===`segmented`||d===`compact`;return y===`dropdown`&&D&&!De?(0,x.jsxs)(`div`,{className:`tabs tabs--mobile-dropdown ${a}`,children:[(0,x.jsxs)(`div`,{className:`tabs__dropdown-container`,ref:ie,children:[(0,x.jsxs)(`button`,{className:`tabs__dropdown-trigger`,onClick:he,onKeyDown:ge,"aria-expanded":k,"aria-haspopup":`listbox`,"aria-label":s||h,children:[(0,x.jsx)(`span`,{className:`tabs__dropdown-label`,children:Se}),(0,x.jsx)(`svg`,{className:`tabs__dropdown-arrow ${k?`tabs__dropdown-arrow--open`:``}`,width:`16`,height:`16`,viewBox:`0 0 16 16`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M4 6L8 10L12 6`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})]}),k&&(0,x.jsx)(`div`,{className:`tabs__dropdown-menu`,role:`listbox`,children:e.map(e=>(0,x.jsx)(`button`,{role:`option`,"aria-selected":e.id===F,className:`tabs__dropdown-item ${e.id===F?`tabs__dropdown-item--active`:``}`,onClick:()=>{de(e.id),A(!1)},children:e.label},e.id))})]}),p&&(0,x.jsx)(`div`,{id:`tabpanel-${F}`,role:`tabpanel`,"aria-labelledby":`tab-${F}`,className:`tabs__panel ${o}`.trim(),tabIndex:0,onTouchStart:_e,onTouchMove:ve,onTouchEnd:z,children:xe})]}):(0,x.jsxs)(`div`,{className:`tabs ${y===`menu`?`tabs--overflow-menu`:``} ${d===`segmented`?`tabs--variant-segmented`:d===`compact`?`tabs--variant-compact`:``} ${De&&f?`tabs--fill-container`:``} ${a}`.replace(/\s+/g,` `).trim(),children:[(0,x.jsxs)(`div`,{className:`tabs__container`,children:[c&&C&&(0,x.jsx)(`button`,{className:`tabs__scroll-arrow tabs__scroll-arrow--left`,onClick:()=>{L(`left`)},"aria-label":g,tabIndex:-1,children:(0,x.jsx)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 20 20`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M12 15L7 10L12 5`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})}),(0,x.jsx)(`div`,{ref:M,className:`tabs__list`,role:`tablist`,"aria-label":s,onScroll:pe,onKeyDown:t=>{let n=e.findIndex(e=>e.id===F);n<0||I(t,n)},onTouchStart:y===`dropdown`?_e:void 0,onTouchMove:y===`dropdown`?ve:void 0,onTouchEnd:y===`dropdown`?z:void 0,children:Te()}),c&&T&&(0,x.jsx)(`button`,{className:`tabs__scroll-arrow tabs__scroll-arrow--right`,onClick:()=>{L(`right`)},"aria-label":v,tabIndex:-1,children:(0,x.jsx)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 20 20`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{d:`M8 5L13 10L8 15`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})}),Ee()]}),p&&(0,x.jsx)(`div`,{id:`tabpanel-${F}`,role:`tabpanel`,"aria-labelledby":`tab-${F}`,className:`tabs__panel ${o}`.trim(),tabIndex:0,children:xe})]})}var ne=[`a`,`button`,`input`,`select`,`textarea`,`summary`,`label`,`[contenteditable='true']`,`[role='button']`,`[role='link']`,`[tabindex]:not([tabindex='-1'])`].join(`,`);function M(e,t){if(!(e instanceof Element)||!(t instanceof Element))return!1;let n=e.closest(ne);return n!==null&&n!==t&&t.contains(n)}var re={widths:{},visibility:{}};function ie(e,t){let n=e.map((e,t)=>({row:e,i:t}));return n.sort((e,n)=>{let r=t(e.row,n.row);return r===0?e.i-n.i:r}),n.map(({row:e})=>e)}function N(e,t){let{sortComparator:n}=e;if(n)return(e,r)=>n(e,r,t);let r=e.sortValueAccessor;return r?(e,n)=>{let i=r(e),a=r(n);if(i==null&&a==null)return 0;if(i==null)return 1;if(a==null)return-1;let o;return o=typeof i==`number`&&typeof a==`number`?i-a:String(i).localeCompare(String(a)),t===`asc`?o:-o}:()=>0}function ae(e,t,n){if(e.sortable)return e.id===t?n===`asc`?`ascending`:`descending`:`none`}function oe({direction:e}){return e===`none`?(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--none`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`,opacity:`0.35`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`,opacity:`0.35`})]})}):e===`asc`?(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--asc`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`,opacity:`0.25`})]})}):(0,x.jsx)(`span`,{className:`data-table__sort-icon data-table__sort-icon--desc`,"aria-hidden":`true`,children:(0,x.jsxs)(`svg`,{width:`10`,height:`14`,viewBox:`0 0 10 14`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:[(0,x.jsx)(`path`,{d:`M5 1L1 5H9L5 1Z`,fill:`currentColor`,opacity:`0.25`}),(0,x.jsx)(`path`,{d:`M5 13L9 9H1L5 13Z`,fill:`currentColor`})]})})}function se({columns:e,rows:t,getRowKey:n,emptyState:r,loadingState:i,loading:a=!1,columnState:o,onColumnStateChange:s,className:c=``,ariaLabel:l=`Data table`,testId:u,onRowClick:d,isRowClickable:f,rowClassName:p,sortColumnId:m,sortDirection:h,onSortChange:g}){let[v,y]=(0,_.useState)(()=>o??re);(0,_.useEffect)(()=>{o&&y(o)},[o]);let b=m!==void 0||h!==void 0,[S,C]=(0,_.useState)(void 0),[w,T]=(0,_.useState)(void 0),E=b?m:S,D=b?h:w,O=(0,_.useCallback)(e=>{if(!e.sortable)return;let t,n;E===e.id?D===(e.defaultSortDirection??`asc`)?(t=e.id,n=D===`asc`?`desc`:`asc`):(t=void 0,n=void 0):(t=e.id,n=e.defaultSortDirection??`asc`),b||(C(t),T(n)),g?.(t,n)},[E,D,b,g]),k=(0,_.useMemo)(()=>e.filter(e=>e.alwaysVisible?!0:v.visibility[e.id]!==!1),[e,v.visibility]),A=(0,_.useCallback)(e=>{let t=v.widths[e.id];return typeof t==`number`&&t>0?t:e.initialWidth},[v.widths]),j=(0,_.useRef)(null),ee=(0,_.useCallback)((e,t,n)=>{if(t.noResize)return;e.preventDefault(),e.stopPropagation(),j.current={columnId:t.id,startX:e.clientX,startWidth:n,minWidth:t.minWidth??80};let r=e=>{let t=j.current;if(!t)return;let n=e.clientX-t.startX,r=Math.max(t.minWidth,t.startWidth+n);y(e=>({...e,widths:{...e.widths,[t.columnId]:r}}))},i=()=>{j.current=null,window.removeEventListener(`mousemove`,r),window.removeEventListener(`mouseup`,i)};window.addEventListener(`mousemove`,r),window.addEventListener(`mouseup`,i)},[]),te=(0,_.useRef)(v);(0,_.useEffect)(()=>{te.current!==v&&(te.current=v,s?.(v))},[v,s]);let ne=(0,_.useMemo)(()=>{if(!E||!D)return t;let n=e.find(e=>e.id===E);return n?.sortable?ie(t,N(n,D)):t},[t,e,E,D]),se=[`data-table`,c].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{className:se,"data-testid":u,children:(0,x.jsxs)(`table`,{className:`data-table__table`,"aria-label":l,role:`table`,children:[(0,x.jsx)(`thead`,{className:`data-table__head`,children:(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--head`,children:k.map(e=>{let t=A(e),n=e.sortable&&e.id===E;return(0,x.jsxs)(`th`,{scope:`col`,className:[`data-table__cell`,`data-table__cell--head`,e.sortable&&`data-table__cell--sortable`,n&&`data-table__cell--sort-active`,e.className,e.align&&`data-table__cell--align-${e.align}`].filter(Boolean).join(` `),style:t?{width:`${String(t)}px`}:void 0,"aria-sort":ae(e,E,D),onClick:e.sortable?()=>{O(e)}:void 0,onKeyDown:e.sortable?t=>{(t.key===`Enter`||t.key===` `)&&(t.preventDefault(),O(e))}:void 0,tabIndex:e.sortable?0:void 0,children:[(0,x.jsx)(`span`,{className:`data-table__head-label`,children:e.header}),e.sortable&&(0,x.jsx)(oe,{direction:n&&D?D:`none`}),!e.noResize&&(0,x.jsx)(`div`,{role:`separator`,"aria-orientation":`vertical`,className:`data-table__resize-handle`,onMouseDown:n=>{ee(n,e,t??e.minWidth??120)}})]},e.id)})})}),(0,x.jsx)(`tbody`,{className:`data-table__body`,children:a?(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--state`,children:(0,x.jsx)(`td`,{colSpan:k.length||1,className:`data-table__state-cell`,children:i??(0,x.jsx)(`div`,{className:`data-table__loading`})})}):ne.length===0?(0,x.jsx)(`tr`,{className:`data-table__row data-table__row--state`,children:(0,x.jsx)(`td`,{colSpan:k.length||1,className:`data-table__state-cell`,children:r})}):ne.map((e,t)=>{let r=n(e,t),i=!!(d&&(f?.(e,t)??!0)),a=p?.(e,t);return(0,x.jsx)(`tr`,{className:[`data-table__row`,i&&`data-table__row--clickable`,a].filter(Boolean).join(` `),onClick:i?t=>{M(t.target,t.currentTarget)||d?.(e)}:void 0,onKeyDown:i?t=>{(t.key===`Enter`||t.key===` `)&&(M(t.target,t.currentTarget)||(t.preventDefault(),d?.(e)))}:void 0,tabIndex:i?0:void 0,role:i?`button`:void 0,children:k.map(n=>(0,x.jsx)(`td`,{className:[`data-table__cell`,n.className,n.align&&`data-table__cell--align-${n.align}`].filter(Boolean).join(` `),children:n.render(e,t)},n.id))},r)})})]})})}var ce=(0,_.memo)(se),P=(0,_.forwardRef)(function({tone:e=`secondary`,busy:t=!1,className:n=``,children:r,"aria-label":i,...a},o){return(0,x.jsx)(w,{...a,ref:o,inline:!0,variant:e,loading:t,"aria-busy":t||a[`aria-busy`],ariaLabel:i,className:`ds-button ds-button-${e} ${n}`.trim(),children:(0,x.jsx)(`span`,{children:r})})}),F=(0,_.forwardRef)(function({label:e,icon:t,busy:n=!1,className:r=``,...i},a){return(0,x.jsx)(w,{...i,ref:a,inline:!0,iconOnly:!0,icon:(0,x.jsx)(C,{name:t}),ariaLabel:e,title:i.title??e,variant:`secondary`,loading:n,"aria-busy":n||i[`aria-busy`],className:`ds-icon-button ${r}`.trim()})}),le={unloaded:`terminated`,loading:`preparing`,ready:`running`,draining:`stopping`,unloading:`stopping`,failed:`error`};function ue(e){return(0,x.jsx)(`span`,{className:`ds-status-wrap`,"data-state":e.state,"aria-label":e.label,children:(0,x.jsx)(k,{state:le[e.state],label:e.children,pulse:!1,className:`ds-status`})})}function de(e){let t=e.value===void 0||!Number.isFinite(e.value)?null:Math.max(0,Math.min(100,e.value));return(0,x.jsxs)(`div`,{className:`ds-progress`,children:[(0,x.jsx)(A,{value:t,ariaLabel:e.label,animated:!1}),e.detail?(0,x.jsx)(`small`,{children:e.detail}):null]})}function I(e){return(0,x.jsx)(`section`,{className:`ds-empty`,"data-testid":e.testId,ref:e=>{let t=e?.querySelector(`.empty-state__title`);t?.setAttribute(`role`,`heading`),t?.setAttribute(`aria-level`,`2`)},children:(0,x.jsx)(ee,{title:e.title,description:e.body,illustration:(0,x.jsx)(C,{name:`models`}),children:e.action})})}function fe(e){let t=(0,_.useId)();if(e.tabs.length===0)return(0,x.jsx)(`section`,{className:`ds-tabs`});let n=e.tabs.map(e=>({id:`${t}-${e.id}`,label:e.label,content:e.panel})),r=e.tabs.find(t=>t.id===e.active)??e.tabs[0];return(0,x.jsx)(`section`,{onKeyDownCapture:e=>{e.nativeEvent.isComposing&&e.stopPropagation()},children:(0,x.jsx)(te,{className:`ds-tabs`,tabs:n,activeTab:`${t}-${r.id}`,onTabChange:n=>{let r=e.tabs.find(e=>`${t}-${e.id}`===n);r&&e.onChange(r.id)},ariaLabel:e.label??`Sections`,variant:`segmented`,fillContainer:!0,overflowMode:`menu`,showOverflowControls:!1,showGroupLabels:!1})})}var pe=`ds-row-primary`,L=`a[href], button, input, select, textarea, summary, label, [contenteditable="true"], [role="button"], [role="link"], [tabindex]:not([tabindex="-1"])`;function R(e){if(e.defaultPrevented||e.button!==0||!(e.target instanceof Element))return;let t=e.target.closest(`tbody > tr`);if(!t||!e.currentTarget.contains(t)||t.classList.contains(`data-table__row--state`))return;let n=t.querySelector(`.${pe}`);if(!n||n.matches(`:disabled, [aria-disabled="true"]`))return;let r=e.target.closest(L);if(r&&t.contains(r))return;let i=window.getSelection();(!i||i.isCollapsed||i.toString().trim()===``||!t.contains(i.anchorNode)&&!t.contains(i.focusNode))&&(n.focus({preventScroll:!0}),n.click())}function me({activateRowPrimary:e=!1,...t}){let n=(0,x.jsx)(ce,{...t,className:`ds-common-table ${t.className??``}`.trim()});return e?(0,x.jsx)(`div`,{className:`ds-row-activation`,onClick:R,children:n}):n}function he({className:e,style:t,size:n}){let r=n??`1em`;return(0,x.jsx)(`svg`,{xmlns:`http://www.w3.org/2000/svg`,width:r,height:r,fill:`none`,viewBox:`0 0 24 24`,strokeWidth:1.5,stroke:`currentColor`,className:e,style:t,"aria-hidden":`true`,children:(0,x.jsx)(`path`,{strokeLinecap:`round`,strokeLinejoin:`round`,d:`M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z`})})}function ge({tone:e=`danger`,icon:t,title:n,message:r,primaryAction:i,secondaryAction:a,className:o=``,showIcon:s=!0}){let c=(0,_.useCallback)(()=>{i?.onClick()},[i]),l=(0,_.useCallback)(()=>{a?.onClick()},[a]),u=[`error-state`,`error-state--${e}`,o].filter(Boolean).join(` `);return(0,x.jsxs)(`div`,{className:u,role:`alert`,"aria-live":`polite`,children:[s&&(0,x.jsx)(`div`,{className:`error-state__icon`,"aria-hidden":`true`,children:t??(0,x.jsx)(he,{size:48})}),(0,x.jsx)(`h2`,{className:`error-state__title`,children:n}),(0,x.jsx)(`p`,{className:`error-state__message`,children:r}),(i||a)&&(0,x.jsxs)(`div`,{className:`error-state__actions`,children:[i&&(0,x.jsx)(w,{variant:`primary`,onClick:c,className:`error-state__action-btn error-state__action-btn--primary`,children:i.label}),a&&(0,x.jsx)(w,{variant:`secondary`,onClick:l,className:`error-state__action-btn error-state__action-btn--secondary`,children:a.label})]})]})}function _e({width:e=`100%`,height:t=`20px`,variant:n=`rect`,className:r=``,testId:i,loadingLabel:a=`Loading`,decorative:o=!1}){let s=[`skeleton`,`skeleton--${n}`,r].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{className:s,style:{width:e,height:t},...o?{"aria-hidden":!0}:{role:`status`,"aria-busy":!0,"aria-label":a},"data-testid":i,children:(0,x.jsx)(`span`,{className:`skeleton__shimmer`})})}var ve={error:`danger`,warning:`warning`,info:`accent`};function z(e){let t=e.tone??`error`;return(0,x.jsxs)(`section`,{className:`ds-banner`,"data-tone":t,role:t===`info`?`status`:`alert`,"data-testid":e.testId,ref:e=>{let t=e?.querySelector(`.error-state`);t?.removeAttribute(`role`),t?.removeAttribute(`aria-live`),e?.querySelector(`.error-state__title`)?.setAttribute(`role`,`none`)},children:[(0,x.jsx)(ge,{tone:ve[t],title:e.title,message:e.body,showIcon:!1}),e.action]})}function ye(e){return(0,x.jsxs)(`div`,{className:`ds-loading`,role:`status`,children:[(0,x.jsx)(`p`,{className:`ds-loading__label`,children:e.label}),(0,x.jsx)(`div`,{className:`ds-loading__shapes`,"aria-hidden":`true`,children:Array.from({length:e.rows??3},(e,t)=>(0,x.jsx)(_e,{decorative:!0,width:`100%`,height:`20px`},t))})]})}var be=(0,_.createContext)(!1),xe=m();function Se({value:e,onChange:t,onBlur:n,options:r,label:i,placeholder:a=`Select...`,disabled:o=!1,size:s=`default`,fullWidth:c=!1,className:l=``,searchable:u=!1,searchPlaceholder:d,"aria-label":f,"aria-describedby":p,invalid:m=!1,noOptionsLabel:h=`No options`}){let[g,v]=(0,_.useState)(!1),[y,b]=(0,_.useState)(-1),[S,C]=(0,_.useState)(``),[w,T]=(0,_.useState)({top:0,left:0,width:0}),E=(0,_.useRef)(null),D=(0,_.useRef)(null),O=(0,_.useRef)(null),k=(0,_.useRef)(null),A=(0,_.useRef)([]),j=(0,_.useId)(),ee=(0,_.useId)(),te=(0,_.useId)(),ne=i!=null&&i!==!1,M=r.find(t=>t.value===e),re=(0,_.useMemo)(()=>{let e=S.trim().toLowerCase();return!u||e.length===0?r:r.filter(t=>[t.label,t.value,t.description??``].some(t=>t.toLowerCase().includes(e)))},[r,S,u]),ie=(0,_.useCallback)((e,t)=>{for(let n=e;n>=0&&n<re.length;n+=t){let e=re[n];if(e&&!e.disabled)return n}return-1},[re]),N=e=>`${ee}-option-${String(e)}`,ae=y>=0?N(y):void 0,oe=(0,_.useCallback)(()=>{if(!D.current)return;let e=D.current.getBoundingClientRect();T({top:e.bottom+4,left:e.left,width:e.width})},[]),se=(0,_.useCallback)(()=>{if(o)return;oe(),v(!0);let t=re.findIndex(t=>t.value===e&&!t.disabled);b(t===-1?ie(0,1):t)},[o,re,e,oe,ie]),ce=(0,_.useCallback)(()=>{v(!1),b(-1),C(``),D.current?.focus()},[]),P=(0,_.useCallback)((e,n)=>{n&&(n.stopPropagation(),n.preventDefault()),!e.disabled&&(t(e.value),ce())},[t,ce]);(0,_.useEffect)(()=>{let e=e=>{if(!g)return;let t=e.target,n=E.current&&!E.current.contains(t),r=O.current&&!O.current.contains(t);n&&r&&ce()},t=e=>{e.key===`Escape`&&g&&ce()};return g&&(document.addEventListener(`mousedown`,e),document.addEventListener(`keydown`,t)),()=>{document.removeEventListener(`mousedown`,e),document.removeEventListener(`keydown`,t)}},[g,ce]),(0,_.useEffect)(()=>{if(!g)return;let e=()=>{oe()};return window.addEventListener(`resize`,e),window.addEventListener(`scroll`,e,!0),()=>{window.removeEventListener(`resize`,e),window.removeEventListener(`scroll`,e,!0)}},[g,oe]),(0,_.useEffect)(()=>{g&&u&&k.current?.focus()},[g,u]),(0,_.useEffect)(()=>{if(!g||!u)return;A.current=[];let t=re.findIndex(t=>t.value===e&&!t.disabled);b(t>=0?t:ie(0,1))},[g,S,u,e,re,ie]);let F=(0,_.useCallback)(e=>{if(!o)switch(e.key){case`Enter`:case` `:if(e.preventDefault(),g){let e=re[y];e&&P(e)}else se();break;case`Escape`:e.preventDefault(),g&&ce();break;case`ArrowDown`:if(e.preventDefault(),!g)se();else{let e=ie(y+1,1);e>=0&&b(e)}break;case`ArrowUp`:if(e.preventDefault(),g){let e=ie(y-1,-1);e>=0&&b(e)}break;case`Tab`:g&&ce()}},[o,g,y,re,ie,se,ce,P]),le=(0,_.useCallback)(e=>{switch(e.key){case`Enter`:{e.preventDefault();let t=re[y];t&&P(t);break}case`Escape`:e.preventDefault(),e.stopPropagation(),ce();break;case`ArrowDown`:e.preventDefault(),b(e=>{let t=ie(e+1,1);return t>=0?t:e});break;case`ArrowUp`:e.preventDefault(),b(e=>{let t=ie(e-1,-1);return t>=0?t:e});break;case`Tab`:ce()}},[ce,y,P,re,ie]);(0,_.useEffect)(()=>{let e=A.current[y];g&&y>=0&&e&&e.scrollIntoView({block:`nearest`,behavior:`smooth`})},[g,y]);let ue=s==="default"?``:`select--${s}`,de=c?`select--full-width`:``,I=ne?`select--labelled`:``,fe=ne?te:void 0,pe=d??f;return(0,x.jsxs)(`div`,{ref:E,className:`select ${ue} ${de} ${I} ${g?`select--open`:``} ${o?`select--disabled`:``} ${l}`,children:[ne&&(0,x.jsx)(`span`,{className:`select__label`,id:te,children:i}),(0,x.jsxs)(`button`,{ref:D,type:`button`,className:`select__trigger`,onClick:()=>{g?ce():se()},onKeyDown:F,onBlur:n,disabled:o,"aria-haspopup":`listbox`,"aria-expanded":g,"aria-controls":g?j:void 0,"aria-activedescendant":g?ae:void 0,"aria-label":f,"aria-labelledby":fe,"aria-describedby":p,"aria-invalid":m||void 0,children:[M?(0,x.jsxs)(`span`,{className:`select__value`,children:[M.icon&&(0,x.jsx)(`span`,{className:`select__value-icon`,children:M.icon}),(0,x.jsx)(`span`,{className:`select__value-label`,children:M.label})]}):(0,x.jsx)(`span`,{className:`select__placeholder`,children:a}),(0,x.jsx)(`span`,{className:`select__chevron ${g?`select__chevron--open`:``}`,children:(0,x.jsx)(`svg`,{width:`10`,height:`6`,viewBox:`0 0 10 6`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:(0,x.jsx)(`path`,{d:`M1 1L5 5L9 1`,stroke:`currentColor`,strokeWidth:`1.5`,strokeLinecap:`round`,strokeLinejoin:`round`})})})]}),g&&(0,xe.createPortal)((0,x.jsxs)(`div`,{ref:O,className:`select__dropdown select__dropdown--portal`,style:{position:`fixed`,top:`${String(w.top)}px`,left:`${String(w.left)}px`,width:`${String(w.width)}px`},children:[u&&(0,x.jsx)(`div`,{className:`select__search`,children:(0,x.jsx)(`input`,{ref:k,className:`select__search-input`,type:`search`,value:S,onChange:e=>C(e.currentTarget.value),onKeyDown:le,placeholder:d,"aria-label":pe,"aria-labelledby":pe?void 0:fe,"aria-controls":j,"aria-activedescendant":ae,autoComplete:`off`,spellCheck:!1})}),(0,x.jsx)(`div`,{className:`select__options`,role:`listbox`,id:j,"aria-label":f,"aria-labelledby":fe,children:re.length===0?(0,x.jsx)(`div`,{className:`select__empty`,children:h}):re.map((t,n)=>(0,x.jsxs)(`div`,{id:N(n),ref:e=>{A.current[n]=e},className:`select__option ${t.value===e?`select__option--selected`:``} ${n===y?`select__option--focused`:``} ${t.disabled?`select__option--disabled`:``}`,role:`option`,"aria-selected":t.value===e,"aria-disabled":t.disabled||void 0,"aria-label":t.description?t.label+`. `+t.description:void 0,onClick:e=>{P(t,e)},onMouseEnter:()=>{t.disabled||b(n)},children:[t.icon&&(0,x.jsx)(`span`,{className:`select__option-icon`,children:t.icon}),(0,x.jsxs)(`div`,{className:`select__option-content`,children:[(0,x.jsx)(`span`,{className:`select__option-label`,children:t.label}),t.description&&(0,x.jsx)(`span`,{className:`select__option-description`,children:t.description})]}),t.value===e&&(0,x.jsx)(`span`,{className:`select__option-check`,children:(0,x.jsx)(`svg`,{width:`14`,height:`10`,viewBox:`0 0 14 10`,fill:`none`,xmlns:`http://www.w3.org/2000/svg`,children:(0,x.jsx)(`path`,{d:`M1 5L5 9L13 1`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`})})})]},t.value))})]}),document.body)]})}var Ce=[{key:`models.library.subtitle`,en:`Inspect local checkpoints. Selecting never loads a model.`,ko:`로컬 체크포인트를 확인하세요. 선택만으로 모델을 로드하지 않습니다.`,test_id:`models-library-subtitle`},{key:`models.library.search`,en:`Search local models`,ko:`로컬 모델 검색`,test_id:`models-library-search`},{key:`models.library.source`,en:`Source`,ko:`소스`,test_id:`models-library-source`},{key:`models.library.task`,en:`Task`,ko:`작업`,test_id:`models-library-task`},{key:`models.library.status`,en:`Lifecycle`,ko:`수명주기`,test_id:`models-library-status`},{key:`models.library.sort`,en:`Sort`,ko:`정렬`,test_id:`models-library-sort`},{key:`models.library.name`,en:`Model`,ko:`모델`,test_id:`models-library-name`},{key:`models.library.all`,en:`All`,ko:`전체`,test_id:`models-library-all`},{key:`models.library.add`,en:`Add Model`,ko:`모델 추가`,test_id:`models-library-add`},{key:`models.library.rescan`,en:`Rescan local roots`,ko:`로컬 경로 다시 검색`,test_id:`models-library-rescan`},{key:`models.library.refresh`,en:`Refresh server state`,ko:`서버 상태 새로고침`,test_id:`models-library-refresh`},{key:`models.library.roots`,en:`Configured roots`,ko:`설정된 경로`,test_id:`models-library-roots`},{key:`models.library.roots_help`,en:`To change roots, restart the server with your local directory. This command is an example, not an executed action.`,ko:`경로를 변경하려면 로컬 디렉터리를 지정하여 서버를 다시 시작하세요. 아래 명령은 예시이며 자동 실행되지 않습니다.`,test_id:`models-library-roots-help`},{key:`models.library.single`,en:`Single-model mode: lifecycle and cache changes are read-only. Restart without -m to manage models.`,ko:`단일 모델 모드: 수명주기 및 캐시는 읽기 전용입니다. 모델을 관리하려면 -m 없이 다시 시작하세요.`,test_id:`models-library-single`},{key:`models.library.unknown`,en:`Unknown`,ko:`알 수 없음`,test_id:`models-library-unknown`},{key:`models.library.yes`,en:`Yes`,ko:`예`,test_id:`models-library-yes`},{key:`models.library.no`,en:`No`,ko:`아니요`,test_id:`models-library-no`},{key:`models.library.inspect`,en:`Inspect {name}`,ko:`{name} 자세히 보기`,test_id:`models-library-inspect`},{key:`models.library.selected`,en:`Selected`,ko:`선택됨`,test_id:`models-library-selected`},{key:`models.library.details`,en:`Model details`,ko:`모델 상세`,test_id:`models-library-details`},{key:`models.library.support`,en:`Support / completeness`,ko:`지원 / 완전성`,test_id:`models-library-support`},{key:`models.library.supported`,en:`Supported architecture`,ko:`지원되는 아키텍처`,test_id:`models-library-supported`},{key:`models.library.unsupported`,en:`Unsupported architecture`,ko:`지원되지 않는 아키텍처`,test_id:`models-library-unsupported`},{key:`models.library.complete`,en:`Complete files`,ko:`파일 완전함`,test_id:`models-library-complete`},{key:`models.library.incomplete`,en:`Incomplete files`,ko:`불완전한 파일`,test_id:`models-library-incomplete`},{key:`models.library.architecture`,en:`Architecture`,ko:`아키텍처`,test_id:`models-library-architecture`},{key:`models.library.backend`,en:`Runnable on this backend`,ko:`현재 백엔드 실행 가능`,test_id:`models-library-backend`},{key:`models.library.tested`,en:`Checkpoint validated`,ko:`검증된 체크포인트`,test_id:`models-library-tested`},{key:`models.library.tested_help`,en:`Family support does not prove this checkpoint was tested.`,ko:`계열 지원은 해당 체크포인트의 검증을 의미하지 않습니다.`,test_id:`models-library-tested-help`},{key:`models.library.disk`,en:`Disk size`,ko:`디스크 크기`,test_id:`models-library-disk`},{key:`models.library.memory`,en:`Estimated memory (not measured)`,ko:`예상 메모리 (측정값 아님)`,test_id:`models-library-memory`},{key:`models.library.context`,en:`Effective context (tokens)`,ko:`실제 컨텍스트 (토큰)`,test_id:`models-library-context`},{key:`models.library.profile`,en:`Next-load profile`,ko:`다음 로드 프로필`,test_id:`models-library-profile`},{key:`models.library.defaults`,en:`Server defaults; see Settings for scope and overrides.`,ko:`서버 기본값입니다. 범위와 재정의는 설정에서 확인하세요.`,test_id:`models-library-defaults`},{key:`models.library.error`,en:`Action could not complete`,ko:`작업을 완료하지 못함`,test_id:`models-library-error`},{key:`models.library.last_error`,en:`Last server error`,ko:`최근 서버 오류`,test_id:`models-library-last-error`},{key:`models.library.chat`,en:`Use in Chat`,ko:`채팅에서 사용`,test_id:`models-library-chat`},{key:`models.library.chat_reason`,en:`Chat requires a ready provider with an available chat capability. Other tasks remain available through their API.`,ko:`채팅에는 준비된 제공자와 사용 가능한 채팅 기능이 필요합니다. 다른 작업은 API로 사용하세요.`,test_id:`models-library-chat-reason`},{key:`models.library.api`,en:`API task documentation`,ko:`API 작업 문서`,test_id:`models-library-api`},{key:`models.library.delete`,en:`Delete cached files`,ko:`캐시 파일 삭제`,test_id:`models-library-delete`},{key:`models.library.delete_body`,en:`Permanently delete the managed cache entry {name} ({source}). This removes files from disk, unlike Unload. To confirm, type the model ID shown below, not the model name.`,ko:`관리 캐시 항목 {name} ({source})을 영구 삭제합니다. 언로드와 달리 디스크 파일을 제거합니다. 확인하려면 모델 이름이 아니라 아래에 표시된 모델 ID를 입력하세요.`,test_id:`models-library-delete-body`},{key:`models.library.confirm_name`,en:`Opaque model ID shown above (starts with mdl_)`,ko:`위에 표시된 모델 ID (mdl_로 시작)`,test_id:`models-library-confirm-name`},{key:`models.library.unload_body`,en:`Unload {name} ({source}): {count} active requests must drain before the worker exits. Files remain on disk. A timeout is a failure, not a completed unload.`,ko:`{name} ({source}) 언로드: 활성 요청 {count}개를 배출한 뒤 작업자가 종료됩니다. 디스크 파일은 유지됩니다. 시간 초과는 완료가 아닌 실패입니다.`,test_id:`models-library-unload-body`},{key:`models.library.confirm`,en:`Confirm`,ko:`확인`,test_id:`models-library-confirm`},{key:`models.library.cancel`,en:`Cancel`,ko:`취소`,test_id:`models-library-cancel`},{key:`models.library.repo`,en:`Public HuggingFace repo ID`,ko:`공개 HuggingFace 저장소 ID`,test_id:`models-library-repo`},{key:`models.library.revision`,en:`Revision (optional)`,ko:`리비전 (선택)`,test_id:`models-library-revision`},{key:`models.library.network`,en:`Only this explicit action contacts HuggingFace. Files go to the server-managed cache below. Size is unknown until the server resolves file metadata. Search never contacts external services.`,ko:`이 작업을 명시적으로 실행할 때만 HuggingFace에 연결합니다. 파일은 아래 서버 관리 캐시에 저장됩니다. 크기는 서버가 파일 메타데이터를 확인할 때까지 알 수 없습니다. 검색은 외부 서비스에 연결하지 않습니다.`,test_id:`models-library-network`},{key:`models.library.public`,en:`This is a public, ungated repository`,ko:`공개된 비게이트 저장소입니다`,test_id:`models-library-public`},{key:`models.library.private`,en:`Private or gated downloads are unavailable in the WebUI. Use an authenticated HuggingFace CLI outside the server, then rescan a configured local root. Do not paste credentials here.`,ko:`비공개 또는 게이트 다운로드는 WebUI에서 지원하지 않습니다. 서버 외부에서 인증된 HuggingFace CLI를 사용한 뒤 설정된 로컬 경로를 다시 검색하세요. 여기에 인증 정보를 입력하지 마세요.`,test_id:`models-library-private`},{key:`models.library.repo_invalid`,en:`Enter owner/repository, not a URL or a local path.`,ko:`URL이나 로컬 경로가 아닌 소유자/저장소를 입력하세요.`,test_id:`models-library-repo-invalid`},{key:`models.library.operations`,en:`Library operations`,ko:`라이브러리 작업`,test_id:`models-library-operations`},{key:`models.library.pending`,en:`Request outcome is being reconciled. Do not resubmit; refresh to observe the server operation.`,ko:`요청 결과를 확인하고 있습니다. 재제출하지 말고 새로고침하여 서버 작업을 확인하세요.`,test_id:`models-library-pending`},{key:`models.library.retry_download`,en:`Retry download`,ko:`다운로드 재시도`,test_id:`models-library-retry-download`},{key:`models.library.cancel_download`,en:`Cancel download`,ko:`다운로드 취소`,test_id:`models-library-cancel-download`},{key:`models.library.cancel_body`,en:`Cancel the download of {name}? Unpublished partial files are not a usable model.`,ko:`{name} 다운로드를 취소할까요? 공개되지 않은 부분 파일은 사용할 수 있는 모델이 아닙니다.`,test_id:`models-library-cancel-body`},{key:`models.library.progress`,en:`Download progress`,ko:`다운로드 진행률`,test_id:`models-library-progress`},{key:`models.library.worker`,en:`Worker exit observed`,ko:`작업자 종료 확인`,test_id:`models-library-worker`},{key:`models.library.stale`,en:`State changed or is not current. Refresh, inspect the latest revision, and explicitly confirm again.`,ko:`상태가 변경되었거나 최신이 아닙니다. 새로고침하고 최신 리비전을 확인한 뒤 다시 명시적으로 확인하세요.`,test_id:`models-library-stale`},{key:`models.library.capacity`,en:`Capacity / conflicting operation`,ko:`용량 / 작업 충돌`,test_id:`models-library-capacity`},{key:`models.library.capacity_body`,en:`Do not repeatedly load. Review active models below. If capacity is full, explicitly choose an idle ready model to unload before this load. Busy models are never offered as eviction targets. Eviction is not rolled back if the new load fails.`,ko:`반복해서 로드하지 마세요. 아래 활성 모델을 확인하세요. 용량이 가득 차면 이번 로드 전에 언로드할 유휴 준비 모델을 명시적으로 선택하세요. 사용 중인 모델은 교체 대상으로 제공되지 않습니다. 새 로드 실패 시 교체는 되돌려지지 않습니다.`,test_id:`models-library-capacity-body`},{key:`models.library.eviction`,en:`Explicit eviction target`,ko:`명시적 교체 대상`,test_id:`models-library-eviction`},{key:`models.library.choose`,en:`Choose a model`,ko:`모델 선택`,test_id:`models-library-choose`},{key:`models.library.evict_load`,en:`Unload target and load`,ko:`대상 언로드 후 로드`,test_id:`models-library-evict-load`},{key:`models.library.active`,en:`Active requests`,ko:`활성 요청`,test_id:`models-library-active`},{key:`models.library.page`,en:`Page {page} of {pages} · {count} models`,ko:`{pages}페이지 중 {page} · 모델 {count}개`,test_id:`models-library-page`},{key:`models.library.previous`,en:`Previous page`,ko:`이전 페이지`,test_id:`models-library-previous`},{key:`models.library.next`,en:`Next page`,ko:`다음 페이지`,test_id:`models-library-next`},{key:`models.library.filtered`,en:`No matching local models`,ko:`일치하는 로컬 모델 없음`,test_id:`models-library-filtered`},{key:`models.library.filtered_body`,en:`Change the local search or filters.`,ko:`로컬 검색어나 필터를 변경하세요.`,test_id:`models-library-filtered-body`},{key:`models.library.waiting`,en:`Waiting for an authoritative catalog snapshot`,ko:`서버의 카탈로그 스냅샷을 기다리는 중`,test_id:`models-library-waiting`},{key:`models.library.sort_name`,en:`Name (A–Z)`,ko:`이름 (오름차순)`,test_id:`models-library-sort-name`},{key:`models.library.sort_status`,en:`Lifecycle`,ko:`수명주기`,test_id:`models-library-sort-status`},{key:`models.library.sort_source`,en:`Source`,ko:`소스`,test_id:`models-library-sort-source`},{key:`models.library.sort_task`,en:`Task`,ko:`작업`,test_id:`models-library-sort-task`},{key:`models.library.unavailable`,en:`Not available for this server or model`,ko:`현재 서버 또는 모델에서 사용할 수 없음`,test_id:`models-library-unavailable`},{key:`models.library.permission`,en:`Root could not be read. Check server directory permissions, then rescan.`,ko:`경로를 읽지 못했습니다. 서버 디렉터리 권한을 확인한 뒤 다시 검색하세요.`,test_id:`models-library-permission`},{key:`models.library.copy`,en:`Copy command`,ko:`명령 복사`,test_id:`models-library-copy`},{key:`models.library.copied`,en:`Copied`,ko:`복사됨`,test_id:`models-library-copied`},{key:`models.library.copy_failed`,en:`Select and copy the command below.`,ko:`아래 명령을 선택하여 복사하세요.`,test_id:`models-library-copy-failed`},{key:`models.library.accepted`,en:`Accepted; waiting for observed server state.`,ko:`접수됨. 서버에서 관측된 상태를 기다리는 중입니다.`,test_id:`models-library-accepted`},{key:`models.library.operation_state`,en:`Operation state`,ko:`작업 상태`,test_id:`models-library-operation-state`}],we=[{key:`chat.intro`,en:`Memory-only by default. Changing the model affects the next turn, never the running request.`,ko:`기본적으로 메모리에만 보관합니다. 모델을 바꾸면 다음 턴에만 적용되며 실행 중인 요청은 바뀌지 않습니다.`,test_id:`chat-intro`},{key:`chat.new_conversation`,en:`New conversation`,ko:`새 대화`,test_id:`chat-new-conversation`},{key:`chat.conversation.default_title`,en:`New conversation`,ko:`새 대화`,test_id:`chat-conversation-default-title`},{key:`chat.list.label`,en:`Conversations`,ko:`대화 목록`,test_id:`chat-list-label`},{key:`chat.list.open`,en:`Show conversations`,ko:`대화 목록 보기`,test_id:`chat-list-open`},{key:`chat.list.empty.title`,en:`No conversations yet`,ko:`아직 대화가 없습니다`,test_id:`chat-list-empty-title`},{key:`chat.list.empty.body`,en:`Send a message to start one, or choose New conversation.`,ko:`메시지를 보내 대화를 시작하거나 새 대화를 선택하세요.`,test_id:`chat-list-empty-body`},{key:`chat.list.no_messages`,en:`No messages yet`,ko:`아직 메시지 없음`,test_id:`chat-list-no-messages`},{key:`chat.list.limit`,en:`Conversation limit reached (50). Delete or export older conversations first.`,ko:`대화 개수 한도(50개)에 도달했습니다. 먼저 이전 대화를 삭제하거나 내보내세요.`,test_id:`chat-list-limit`},{key:`chat.list.rename`,en:`Rename {title}`,ko:`{title} 이름 바꾸기`,test_id:`chat-list-rename`},{key:`chat.list.rename_field`,en:`New name for {title}`,ko:`{title}의 새 이름`,test_id:`chat-list-rename-field`},{key:`chat.list.delete`,en:`Delete {title}`,ko:`{title} 삭제`,test_id:`chat-list-delete`},{key:`chat.list.delete.confirm.title`,en:`Delete {title}?`,ko:`{title} 대화를 삭제할까요?`,test_id:`chat-list-delete-confirm-title`},{key:`chat.list.delete.confirm.body`,en:`The conversation and all of its turns are removed. Export your history first if you want to keep a copy.`,ko:`대화와 모든 턴이 삭제됩니다. 사본을 남기려면 먼저 기록을 내보내세요.`,test_id:`chat-list-delete-confirm-body`},{key:`chat.list.delete.confirm`,en:`Delete conversation`,ko:`대화 삭제`,test_id:`chat-list-delete-confirm`},{key:`chat.model.label`,en:`Model for next turn`,ko:`다음 턴에 사용할 모델`,test_id:`chat-model-label`},{key:`chat.model.choose`,en:`Select a model`,ko:`모델 선택`,test_id:`chat-model-choose`},{key:`chat.model.group.ready`,en:`Ready to chat`,ko:`대화 준비됨`,test_id:`chat-model-group-ready`},{key:`chat.model.group.unloaded`,en:`Not loaded`,ko:`로드되지 않음`,test_id:`chat-model-group-unloaded`},{key:`chat.model.group.failed`,en:`Load failed`,ko:`로드 실패`,test_id:`chat-model-group-failed`},{key:`chat.model.group.unavailable`,en:`Not available for chat`,ko:`대화에 사용할 수 없음`,test_id:`chat-model-group-unavailable`},{key:`chat.model.hint.choose`,en:`Choose a model to start.`,ko:`시작하려면 모델을 선택하세요.`,test_id:`chat-model-hint-choose`},{key:`chat.model.hint.load`,en:`Not loaded. Load it to chat; selecting a model never loads it.`,ko:`로드되지 않았습니다. 대화하려면 로드하세요. 모델을 선택하는 것만으로는 로드하지 않습니다.`,test_id:`chat-model-hint-load`},{key:`chat.model.hint.cannot_load`,en:`This model cannot be loaded from here. Open Models for the reason.`,ko:`여기에서는 이 모델을 로드할 수 없습니다. 모델 화면에서 이유를 확인하세요.`,test_id:`chat-model-hint-cannot-load`},{key:`chat.model.hint.wait`,en:`Not ready yet. Send turns on once the model is ready to chat.`,ko:`아직 준비되지 않았습니다. 모델이 대화 준비를 마치면 보내기가 켜집니다.`,test_id:`chat-model-hint-wait`},{key:`chat.model.hint.unavailable`,en:`This model does not offer chat. Embeddings, reranking and other tasks use their documented API, not this composer.`,ko:`이 모델은 대화를 지원하지 않습니다. 임베딩, 재순위화 등 다른 작업은 이 입력창이 아니라 문서화된 API를 사용합니다.`,test_id:`chat-model-hint-unavailable`},{key:`chat.load.confirm.title`,en:`Load {model}?`,ko:`{model} 모델을 로드할까요?`,test_id:`chat-load-confirm-title`},{key:`chat.load.confirm.body`,en:`Selecting a model never loads it. Loading starts a server operation that reserves memory for this model and keeps running if you leave Chat.`,ko:`모델을 선택하는 것만으로는 로드하지 않습니다. 로드는 이 모델에 메모리를 예약하는 서버 작업을 시작하며, 대화 화면을 떠나도 계속 진행됩니다.`,test_id:`chat-load-confirm-body`},{key:`chat.load.requested`,en:`Load requested. Send turns on when the model is ready.`,ko:`로드를 요청했습니다. 모델이 준비되면 보내기가 켜집니다.`,test_id:`chat-load-requested`},{key:`chat.no_model.action`,en:`Open Models`,ko:`모델 화면 열기`,test_id:`chat-no-model-action`},{key:`chat.settings.summary`,en:`Conversation settings`,ko:`대화 설정`,test_id:`chat-settings-summary`},{key:`chat.settings.name`,en:`Conversation name`,ko:`대화 이름`,test_id:`chat-settings-name`},{key:`chat.settings.system_prompt`,en:`System prompt`,ko:`시스템 프롬프트`,test_id:`chat-settings-system-prompt`},{key:`chat.settings.sampling`,en:`Sampling defaults are set in {link} and frozen when sending.`,ko:`샘플링 기본값은 {link}에서 지정하며 전송할 때 고정됩니다.`,test_id:`chat-settings-sampling`},{key:`chat.settings.title`,en:`Chat settings`,ko:`대화 화면 설정`,test_id:`chat-settings-title`},{key:`chat.settings.none`,en:`Start or select a conversation to name it and set its system prompt.`,ko:`대화를 시작하거나 선택하면 이름과 시스템 프롬프트를 지정할 수 있습니다.`,test_id:`chat-settings-none`},{key:`chat.overrides`,en:`Overrides: {keys}`,ko:`재정의: {keys}`,test_id:`chat-overrides`},{key:`chat.composer.label`,en:`Message`,ko:`메시지`,test_id:`chat-composer-label`},{key:`chat.composer.hint`,en:`Enter sends · Shift+Enter inserts a line · Cmd/Ctrl+Enter also sends.`,ko:`Enter로 보내고 Shift+Enter로 줄을 바꿉니다. Cmd/Ctrl+Enter로도 보낼 수 있습니다.`,test_id:`chat-composer-hint`},{key:`chat.composer.count`,en:`{count} / {max}`,ko:`{count} / {max}자`,test_id:`chat-composer-count`},{key:`chat.composer.limit`,en:`Limit reached`,ko:`한도 도달`,test_id:`chat-composer-limit`},{key:`chat.composer.keys`,en:`Keyboard hints`,ko:`키보드 도움말`,test_id:`chat-composer-keys`},{key:`chat.images.attach`,en:`Attach images`,ko:`이미지 첨부`,test_id:`chat-images-attach`},{key:`chat.images.label`,en:`Local images`,ko:`로컬 이미지`,test_id:`chat-images-label`},{key:`chat.images.remove`,en:`Remove {name}`,ko:`{name} 제거`,test_id:`chat-images-remove`},{key:`chat.stop`,en:`Stop`,ko:`중지`,test_id:`chat-stop`},{key:`chat.error.title`,en:`Chat could not continue`,ko:`대화를 계속할 수 없습니다`,test_id:`chat-error-title`},{key:`chat.error.view_closed`,en:`View closed. The request was aborted and was not retried.`,ko:`화면을 닫았습니다. 요청을 중단했으며 다시 시도하지 않았습니다.`,test_id:`chat-error-view-closed`},{key:`chat.error.vision_unsupported`,en:`The selected provider does not confirm vision support. Remove images or select a vision model.`,ko:`선택한 provider가 비전 지원을 확인하지 않습니다. 이미지를 제거하거나 비전 모델을 선택하세요.`,test_id:`chat-error-vision-unsupported`},{key:`chat.error.conversation_limit`,en:`Conversation limit reached. Delete or export older conversations first.`,ko:`대화 개수 한도에 도달했습니다. 먼저 이전 대화를 삭제하거나 내보내세요.`,test_id:`chat-error-conversation-limit`},{key:`chat.error.invalid_parameters`,en:`Invalid next-turn parameters.`,ko:`다음 턴 매개변수가 올바르지 않습니다.`,test_id:`chat-error-invalid-parameters`},{key:`chat.error.turn_limit`,en:`This conversation reached its 100-turn limit. Start a new conversation.`,ko:`이 대화는 100턴 한도에 도달했습니다. 새 대화를 시작하세요.`,test_id:`chat-error-turn-limit`},{key:`chat.error.images_exceed`,en:`Conversation images exceed the current server count, size or decode limits. Remove attachments or start a new conversation.`,ko:`대화 이미지가 현재 서버의 개수, 크기 또는 디코드 한도를 넘습니다. 첨부를 제거하거나 새 대화를 시작하세요.`,test_id:`chat-error-images-exceed`},{key:`chat.error.body_limit`,en:`The complete request exceeds the server or 16 MiB browser JSON body limit. Remove attachments or start a shorter conversation.`,ko:`전체 요청이 서버 또는 16 MiB 브라우저 JSON 본문 한도를 넘습니다. 첨부를 제거하거나 더 짧은 대화를 시작하세요.`,test_id:`chat-error-body-limit`},{key:`chat.error.memory_budget`,en:`Conversation memory budget reached. Export and clear older history first.`,ko:`대화 메모리 예산에 도달했습니다. 먼저 이전 기록을 내보내고 지우세요.`,test_id:`chat-error-memory-budget`},{key:`chat.error.memory_budget_stream`,en:`Conversation memory budget reached. No further output was retained.`,ko:`대화 메모리 예산에 도달했습니다. 이후 출력은 보존하지 않았습니다.`,test_id:`chat-error-memory-budget-stream`},{key:`chat.error.generation_failed`,en:`Generation failed or disconnected. Partial output is preserved. Check context limits, authentication and server availability; nothing was retried.`,ko:`생성에 실패했거나 연결이 끊겼습니다. 부분 출력은 보존됩니다. 컨텍스트 한도, 인증, 서버 상태를 확인하세요. 자동으로 다시 시도하지 않았습니다.`,test_id:`chat-error-generation-failed`},{key:`chat.error.images_invalid`,en:`Images must be bounded local PNG, JPEG or WebP files. No URL, SVG or HTML input is accepted.`,ko:`이미지는 크기 한도 안의 로컬 PNG, JPEG 또는 WebP 파일이어야 합니다. URL, SVG, HTML 입력은 허용하지 않습니다.`,test_id:`chat-error-images-invalid`},{key:`chat.announce.stopped`,en:`Stopped locally; checking server activity.`,ko:`로컬에서 중지했습니다. 서버 활동을 확인하는 중입니다.`,test_id:`chat-announce-stopped`},{key:`chat.announce.aborted_observed`,en:`Request aborted. Server observation refreshed at {time}. See Activity for active requests; this is not a per-request cancellation receipt.`,ko:`요청을 중단했습니다. 서버 관측값을 {time}에 새로 고쳤습니다. 활성 요청은 활동 화면에서 확인하세요. 이 메시지는 요청별 취소 확인이 아닙니다.`,test_id:`chat-announce-aborted-observed`},{key:`chat.announce.unknown_time`,en:`an unknown time`,ko:`알 수 없는 시각`,test_id:`chat-announce-unknown-time`},{key:`chat.announce.aborted_unobserved`,en:`Request aborted. Backend cancellation could not be observed; inspect Activity before unloading.`,ko:`요청을 중단했습니다. 백엔드 취소를 확인할 수 없으니 언로드하기 전에 활동 화면을 확인하세요.`,test_id:`chat-announce-aborted-unobserved`},{key:`chat.announce.generating`,en:`Generating.`,ko:`생성하는 중입니다.`,test_id:`chat-announce-generating`},{key:`chat.announce.complete`,en:`Response finished.`,ko:`응답이 끝났습니다.`,test_id:`chat-announce-complete`},{key:`chat.announce.failed`,en:`Generation failed; partial response preserved.`,ko:`생성에 실패했습니다. 부분 응답은 보존됩니다.`,test_id:`chat-announce-failed`},{key:`chat.transcript.label`,en:`Conversation transcript`,ko:`대화 기록`,test_id:`chat-transcript-label`},{key:`chat.transcript.empty`,en:`No messages yet. Start with a prompt after loading a model.`,ko:`아직 메시지가 없습니다. 모델을 로드한 뒤 프롬프트를 입력해 시작하세요.`,test_id:`chat-transcript-empty`},{key:`chat.transcript.jump`,en:`Jump to latest`,ko:`최신 메시지로 이동`,test_id:`chat-transcript-jump`},{key:`chat.transcript.turn_label`,en:`Turn with {model}`,ko:`{model} 턴`,test_id:`chat-transcript-turn-label`},{key:`chat.message.you`,en:`You`,ko:`나`,test_id:`chat-message-you`},{key:`chat.message.prompt_actions`,en:`Prompt actions`,ko:`프롬프트 작업`,test_id:`chat-message-prompt-actions`},{key:`chat.message.response_actions`,en:`Response actions`,ko:`응답 작업`,test_id:`chat-message-response-actions`},{key:`chat.message.elapsed`,en:`{status} · {seconds} s`,ko:`{status} · {seconds}초`,test_id:`chat-message-elapsed`},{key:`chat.message.copy_prompt`,en:`Copy prompt`,ko:`프롬프트 복사`,test_id:`chat-message-copy-prompt`},{key:`chat.message.copied_prompt`,en:`Copied prompt.`,ko:`프롬프트를 복사했습니다.`,test_id:`chat-message-copied-prompt`},{key:`chat.message.copy_prompt_failed`,en:`Copy unavailable; select the prompt text instead.`,ko:`복사할 수 없습니다. 프롬프트 텍스트를 직접 선택하세요.`,test_id:`chat-message-copy-prompt-failed`},{key:`chat.message.retry`,en:`Retry response`,ko:`응답 다시 시도`,test_id:`chat-message-retry`},{key:`chat.message.not_executed`,en:`Not executed`,ko:`실행하지 않음`,test_id:`chat-message-not-executed`},{key:`chat.transcript.status.streaming`,en:`Generating`,ko:`생성 중`,test_id:`chat-transcript-status-streaming`},{key:`chat.transcript.status.complete`,en:`Done`,ko:`완료`,test_id:`chat-transcript-status-complete`},{key:`chat.transcript.status.cancelled`,en:`Stopped`,ko:`중지됨`,test_id:`chat-transcript-status-cancelled`},{key:`chat.transcript.status.interrupted`,en:`Did not finish`,ko:`완료되지 않음`,test_id:`chat-transcript-status-interrupted`},{key:`chat.transcript.status.error`,en:`Failed`,ko:`실패`,test_id:`chat-transcript-status-error`},{key:`chat.transcript.images`,en:`{count} local image attachment(s)`,ko:`로컬 이미지 첨부 {count}개`,test_id:`chat-transcript-images`},{key:`chat.transcript.edit`,en:`Edit and regenerate`,ko:`편집 후 다시 생성`,test_id:`chat-transcript-edit`},{key:`chat.transcript.edit.confirm.title`,en:`Edit this prompt?`,ko:`이 프롬프트를 편집할까요?`,test_id:`chat-transcript-edit-confirm-title`},{key:`chat.transcript.edit.confirm.body`,en:`This response and all later turns in this conversation are discarded. The prompt returns to the composer.`,ko:`이 응답과 이 대화의 이후 모든 턴이 삭제됩니다. 프롬프트는 입력창으로 돌아갑니다.`,test_id:`chat-transcript-edit-confirm-body`},{key:`chat.transcript.edit.confirm`,en:`Discard and edit`,ko:`삭제하고 편집`,test_id:`chat-transcript-edit-confirm`},{key:`chat.transcript.reasoning_thinking`,en:`Reasoning · Thinking`,ko:`추론 · 생각하는 중`,test_id:`chat-transcript-reasoning-thinking`},{key:`chat.transcript.thinking`,en:`Thinking…`,ko:`생각하는 중…`,test_id:`chat-transcript-thinking`},{key:`chat.transcript.waiting`,en:`Waiting for response…`,ko:`응답을 기다리는 중…`,test_id:`chat-transcript-waiting`},{key:`chat.transcript.tool_call`,en:`Tool call: {name} (not executed)`,ko:`도구 호출: {name} (실행하지 않음)`,test_id:`chat-transcript-tool-call`},{key:`chat.transcript.tool_name_pending`,en:`Name pending`,ko:`이름 대기 중`,test_id:`chat-transcript-tool-name-pending`},{key:`chat.transcript.copy`,en:`Copy response`,ko:`응답 복사`,test_id:`chat-transcript-copy`},{key:`chat.transcript.copied`,en:`Copied response.`,ko:`응답을 복사했습니다.`,test_id:`chat-transcript-copied`},{key:`chat.transcript.copy_failed`,en:`Copy unavailable; select the response text instead.`,ko:`복사할 수 없습니다. 응답 텍스트를 직접 선택하세요.`,test_id:`chat-transcript-copy-failed`},{key:`chat.transcript.details`,en:`Response details`,ko:`응답 상세`,test_id:`chat-transcript-details`},{key:`chat.transcript.finish_reason`,en:`Finish reason`,ko:`종료 사유`,test_id:`chat-transcript-finish-reason`},{key:`chat.transcript.unknown`,en:`Unknown`,ko:`알 수 없음`,test_id:`chat-transcript-unknown`},{key:`chat.transcript.usage`,en:`Usage (server reported)`,ko:`사용량 (서버 보고)`,test_id:`chat-transcript-usage`},{key:`chat.transcript.usage_value`,en:`{prompt} prompt / {completion} completion / {total} total tokens`,ko:`프롬프트 {prompt} / 완성 {completion} / 전체 {total} 토큰`,test_id:`chat-transcript-usage-value`},{key:`chat.transcript.first_delta`,en:`First delta (client observed)`,ko:`첫 델타 (클라이언트 관측)`,test_id:`chat-transcript-first-delta`},{key:`chat.transcript.decode_rate`,en:`Decode rate (client estimate)`,ko:`디코드 속도 (클라이언트 추정)`,test_id:`chat-transcript-decode-rate`},{key:`chat.transcript.metrics_note`,en:`First delta measures send to first non-empty content, reasoning or tool delta, including transport. Decode rate uses reported completion tokens minus one over the remaining client-observed stream duration; it is not a server kernel metric.`,ko:`첫 델타는 전송부터 비어 있지 않은 첫 내용, 추론 또는 도구 델타까지의 시간이며 전송 구간을 포함합니다. 디코드 속도는 보고된 완성 토큰 수에서 1을 뺀 값을 클라이언트가 관측한 나머지 스트림 시간으로 나눈 값이며 서버 커널 지표가 아닙니다.`,test_id:`chat-transcript-metrics-note`},{key:`chat.params.summary`,en:`Parameters for next turn`,ko:`다음 턴 매개변수`,test_id:`chat-params-summary`},{key:`chat.params.body`,en:`These overrides apply once, after Send accepts a request. Blank fields inherit Settings; absent Settings fields inherit the server. Changes never affect a running turn.`,ko:`이 재정의 값은 보내기로 요청이 수락된 뒤 한 번만 적용됩니다. 빈 필드는 설정 값을, 설정에 없는 필드는 서버 값을 상속합니다. 실행 중인 턴에는 영향을 주지 않습니다.`,test_id:`chat-params-body`},{key:`chat.params.field`,en:`Next turn {name}`,ko:`다음 턴 {name}`,test_id:`chat-params-field`},{key:`chat.params.inherited`,en:`Inherited: {value}`,ko:`상속 값: {value}`,test_id:`chat-params-inherited`},{key:`chat.params.clear`,en:`Clear next-turn overrides`,ko:`다음 턴 재정의 지우기`,test_id:`chat-params-clear`},{key:`chat.privacy.summary`,en:`Local history and privacy`,ko:`로컬 기록과 개인정보`,test_id:`chat-privacy-summary`},{key:`chat.privacy.body`,en:`Conversations stay in memory unless you opt in below. Signing out or refreshing requires authentication again; saved history belongs to this browser origin, not a server account. No API keys are stored.`,ko:`아래에서 동의하지 않으면 대화는 메모리에만 남습니다. 로그아웃하거나 새로고침하면 다시 인증해야 합니다. 저장된 기록은 서버 계정이 아니라 이 브라우저 origin에 속합니다. API 키는 저장하지 않습니다.`,test_id:`chat-privacy-body`},{key:`chat.privacy.save`,en:`Save conversations on this device (IndexedDB)`,ko:`이 기기에 대화 저장 (IndexedDB)`,test_id:`chat-privacy-save`},{key:`chat.privacy.include_images`,en:`Also include attached image bytes in saved history and exports (separate consent)`,ko:`저장 기록과 내보내기에 첨부 이미지 바이트도 포함 (별도 동의)`,test_id:`chat-privacy-include-images`},{key:`chat.privacy.note`,en:`Turning saving off stops future writes; use Clear All to remove previously saved history. Re-enable explicitly after refresh to read it.`,ko:`저장을 끄면 이후 쓰기만 멈춥니다. 이미 저장된 기록은 모두 지우기로 삭제하세요. 새로고침 후 기록을 읽으려면 다시 명시적으로 켜세요.`,test_id:`chat-privacy-note`},{key:`chat.privacy.export`,en:`Export JSON`,ko:`JSON 내보내기`,test_id:`chat-privacy-export`},{key:`chat.privacy.export_failed`,en:`Export exceeds the bounded history size or contains unsupported data.`,ko:`내보내기가 기록 크기 한도를 넘거나 지원하지 않는 데이터를 포함합니다.`,test_id:`chat-privacy-export-failed`},{key:`chat.privacy.clear`,en:`Clear All`,ko:`모두 지우기`,test_id:`chat-privacy-clear`},{key:`chat.privacy.cleared`,en:`All history cleared.`,ko:`모든 기록을 지웠습니다.`,test_id:`chat-privacy-cleared`},{key:`chat.privacy.clear_failed`,en:`Stored history could not be cleared. Check browser storage permissions; no success is claimed.`,ko:`저장된 기록을 지우지 못했습니다. 브라우저 저장소 권한을 확인하세요. 성공으로 처리하지 않았습니다.`,test_id:`chat-privacy-clear-failed`},{key:`chat.privacy.import`,en:`Import conversation JSON (replaces current history)`,ko:`대화 JSON 가져오기 (현재 기록을 대체)`,test_id:`chat-privacy-import`},{key:`chat.privacy.import_too_large`,en:`Import exceeds 16 MiB.`,ko:`가져오기가 16 MiB를 넘습니다.`,test_id:`chat-privacy-import-too-large`},{key:`chat.privacy.import_rejected`,en:`Import rejected: unsupported version, invalid data or exceeded limits.`,ko:`가져오기를 거부했습니다. 지원하지 않는 버전, 잘못된 데이터 또는 한도 초과입니다.`,test_id:`chat-privacy-import-rejected`},{key:`chat.privacy.save_failed`,en:`History was not saved. Storage may be full or unavailable; export your conversation.`,ko:`기록을 저장하지 못했습니다. 저장소가 가득 찼거나 사용할 수 없을 수 있으니 대화를 내보내세요.`,test_id:`chat-privacy-save-failed`},{key:`chat.privacy.open_failed`,en:`Local history could not be opened or its images failed validation. Nothing was saved.`,ko:`로컬 기록을 열 수 없거나 이미지 검증에 실패했습니다. 아무것도 저장하지 않았습니다.`,test_id:`chat-privacy-open-failed`},{key:`chat.privacy.title`,en:`Local history`,ko:`로컬 기록`,test_id:`chat-privacy-title`},{key:`chat.privacy.replace_saved.confirm.title`,en:`Replace the in-memory conversations with saved history?`,ko:`메모리의 대화를 저장된 기록으로 바꿀까요?`,test_id:`chat-privacy-replace-saved-confirm-title`},{key:`chat.privacy.replace_saved.confirm.body`,en:`The conversations in memory are discarded and replaced by the saved history. Export current conversations first if needed.`,ko:`메모리의 대화는 삭제되고 저장된 기록으로 대체됩니다. 필요하면 먼저 현재 대화를 내보내세요.`,test_id:`chat-privacy-replace-saved-confirm-body`},{key:`chat.privacy.replace_saved.confirm`,en:`Replace with saved history`,ko:`저장된 기록으로 바꾸기`,test_id:`chat-privacy-replace-saved-confirm`},{key:`chat.privacy.clear.confirm.title`,en:`Delete all in-memory and saved conversations for this browser origin?`,ko:`이 브라우저 origin의 메모리 대화와 저장된 대화를 모두 삭제할까요?`,test_id:`chat-privacy-clear-confirm-title`},{key:`chat.privacy.clear.confirm`,en:`Delete all history`,ko:`모든 기록 삭제`,test_id:`chat-privacy-clear-confirm`},{key:`chat.privacy.replace_import.confirm.title`,en:`Replace the current conversations with validated imported history?`,ko:`현재 대화를 검증된 가져온 기록으로 바꿀까요?`,test_id:`chat-privacy-replace-import-confirm-title`},{key:`chat.privacy.replace_import.confirm.body`,en:`The current conversations are discarded and replaced by the imported history. Export current conversations first if needed.`,ko:`현재 대화는 삭제되고 가져온 기록으로 대체됩니다. 필요하면 먼저 현재 대화를 내보내세요.`,test_id:`chat-privacy-replace-import-confirm-body`},{key:`chat.privacy.replace_import.confirm`,en:`Replace with imported history`,ko:`가져온 기록으로 바꾸기`,test_id:`chat-privacy-replace-import-confirm`}],Te=[{key:`settings.server.privacy.body`,en:`Appearance and explicitly saved load profiles use this browser’s preferences. Request defaults stay in memory; API credentials and system prompts are never stored by Settings. Conversation history is controlled separately in Chat.`,ko:`모양과 명시적으로 저장한 로드 프로필은 브라우저 환경설정을 사용합니다. 요청 기본값은 메모리에만 유지하며 API 자격 증명과 시스템 프롬프트는 설정에서 저장하지 않습니다. 대화 기록은 대화 화면에서 별도로 관리합니다.`,test_id:`settings-server-privacy-body`},{key:`settings.server.unavailable.title`,en:`Server controls unavailable`,ko:`서버 제어 사용 불가`,test_id:`settings-server-unavailable-title`},{key:`settings.server.unavailable.body`,en:`Connect from Models or Chat, or refresh the connection before changing server settings. Browser appearance and request defaults remain available. Schema mismatch disables server controls.`,ko:`모델이나 대화 화면에서 연결하거나 연결을 새로 고친 후 서버 설정을 변경하세요. 브라우저 모양과 요청 기본값은 계속 사용할 수 있습니다. 스키마 불일치 시 서버 제어가 비활성화됩니다.`,test_id:`settings-server-unavailable-body`},{key:`settings.server.model`,en:`Selected model for settings`,ko:`설정 대상 모델`,test_id:`settings-server-model`},{key:`settings.server.model.none`,en:`No model selected`,ko:`선택한 모델 없음`,test_id:`settings-server-model-none`},{key:`settings.server.model.hint`,en:`Selection changes browser context only. It never loads, unloads or reroutes an existing request.`,ko:`선택은 브라우저 문맥만 변경하며 모델을 로드하거나 언로드하거나 기존 요청의 대상을 변경하지 않습니다.`,test_id:`settings-server-model-hint`},{key:`settings.server.context.title`,en:`Effective context · observed, not estimated`,ko:`실제 컨텍스트 · 추정이 아닌 관측값`,test_id:`settings-server-context-title`},{key:`settings.server.context.n_ctx`,en:`Per-slot context tokens (/props n_ctx)`,ko:`슬롯별 컨텍스트 토큰 (/props n_ctx)`,test_id:`settings-server-context-n-ctx`},{key:`settings.server.context.n_ctx_hint`,en:`Zero/missing is unknown; do not multiply this value into a shared-pool capacity. Unified KV remains separate work (#1815).`,ko:`0 또는 누락은 알 수 없음입니다. 이 값에 슬롯 수를 곱해 공유 풀 용량으로 간주하지 않습니다. 통합 KV는 별도 작업입니다 (#1815).`,test_id:`settings-server-context-n-ctx-hint`},{key:`settings.server.context.slots`,en:`Configured slots (/props total_slots)`,ko:`설정된 슬롯 (/props total_slots)`,test_id:`settings-server-context-slots`},{key:`settings.server.context.kv_mode`,en:`Active resolved KV cache mode (/props)`,ko:`현재 실제 KV 캐시 모드 (/props)`,test_id:`settings-server-context-kv-mode`},{key:`settings.server.context.geometry`,en:`Reported context geometry`,ko:`보고된 컨텍스트 구성`,test_id:`settings-server-context-geometry`},{key:`settings.server.context.geometry_unknown`,en:`Unknown / props disabled or unavailable`,ko:`알 수 없음 / props 비활성화 또는 사용 불가`,test_id:`settings-server-context-geometry-unknown`},{key:`settings.server.unknown`,en:`Unknown`,ko:`알 수 없음`,test_id:`settings-server-unknown`},{key:`settings.server.no_model.title`,en:`No ready model selected`,ko:`준비된 모델이 선택되지 않음`,test_id:`settings-server-no-model-title`},{key:`settings.server.no_model.body`,en:`Selection never loads a model. Load explicitly from Models to read live settings.`,ko:`모델 선택은 로드를 수행하지 않습니다. 모델 화면에서 명시적으로 로드하여 실시간 설정을 확인하세요.`,test_id:`settings-server-no-model-body`},{key:`settings.server.live_disabled.title`,en:`Live settings disabled by operator`,ko:`운영자가 실시간 설정을 비활성화함`,test_id:`settings-server-live-disabled-title`},{key:`settings.server.live_disabled.body`,en:`Restart with --settings to enable this endpoint. WebUI never enables settings, props, metrics or slots implicitly.`,ko:`엔드포인트를 활성화하려면 --settings로 재시작하세요. WebUI는 settings, props, metrics 또는 slots를 자동으로 활성화하지 않습니다.`,test_id:`settings-server-live-disabled-body`},{key:`settings.generation.title`,en:`Generation · next request`,ko:`생성 · 다음 요청`,test_id:`settings-generation-title`},{key:`settings.generation.body`,en:`These defaults stay in browser memory only. Blank fields are omitted and inherit server defaults. Existing turns are frozen; changing defaults never alters an in-flight request. System prompts belong to individual conversations in Chat and are not stored here.`,ko:`기본값은 브라우저 메모리에만 남습니다. 빈 필드는 생략되어 서버 기본값을 사용합니다. 진행 중인 요청은 변경되지 않습니다. 시스템 프롬프트는 대화 화면의 개별 대화에서 설정하며 여기에 저장하지 않습니다.`,test_id:`settings-generation-body`},{key:`settings.generation.invalid`,en:`Invalid defaults`,ko:`잘못된 기본값`,test_id:`settings-generation-invalid`},{key:`settings.generation.error_title`,en:`Invalid request defaults`,ko:`잘못된 요청 기본값`,test_id:`settings-generation-error-title`},{key:`settings.generation.save`,en:`Save session defaults`,ko:`세션 기본값 저장`,test_id:`settings-generation-save`},{key:`settings.generation.reset_open`,en:`Reset request defaults…`,ko:`요청 기본값 초기화…`,test_id:`settings-generation-reset-open`},{key:`settings.generation.reset.title`,en:`Reset request defaults?`,ko:`요청 기본값을 초기화할까요?`,test_id:`settings-generation-reset-title`},{key:`settings.generation.reset.body`,en:`Only browser-session request defaults will be cleared. Server and next-load profiles remain unchanged.`,ko:`브라우저 세션 요청 기본값만 지워집니다. 서버와 다음 로드 프로필은 변경되지 않습니다.`,test_id:`settings-generation-reset-body`},{key:`settings.generation.reset.confirm`,en:`Reset request defaults`,ko:`요청 기본값 초기화`,test_id:`settings-generation-reset-confirm`},{key:`settings.live.title`,en:`Loaded model · live server values`,ko:`로드된 모델 · 실시간 서버 값`,test_id:`settings-live-title`},{key:`settings.live.body`,en:`Changes affect newly admitted requests. Fingerprints detect already-observed external changes, not atomic compare-and-swap. Numeric bounds remain enforced by the server; this schema publishes types and allowed enums, not numeric min/max.`,ko:`변경은 새로 수락되는 요청에 적용됩니다. 지문은 관측된 외부 변경을 감지하지만 원자적 비교 교환은 아닙니다. 숫자 범위는 서버가 검증합니다. 스키마에는 타입과 허용 열거값만 있습니다.`,test_id:`settings-live-body`},{key:`settings.live.unavailable`,en:`Settings unavailable. The server may have disabled --settings.`,ko:`설정을 사용할 수 없습니다. 서버에서 --settings가 비활성화되었을 수 있습니다.`,test_id:`settings-live-unavailable`},{key:`settings.live.refreshed`,en:`Current values refreshed. Review the retained draft before applying.`,ko:`현재 값을 새로 고쳤습니다. 남아 있는 초안을 검토한 후 적용하세요.`,test_id:`settings-live-refreshed`},{key:`settings.live.refresh_failed`,en:`Refresh failed. No changes submitted.`,ko:`새로 고침에 실패했습니다. 변경 사항은 전송되지 않았습니다.`,test_id:`settings-live-refresh-failed`},{key:`settings.live.invalid_value`,en:`Invalid value`,ko:`잘못된 값`,test_id:`settings-live-invalid-value`},{key:`settings.live.conflict`,en:`Another client changed these settings. Current values refreshed; review and Apply again to reconfirm.`,ko:`다른 클라이언트가 설정을 변경했습니다. 갱신된 현재 값을 검토하고 다시 적용하여 확인하세요.`,test_id:`settings-live-conflict`},{key:`settings.live.partial`,en:`{applied} applied; {rejected} rejected. Rejected drafts retained.`,ko:`{applied}개 적용, {rejected}개 거부됨. 거부된 초안은 유지됩니다.`,test_id:`settings-live-partial`},{key:`settings.live.applied`,en:`Accepted fields applied; reading effective values.`,ko:`승인된 필드를 적용했습니다. 실제 값을 다시 읽습니다.`,test_id:`settings-live-applied`},{key:`settings.live.unknown_outcome`,en:`The outcome may be unknown. Refresh effective values before retrying; no automatic PATCH retry.`,ko:`결과를 확정할 수 없습니다. 재시도 전에 실제 값을 새로 고치세요. PATCH는 자동 재시도하지 않습니다.`,test_id:`settings-live-unknown-outcome`},{key:`settings.live.result_title`,en:`Settings result`,ko:`설정 결과`,test_id:`settings-live-result-title`},{key:`settings.live.refresh`,en:`Refresh current values`,ko:`현재 값 새로 고침`,test_id:`settings-live-refresh`},{key:`settings.live.apply`,en:`Apply live draft`,ko:`실시간 초안 적용`,test_id:`settings-live-apply`},{key:`settings.live.reset_open`,en:`Reset live draft…`,ko:`실시간 초안 초기화…`,test_id:`settings-live-reset-open`},{key:`settings.live.startup`,en:`Server startup · restart required (read-only)`,ko:`서버 시작 · 재시작 필요 (읽기 전용)`,test_id:`settings-live-startup`},{key:`settings.live.reset.title`,en:`Reset live draft only?`,ko:`실시간 초안만 초기화할까요?`,test_id:`settings-live-reset-title`},{key:`settings.live.reset.body`,en:`Stage server startup defaults for mutable fields. No server change occurs until Apply.`,ko:`변경 가능한 필드에 서버 시작 기본값을 초안으로 설정합니다. 적용 전에는 서버가 변경되지 않습니다.`,test_id:`settings-live-reset-body`},{key:`settings.live.reset.confirm`,en:`Reset draft`,ko:`초안 초기화`,test_id:`settings-live-reset-confirm`},{key:`settings.profile.title`,en:`Next load · browser profile`,ko:`다음 로드 · 브라우저 프로필`,test_id:`settings-profile-title`},{key:`settings.profile.body`,en:`Explicit CLI settings take precedence over this profile; the profile overrides a preset only where CLI has not pinned a value. Effective resolved values must be read after loading. Saving is not loading. A loaded model requires an explicit unload and load from Models; failed loads retain this pending profile.`,ko:`명시적 CLI 설정이 프로필보다 우선합니다. CLI로 고정되지 않은 값에만 프로필이 프리셋보다 우선합니다. 실제 값은 로드 후 확인하세요. 저장은 로드가 아닙니다. 로드된 모델은 모델 화면에서 명시적으로 언로드 후 로드해야 하며, 실패 시 프로필은 대기 상태로 남습니다.`,test_id:`settings-profile-body`},{key:`settings.profile.saved`,en:`Browser profile saved. Nothing loaded or changed on the server.`,ko:`브라우저 프로필을 저장했습니다. 서버에서 로드하거나 변경한 내용은 없습니다.`,test_id:`settings-profile-saved`},{key:`settings.profile.failed`,en:`Profile operation failed`,ko:`프로필 작업에 실패했습니다`,test_id:`settings-profile-failed`},{key:`settings.profile.single.title`,en:`Single-model mode: restart required`,ko:`단일 모델 모드: 재시작 필요`,test_id:`settings-profile-single-title`},{key:`settings.profile.single.body`,en:`These preferences cannot be applied to this running worker. Restart with validated CLI flags, or start model-free mode to use load operations.`,ko:`실행 중인 워커에 적용할 수 없습니다. 검증된 CLI 플래그로 재시작하거나 모델 없는 모드에서 로드 작업을 사용하세요.`,test_id:`settings-profile-single-body`},{key:`settings.profile.scope`,en:`Profile scope`,ko:`프로필 범위`,test_id:`settings-profile-scope`},{key:`settings.profile.scope.reusable`,en:`Reusable defaults`,ko:`재사용 기본값`,test_id:`settings-profile-scope-reusable`},{key:`settings.profile.ctx_hint`,en:`Blank: inherit. Actual per-slot context is resolved by the server, not estimated here.`,ko:`빈 값: 상속. 슬롯별 실제 컨텍스트는 서버가 결정하며 여기서 추정하지 않습니다.`,test_id:`settings-profile-ctx-hint`},{key:`settings.profile.parallel_hint`,en:`Scheduler/worker geometry changes only on the next explicit load.`,ko:`스케줄러/워커 구성은 다음 명시적 로드에서만 변경됩니다.`,test_id:`settings-profile-parallel-hint`},{key:`settings.profile.inherit`,en:`Inherit`,ko:`상속`,test_id:`settings-profile-inherit`},{key:`settings.profile.result_title`,en:`Profile result`,ko:`프로필 결과`,test_id:`settings-profile-result-title`},{key:`settings.profile.save`,en:`Save pending profile`,ko:`대기 프로필 저장`,test_id:`settings-profile-save`},{key:`settings.profile.discard`,en:`Discard draft`,ko:`초안 취소`,test_id:`settings-profile-discard`},{key:`settings.profile.reset_open`,en:`Reset selected profile…`,ko:`선택 프로필 초기화…`,test_id:`settings-profile-reset-open`},{key:`settings.profile.saved_pending`,en:`Saved pending profile (not active)`,ko:`저장된 대기 프로필 (미적용)`,test_id:`settings-profile-saved-pending`},{key:`settings.profile.cli`,en:`Sanitized CLI flags · display only; model source omitted`,ko:`안전한 CLI 플래그 · 표시 전용; 모델 소스 생략`,test_id:`settings-profile-cli`},{key:`settings.profile.transfer`,en:`Import / export browser profiles`,ko:`브라우저 프로필 가져오기 / 내보내기`,test_id:`settings-profile-transfer`},{key:`settings.profile.transfer.label`,en:`Versioned profile JSON`,ko:`버전이 있는 프로필 JSON`,test_id:`settings-profile-transfer-label`},{key:`settings.profile.transfer.export`,en:`Export to text`,ko:`텍스트로 내보내기`,test_id:`settings-profile-transfer-export`},{key:`settings.profile.transfer.import`,en:`Validate and import`,ko:`검증 후 가져오기`,test_id:`settings-profile-transfer-import`},{key:`settings.profile.reset.title`,en:`Reset this browser profile?`,ko:`이 브라우저 프로필을 초기화할까요?`,test_id:`settings-profile-reset-title`},{key:`settings.profile.reset.body`,en:`Resets only the selected scope. Model scope falls back to reusable defaults; reusable scope leaves model-specific profiles intact. Does not change a running worker.`,ko:`선택한 범위만 초기화합니다. 모델 범위는 재사용 기본값으로 돌아가며 재사용 범위는 모델별 프로필을 유지합니다. 실행 중인 워커는 변경하지 않습니다.`,test_id:`settings-profile-reset-body`},{key:`settings.profile.reset.confirm`,en:`Reset profile`,ko:`프로필 초기화`,test_id:`settings-profile-reset-confirm`},{key:`settings.context_check.label`,en:`Optional raw prompt budget check`,ko:`선택적 원시 프롬프트 예산 확인`,test_id:`settings-context-check-label`},{key:`settings.context_check.hint`,en:`Sent only when Check is pressed; never persisted. Raw tokens exclude chat templates, conversation history and media. Server validation is final.`,ko:`확인을 누를 때만 전송하며 저장하지 않습니다. 원시 토큰은 대화 템플릿, 대화 기록, 미디어를 제외합니다. 최종 검증은 서버가 수행합니다.`,test_id:`settings-context-check-hint`},{key:`settings.context_check.check`,en:`Check with model tokenizer`,ko:`모델 토크나이저로 확인`,test_id:`settings-context-check-check`},{key:`settings.context_check.unavailable`,en:`Tokenization unavailable; budget unknown.`,ko:`토큰화 불가; 예산을 알 수 없습니다.`,test_id:`settings-context-check-unavailable`},{key:`settings.context_check.not_measured`,en:`Not measured`,ko:`측정하지 않음`,test_id:`settings-context-check-not-measured`},{key:`settings.context_check.count`,en:`{count} raw tokens. {verdict}`,ko:`{count} 원시 토큰. {verdict}`,test_id:`settings-context-check-count`},{key:`settings.context_check.exceeds`,en:`Raw input + requested output already exceeds observed context.`,ko:`원시 입력 + 요청 출력이 관측 컨텍스트를 초과합니다.`,test_id:`settings-context-check-exceeds`},{key:`settings.context_check.unknown`,en:`Context/output is unknown; no fit claim.`,ko:`컨텍스트/출력을 알 수 없어 수용 가능 여부를 판단하지 않습니다.`,test_id:`settings-context-check-unknown`},{key:`settings.context_check.fits`,en:`Raw input + output fits, but templated chat may still exceed the context.`,ko:`원시 입력 + 출력은 들어가지만 템플릿이 적용된 대화는 컨텍스트를 초과할 수 있습니다.`,test_id:`settings-context-check-fits`},{key:`settings.sections`,en:`Settings sections`,ko:`설정 섹션`,test_id:`settings-sections`},{key:`settings.section.appearance`,en:`Appearance`,ko:`화면 표시`,test_id:`settings-section-appearance`},{key:`settings.section.requests`,en:`Requests`,ko:`요청`,test_id:`settings-section-requests`},{key:`settings.section.model`,en:`Model`,ko:`모델`,test_id:`settings-section-model`},{key:`settings.section.server`,en:`Server`,ko:`서버`,test_id:`settings-section-server`},{key:`settings.section.requests.description`,en:`Defaults for the next request from this browser. Memory only; blank fields inherit.`,ko:`이 브라우저에서 보내는 다음 요청의 기본값입니다. 메모리에만 유지되며 빈 필드는 상속합니다.`,test_id:`settings-section-requests-description`},{key:`settings.section.model.description`,en:`Next-load profile and live values for the selected model.`,ko:`선택한 모델의 다음 로드 프로필과 실시간 값입니다.`,test_id:`settings-section-model-description`},{key:`settings.section.server.description`,en:`Read-only values fixed when the server started, and the observed context.`,ko:`서버 시작 시 고정된 읽기 전용 값과 관측된 컨텍스트입니다.`,test_id:`settings-section-server-description`},{key:`settings.help.storage`,en:`About Settings storage`,ko:`설정 저장 방식 안내`,test_id:`settings-help-storage`},{key:`settings.help.requests`,en:`About request defaults`,ko:`요청 기본값 안내`,test_id:`settings-help-requests`},{key:`settings.help.profile`,en:`About next-load profiles`,ko:`다음 로드 프로필 안내`,test_id:`settings-help-profile`},{key:`settings.help.live`,en:`About live values`,ko:`실시간 값 안내`,test_id:`settings-help-live`},{key:`settings.help.context`,en:`About the effective context`,ko:`실제 컨텍스트 안내`,test_id:`settings-help-context`},{key:`settings.requests.effective`,en:`Next request: {value}`,ko:`다음 요청: {value}`,test_id:`settings-requests-effective`},{key:`settings.requests.value_source`,en:`{value} ({source})`,ko:`{value} ({source})`,test_id:`settings-requests-value-source`},{key:`settings.requests.source.override`,en:`next-turn override`,ko:`다음 턴 재정의`,test_id:`settings-requests-source-override`},{key:`settings.requests.source.session`,en:`session default`,ko:`세션 기본값`,test_id:`settings-requests-source-session`},{key:`settings.requests.source.server`,en:`server default`,ko:`서버 기본값`,test_id:`settings-requests-source-server`},{key:`settings.requests.unknown`,en:`server default, not readable`,ko:`서버 기본값, 읽을 수 없음`,test_id:`settings-requests-unknown`},{key:`settings.requests.server_from`,en:`Server defaults read from {model}.`,ko:`{model}에서 서버 기본값을 읽었습니다.`,test_id:`settings-requests-server-from`},{key:`settings.requests.server_loading`,en:`Reading server defaults from {model}…`,ko:`{model}에서 서버 기본값을 읽는 중…`,test_id:`settings-requests-server-loading`},{key:`settings.requests.server_failed`,en:`Server defaults could not be read from {model}.`,ko:`{model}에서 서버 기본값을 읽지 못했습니다.`,test_id:`settings-requests-server-failed`},{key:`settings.requests.server_none`,en:`No ready model selected, so server defaults are not readable.`,ko:`준비된 모델이 선택되지 않아 서버 기본값을 읽을 수 없습니다.`,test_id:`settings-requests-server-none`},{key:`settings.requests.server_disabled`,en:`Live settings are disabled, so server defaults are not readable.`,ko:`실시간 설정이 비활성화되어 서버 기본값을 읽을 수 없습니다.`,test_id:`settings-requests-server-disabled`},{key:`settings.requests.server_offline`,en:`Not connected, so server defaults are not readable.`,ko:`연결되지 않아 서버 기본값을 읽을 수 없습니다.`,test_id:`settings-requests-server-offline`},{key:`settings.value.unset`,en:`not set`,ko:`설정 안 됨`,test_id:`settings-value-unset`},{key:`settings.control.unset`,en:`Leave unset`,ko:`설정하지 않음`,test_id:`settings-control-unset`},{key:`settings.control.invalid_json`,en:`Enter valid JSON`,ko:`올바른 JSON을 입력하세요`,test_id:`settings-control-invalid-json`},{key:`settings.control.not_allowed`,en:`{value} (not an allowed value)`,ko:`{value} (허용되지 않는 값)`,test_id:`settings-control-not-allowed`},{key:`settings.control.server_value`,en:`Server value: {value}`,ko:`서버 값: {value}`,test_id:`settings-control-server-value`},{key:`settings.live.group.sampling`,en:`Sampling`,ko:`샘플링`,test_id:`settings-live-group-sampling`},{key:`settings.live.group.dry`,en:`DRY`,ko:`DRY`,test_id:`settings-live-group-dry`},{key:`settings.live.group.diffusion`,en:`Diffusion`,ko:`디퓨전`,test_id:`settings-live-group-diffusion`},{key:`settings.live.group.template`,en:`Template`,ko:`템플릿`,test_id:`settings-live-group-template`},{key:`settings.live.group.other`,en:`Other`,ko:`기타`,test_id:`settings-live-group-other`},{key:`settings.profile.ctx_size`,en:`Context size (1–262144)`,ko:`컨텍스트 크기 (1–262144)`,test_id:`settings-profile-ctx-size`},{key:`settings.profile.n_parallel`,en:`Parallel slots (1–32)`,ko:`병렬 슬롯 (1–32)`,test_id:`settings-profile-n-parallel`},{key:`settings.profile.kv_cache_mode`,en:`KV cache mode`,ko:`KV 캐시 모드`,test_id:`settings-profile-kv-cache-mode`},{key:`settings.profile.show_cli`,en:`Show as CLI flags`,ko:`CLI 플래그로 보기`,test_id:`settings-profile-show-cli`},{key:`settings.profile.copy`,en:`Copy flags`,ko:`플래그 복사`,test_id:`settings-profile-copy`},{key:`settings.profile.copied`,en:`CLI flags copied.`,ko:`CLI 플래그를 복사했습니다.`,test_id:`settings-profile-copied`},{key:`settings.profile.copy_failed`,en:`Copy failed. Select the flags and copy them instead.`,ko:`복사하지 못했습니다. 플래그를 선택해 직접 복사하세요.`,test_id:`settings-profile-copy-failed`},{key:`settings.server.startup.reasons`,en:`Why these values are read-only`,ko:`읽기 전용인 이유`,test_id:`settings-server-startup-reasons`},{key:`settings.server.props_unavailable.title`,en:`Context unavailable`,ko:`컨텍스트를 확인할 수 없음`,test_id:`settings-server-props-unavailable-title`},{key:`settings.server.props_unavailable.body`,en:`The model did not answer /props. Restart with --props to observe its context.`,ko:`모델이 /props에 응답하지 않았습니다. 컨텍스트를 확인하려면 --props로 재시작하세요.`,test_id:`settings-server-props-unavailable-body`},{key:`settings.server.startup.failed`,en:`Startup values unavailable`,ko:`시작 값을 확인할 수 없음`,test_id:`settings-server-startup-failed`}],Ee=[{key:`settings.live.timeout_seconds`,en:`Decode timeout (seconds)`,ko:`디코드 제한 시간(초)`,test_id:`settings-live-timeout-seconds`},{key:`settings.live.timeout_seconds.help`,en:`Decode watchdog used by newly admitted requests.`,ko:`새로 수락되는 요청에 적용되는 디코드 감시 시간입니다.`,test_id:`settings-live-timeout-seconds-help`},{key:`settings.live.default_temperature`,en:`Temperature`,ko:`온도`,test_id:`settings-live-default-temperature`},{key:`settings.live.default_temperature.help`,en:`Sampling temperature used when a request omits temperature.`,ko:`요청에 temperature가 없을 때 사용하는 샘플링 온도입니다.`,test_id:`settings-live-default-temperature-help`},{key:`settings.live.default_top_p`,en:`Top P`,ko:`Top P`,test_id:`settings-live-default-top-p`},{key:`settings.live.default_top_p.help`,en:`Top-p sampling default, in (0, 1].`,ko:`(0, 1] 범위의 Top-p 샘플링 기본값입니다.`,test_id:`settings-live-default-top-p-help`},{key:`settings.live.default_top_k`,en:`Top K`,ko:`Top K`,test_id:`settings-live-default-top-k`},{key:`settings.live.default_top_k.help`,en:`Top-k sampling default.`,ko:`Top-k 샘플링 기본값입니다.`,test_id:`settings-live-default-top-k-help`},{key:`settings.live.default_min_p`,en:`Min P`,ko:`Min P`,test_id:`settings-live-default-min-p`},{key:`settings.live.default_min_p.help`,en:`Min-p sampling default, in [0, 1].`,ko:`[0, 1] 범위의 Min-p 샘플링 기본값입니다.`,test_id:`settings-live-default-min-p-help`},{key:`settings.live.default_repetition_penalty`,en:`Repetition penalty`,ko:`반복 페널티`,test_id:`settings-live-default-repetition-penalty`},{key:`settings.live.default_repetition_penalty.help`,en:`Repetition penalty used when a request omits it.`,ko:`요청에 없을 때 사용하는 반복 페널티입니다.`,test_id:`settings-live-default-repetition-penalty-help`},{key:`settings.live.default_repetition_context_size`,en:`Repetition window`,ko:`반복 검사 범위`,test_id:`settings-live-default-repetition-context-size`},{key:`settings.live.default_repetition_context_size.help`,en:`Token history window that repetition penalties check.`,ko:`반복 페널티가 검사하는 토큰 기록 범위입니다.`,test_id:`settings-live-default-repetition-context-size-help`},{key:`settings.live.default_max_tokens`,en:`Max tokens`,ko:`최대 토큰`,test_id:`settings-live-default-max-tokens`},{key:`settings.live.default_max_tokens.help`,en:`Generation budget used when a request omits max_tokens.`,ko:`요청에 max_tokens가 없을 때 사용하는 생성 한도입니다.`,test_id:`settings-live-default-max-tokens-help`},{key:`settings.live.default_seed`,en:`Seed`,ko:`시드`,test_id:`settings-live-default-seed`},{key:`settings.live.default_seed.help`,en:`Seed used when a request omits one. Left unset, each request gets a random seed.`,ko:`요청에 시드가 없을 때 사용하는 값입니다. 설정하지 않으면 요청마다 무작위 시드를 사용합니다.`,test_id:`settings-live-default-seed-help`},{key:`settings.live.default_frequency_penalty`,en:`Frequency penalty`,ko:`빈도 페널티`,test_id:`settings-live-default-frequency-penalty`},{key:`settings.live.default_frequency_penalty.help`,en:`Frequency penalty used when a request omits it.`,ko:`요청에 없을 때 사용하는 빈도 페널티입니다.`,test_id:`settings-live-default-frequency-penalty-help`},{key:`settings.live.default_presence_penalty`,en:`Presence penalty`,ko:`존재 페널티`,test_id:`settings-live-default-presence-penalty`},{key:`settings.live.default_presence_penalty.help`,en:`Presence penalty used when a request omits it.`,ko:`요청에 없을 때 사용하는 존재 페널티입니다.`,test_id:`settings-live-default-presence-penalty-help`},{key:`settings.live.default_dry_multiplier`,en:`DRY multiplier`,ko:`DRY 배수`,test_id:`settings-live-default-dry-multiplier`},{key:`settings.live.default_dry_multiplier.help`,en:`DRY repetition multiplier.`,ko:`DRY 반복 억제 배수입니다.`,test_id:`settings-live-default-dry-multiplier-help`},{key:`settings.live.default_dry_base`,en:`DRY base`,ko:`DRY 밑`,test_id:`settings-live-default-dry-base`},{key:`settings.live.default_dry_base.help`,en:`DRY exponential base.`,ko:`DRY 지수의 밑입니다.`,test_id:`settings-live-default-dry-base-help`},{key:`settings.live.default_dry_allowed_length`,en:`DRY allowed length`,ko:`DRY 허용 길이`,test_id:`settings-live-default-dry-allowed-length`},{key:`settings.live.default_dry_allowed_length.help`,en:`Repeat length DRY allows before penalizing.`,ko:`DRY가 페널티 없이 허용하는 반복 길이입니다.`,test_id:`settings-live-default-dry-allowed-length-help`},{key:`settings.live.default_dry_penalty_last_n`,en:`DRY history window`,ko:`DRY 기록 범위`,test_id:`settings-live-default-dry-penalty-last-n`},{key:`settings.live.default_dry_penalty_last_n.help`,en:`Token history window that DRY checks.`,ko:`DRY가 검사하는 토큰 기록 범위입니다.`,test_id:`settings-live-default-dry-penalty-last-n-help`},{key:`settings.live.default_dry_sequence_breakers`,en:`DRY sequence breakers`,ko:`DRY 시퀀스 구분자`,test_id:`settings-live-default-dry-sequence-breakers`},{key:`settings.live.default_dry_sequence_breakers.help`,en:`Strings that end a DRY repeat, as a JSON array.`,ko:`DRY 반복을 끊는 문자열 목록(JSON 배열)입니다.`,test_id:`settings-live-default-dry-sequence-breakers-help`},{key:`settings.live.lang_bias_config`,en:`Language bias`,ko:`언어 편향`,test_id:`settings-live-lang-bias-config`},{key:`settings.live.lang_bias_config.help`,en:`Language-bias policy for newly admitted requests, as a JSON object.`,ko:`새로 수락되는 요청에 적용되는 언어 편향 정책(JSON 객체)입니다.`,test_id:`settings-live-lang-bias-config-help`},{key:`settings.live.reasoning_budget`,en:`Reasoning budget`,ko:`추론 예산`,test_id:`settings-live-reasoning-budget`},{key:`settings.live.reasoning_budget.help`,en:`-1 is unbounded, 0 ends thinking immediately, and N caps reasoning tokens.`,ko:`-1은 무제한, 0은 추론을 즉시 종료, N은 추론 토큰 상한입니다.`,test_id:`settings-live-reasoning-budget-help`},{key:`settings.live.chat_template_kwargs`,en:`Chat template arguments`,ko:`대화 템플릿 인수`,test_id:`settings-live-chat-template-kwargs`},{key:`settings.live.chat_template_kwargs.help`,en:`Template arguments merged under each request's own, as a JSON object.`,ko:`요청별 인수 아래에 병합되는 템플릿 인수(JSON 객체)입니다.`,test_id:`settings-live-chat-template-kwargs-help`},{key:`settings.live.loop_detection`,en:`Loop detection`,ko:`반복 루프 감지`,test_id:`settings-live-loop-detection`},{key:`settings.live.loop_detection.help`,en:`Server-wide N-gram loop-detection override, as a JSON object.`,ko:`서버 전체의 N-gram 반복 루프 감지 재정의(JSON 객체)입니다.`,test_id:`settings-live-loop-detection-help`},{key:`settings.live.max_denoising_steps`,en:`Max denoising steps`,ko:`최대 디노이징 단계`,test_id:`settings-live-max-denoising-steps`},{key:`settings.live.max_denoising_steps.help`,en:`Optional cap on diffusion denoising steps.`,ko:`디퓨전 디노이징 단계의 선택적 상한입니다.`,test_id:`settings-live-max-denoising-steps-help`},{key:`settings.live.diffusion_sampler`,en:`Diffusion sampler`,ko:`디퓨전 샘플러`,test_id:`settings-live-diffusion-sampler`},{key:`settings.live.diffusion_sampler.help`,en:`Default sampler for diffusion generation.`,ko:`디퓨전 생성의 기본 샘플러입니다.`,test_id:`settings-live-diffusion-sampler-help`},{key:`settings.live.diffusion_threshold`,en:`Diffusion threshold`,ko:`디퓨전 임계값`,test_id:`settings-live-diffusion-threshold`},{key:`settings.live.diffusion_threshold.help`,en:`Confidence threshold for diffusion threshold sampling.`,ko:`디퓨전 임계값 샘플링의 신뢰도 임계값입니다.`,test_id:`settings-live-diffusion-threshold-help`}],De=new Set(Ee.map(e=>e.key)),Oe=[...Te,...Ee],ke=[{key:`activity.intro`,en:`Operations and runtime observations. Browsing never loads a model.`,ko:`작업과 런타임 관측입니다. 조회는 모델을 로드하지 않습니다.`,test_id:`activity-intro`},{key:`activity.session`,en:`History belongs to this server session: up to 200 terminal operations for one hour. A restart clears it; reconnect reconciles the authoritative snapshot.`,ko:`기록은 서버 세션에 속합니다. 완료 작업 최대 200개를 한 시간 보관하며, 재시작 시 초기화하고 재연결 시 서버 상태와 대조합니다.`,test_id:`activity-session`},{key:`activity.export`,en:`Export sanitized diagnostics`,ko:`민감 정보 없는 진단 내보내기`,test_id:`activity-export`},{key:`activity.refresh`,en:`Refresh observations`,ko:`관측 새로고침`,test_id:`activity-refresh`},{key:`activity.stale`,en:`Observations are stale. Last values are not current measurements.`,ko:`관측이 최신 상태가 아닙니다. 마지막 값은 현재 측정값이 아닙니다.`,test_id:`activity-stale`},{key:`activity.updated`,en:`Last successful snapshot`,ko:`마지막 성공 스냅샷`,test_id:`activity-updated`},{key:`activity.pending`,en:`Waiting for the first snapshot`,ko:`첫 스냅샷을 기다리는 중`,test_id:`activity-pending`},{key:`activity.operations`,en:`Operations`,ko:`작업`,test_id:`activity-operations`},{key:`activity.operations.counts`,en:`Operations: {active} active · {failed} failed`,ko:`작업: {active} 진행 중 · {failed} 실패`,test_id:`activity-operations-counts`},{key:`activity.operations.empty.title`,en:`No operations in this session`,ko:`이 세션의 작업 없음`,test_id:`activity-operations-empty-title`},{key:`activity.operations.empty.body`,en:`Explicit downloads, loads, drains and removals will appear here.`,ko:`명시적인 다운로드, 로드, 드레인 및 삭제 작업이 표시됩니다.`,test_id:`activity-operations-empty-body`},{key:`activity.cancel`,en:`Request cancellation`,ko:`취소 요청`,test_id:`activity-cancel`},{key:`activity.cancelling`,en:`Cancellation requested; waiting for worker acknowledgement.`,ko:`취소 요청됨. 작업자 확인을 기다립니다.`,test_id:`activity-cancelling`},{key:`activity.failure`,en:`The action failed. Check its state and refresh before retrying from Models.`,ko:`작업에 실패했습니다. 상태를 확인하고 새로고침한 후 모델 화면에서 재시도하세요.`,test_id:`activity-failure`},{key:`activity.cancel_failed`,en:`Cancellation could not be confirmed. Refresh to reconcile the operation.`,ko:`취소를 확인하지 못했습니다. 새로고침하여 작업 상태를 대조하세요.`,test_id:`activity-cancel-failed`},{key:`activity.cancel_unsupported`,en:`Cancellation is unavailable for this operation; worker execution is not interrupted.`,ko:`이 작업은 취소할 수 없습니다. 실행 중인 작업자를 강제로 중단하지 않습니다.`,test_id:`activity-cancel-unsupported`},{key:`activity.runtime`,en:`Selected model runtime`,ko:`선택한 모델 런타임`,test_id:`activity-runtime`},{key:`activity.select`,en:`Select a model to observe`,ko:`관측할 모델 선택`,test_id:`activity-select`},{key:`activity.select_body`,en:`Use the model picker. Selection does not load the model.`,ko:`모델 선택기를 사용하세요. 선택만으로 모델을 로드하지 않습니다.`,test_id:`activity-select-body`},{key:`activity.unavailable`,en:`N/A — no authoritative sample is available.`,ko:`N/A — 확인된 관측값이 없습니다.`,test_id:`activity-unavailable`},{key:`activity.not_available`,en:`N/A`,ko:`N/A`,test_id:`activity-not-available`},{key:`activity.memory`,en:`Memory figures have separate scopes and may overlap. Do not add process, allocator, device and cache values together.`,ko:`메모리 수치는 서로 다른 범위이며 중복될 수 있습니다. 프로세스, 할당자, 장치 및 캐시 값을 합산하지 마세요.`,test_id:`activity-memory`},{key:`activity.timing`,en:`Server counters are not browser TTFT. Chat shows request-send to first reasoning/content delta separately; no rendered word counts are used as tokens.`,ko:`서버 카운터는 브라우저 TTFT가 아닙니다. 채팅은 요청 전송부터 첫 추론/내용 델타까지 별도로 표시하며, 단어 수를 토큰으로 사용하지 않습니다.`,test_id:`activity-timing`},{key:`activity.slots`,en:`Request slots`,ko:`요청 슬롯`,test_id:`activity-slots`},{key:`activity.slot`,en:`Slot`,ko:`슬롯`,test_id:`activity-slot`},{key:`activity.processing`,en:`Processing`,ko:`처리 중`,test_id:`activity-processing`},{key:`activity.idle`,en:`Idle`,ko:`유휴`,test_id:`activity-idle`},{key:`activity.context`,en:`Request context window`,ko:`요청 컨텍스트 한도`,test_id:`activity-context`},{key:`activity.pool`,en:`Shared pool context`,ko:`공유 풀 컨텍스트`,test_id:`activity-pool`},{key:`activity.parallel`,en:`Effective parallelism`,ko:`실제 병렬도`,test_id:`activity-parallel`},{key:`activity.context_line`,en:`{context}: {request} tokens · {pool}: {shared} tokens`,ko:`{context}: {request} 토큰 · {pool}: {shared} 토큰`,test_id:`activity-context-line`},{key:`activity.no_context`,en:`N/A — context denominator is unknown; no occupancy percentage is inferred.`,ko:`N/A — 컨텍스트 분모를 알 수 없어 점유율을 추정하지 않습니다.`,test_id:`activity-no-context`},{key:`activity.slot_progress`,en:`{used} / {total} tokens`,ko:`{used} / {total} 토큰`,test_id:`activity-slot-progress`},{key:`activity.slot_tokens`,en:`Accepted decode: {decoded} tokens · Cached prompt: {cached} tokens`,ko:`수락된 디코드: {decoded} 토큰 · 캐시된 프롬프트: {cached} 토큰`,test_id:`activity-slot-tokens`},{key:`activity.history.show`,en:`Show recent history`,ko:`최근 기록 보기`,test_id:`activity-history-show`},{key:`activity.history.hide`,en:`Hide recent history`,ko:`최근 기록 숨기기`,test_id:`activity-history-hide`},{key:`activity.history.note`,en:`Five-minute in-memory history, at most 150 two-second samples. Hidden tabs stop observation; gaps and counter resets are not interpolated.`,ko:`메모리 내 5분 기록이며 2초 간격 최대 150개입니다. 숨겨진 탭은 관측을 중지합니다. 공백과 카운터 초기화를 보간하지 않습니다.`,test_id:`activity-history-note`},{key:`activity.chart.label`,en:`Active requests: discrete observations over the last five minutes`,ko:`활성 요청: 최근 5분 동안의 개별 관측값`,test_id:`activity-chart-label`},{key:`activity.chart.title`,en:`Active requests (requests, selected model)`,ko:`활성 요청 (요청 수, 선택한 모델)`,test_id:`activity-chart-title`},{key:`activity.chart.point`,en:`{value} requests`,ko:`요청 {value}개`,test_id:`activity-chart-point`},{key:`activity.scope`,en:`Scope`,ko:`범위`,test_id:`activity-scope`},{key:`activity.observed`,en:`Observed at`,ko:`관측 시각`,test_id:`activity-observed`},{key:`activity.metric_details`,en:`All measurements and sources`,ko:`전체 측정값과 출처`,test_id:`activity-metric-details`},{key:`activity.unavailable_count`,en:`Unavailable measurements`,ko:`사용할 수 없는 측정값`,test_id:`activity-unavailable-count`},{key:`activity.unavailable_reason`,en:`Some sources are disabled or do not publish an authoritative sample. Expand details for individual reasons.`,ko:`일부 출처가 비활성화되었거나 확인된 관측값을 제공하지 않습니다. 상세 내용을 펼쳐 개별 사유를 확인하세요.`,test_id:`activity-unavailable-reason`},{key:`activity.no_primary`,en:`No primary request counters are available.`,ko:`사용 가능한 주요 요청 카운터가 없습니다.`,test_id:`activity-no-primary`},{key:`activity.bytes`,en:`bytes downloaded`,ko:`다운로드 바이트`,test_id:`activity-bytes`},{key:`activity.download_progress`,en:`{completed} / {total} bytes`,ko:`{completed} / {total} 바이트`,test_id:`activity-download-progress`},{key:`activity.details`,en:`Operation details`,ko:`작업 상세`,test_id:`activity-details`},{key:`activity.status`,en:`Operation status`,ko:`작업 상태`,test_id:`activity-status`},{key:`activity.unknown_time`,en:`Unknown`,ko:`알 수 없음`,test_id:`activity-unknown-time`},{key:`activity.metric.active_requests`,en:`Active requests`,ko:`활성 요청`,test_id:`activity-metric-active-requests`},{key:`activity.metric.queued_requests`,en:`Queued requests`,ko:`대기 요청`,test_id:`activity-metric-queued-requests`},{key:`activity.metric.completed_requests_total`,en:`Total completed requests`,ko:`누적 완료 요청`,test_id:`activity-metric-completed-requests-total`},{key:`activity.metric.completion_tokens_total`,en:`Total completion tokens`,ko:`누적 완료 토큰`,test_id:`activity-metric-completion-tokens-total`},{key:`activity.metric.gpu_utilization`,en:`GPU utilization`,ko:`GPU 사용률`,test_id:`activity-metric-gpu-utilization`},{key:`activity.metric.ttft`,en:`Time to first token`,ko:`첫 토큰 도달 시간`,test_id:`activity-metric-ttft`},{key:`activity.metric.decode_rate`,en:`Decode rate`,ko:`디코드 속도`,test_id:`activity-metric-decode-rate`},{key:`activity.metric.process_resident_bytes`,en:`Process resident memory`,ko:`프로세스 상주 메모리`,test_id:`activity-metric-process-resident-bytes`},{key:`activity.metric.allocator_active_bytes`,en:`Active allocator memory`,ko:`할당자 활성 메모리`,test_id:`activity-metric-allocator-active-bytes`},{key:`activity.metric.allocator_cache_bytes`,en:`Cached allocator memory`,ko:`할당자 캐시 메모리`,test_id:`activity-metric-allocator-cache-bytes`},{key:`activity.metric.allocator_peak_bytes`,en:`Peak allocator memory`,ko:`할당자 최대 메모리`,test_id:`activity-metric-allocator-peak-bytes`},{key:`activity.metric.device_total_bytes`,en:`Device memory`,ko:`장치 메모리`,test_id:`activity-metric-device-total-bytes`},{key:`activity.metric.model_weights_bytes`,en:`Model weights estimate`,ko:`모델 가중치 추정량`,test_id:`activity-metric-model-weights-bytes`},{key:`activity.metric.kv_cache_bytes`,en:`KV cache memory`,ko:`KV 캐시 메모리`,test_id:`activity-metric-kv-cache-bytes`},{key:`activity.metric.generation_time_ms_total`,en:`Total generation time`,ko:`누적 생성 시간`,test_id:`activity-metric-generation-time-ms-total`},{key:`activity.metric.decode_tokens_total`,en:`Accepted decode tokens`,ko:`수락된 디코드 토큰`,test_id:`activity-metric-decode-tokens-total`},{key:`activity.metric.decode_time_us_total`,en:`Total decode time`,ko:`누적 디코드 시간`,test_id:`activity-metric-decode-time-us-total`},{key:`activity.metric.prompt_cache_bytes`,en:`Prompt cache memory`,ko:`프롬프트 캐시 메모리`,test_id:`activity-metric-prompt-cache-bytes`},{key:`activity.metric.prompt_cache_entries`,en:`Prompt cache entries`,ko:`프롬프트 캐시 항목`,test_id:`activity-metric-prompt-cache-entries`}],Ae=[{key:`app.title`,en:`mlxcel WebUI`,ko:`mlxcel 웹 UI`,test_id:`app-title`},{key:`app.subtitle`,en:`Local model control plane`,ko:`로컬 모델 제어판`,test_id:`app-subtitle`},{key:`nav.models`,en:`Models`,ko:`모델`,test_id:`nav-models`},{key:`nav.chat`,en:`Chat`,ko:`대화`,test_id:`nav-chat`},{key:`nav.activity`,en:`Activity`,ko:`활동`,test_id:`nav-activity`},{key:`nav.settings`,en:`Settings`,ko:`설정`,test_id:`nav-settings`},{key:`nav.gallery`,en:`Gallery`,ko:`갤러리`,test_id:`nav-gallery`},{key:`nav.primary`,en:`Primary navigation`,ko:`주 내비게이션`,test_id:`nav-primary`},{key:`nav.home`,en:`mlxcel home`,ko:`mlxcel 홈`,test_id:`nav-home`},{key:`toolbar.command`,en:`Command`,ko:`명령`,test_id:`toolbar-command`},{key:`toolbar.help`,en:`Keyboard help`,ko:`키보드 도움말`,test_id:`toolbar-help`},{key:`toolbar.logout`,en:`Clear WebUI session`,ko:`WebUI 세션 지우기`,test_id:`toolbar-logout`},{key:`toolbar.menu`,en:`Open navigation`,ko:`내비게이션 열기`,test_id:`toolbar-menu`},{key:`toolbar.loaded.label`,en:`Loaded models`,ko:`로드된 모델`,test_id:`toolbar-loaded`},{key:`toolbar.loaded.none`,en:`No model loaded`,ko:`로드된 모델 없음`,test_id:`toolbar-loaded-none`},{key:`toolbar.loaded.unknown`,en:`Loaded models unknown`,ko:`로드된 모델 알 수 없음`,test_id:`toolbar-loaded-unknown`},{key:`toolbar.loaded.count`,en:`{count} loaded`,ko:`{count}개 로드됨`,test_id:`toolbar-loaded-count`},{key:`toolbar.loaded.more`,en:`+{count}`,ko:`+{count}`,test_id:`toolbar-loaded-more`},{key:`toolbar.loaded.more_label`,en:`+{count}, show all loaded models`,ko:`+{count}, 로드된 모델 모두 보기`,test_id:`toolbar-loaded-more-label`},{key:`connection.ready`,en:`Shell loaded; local API not connected`,ko:`셸 로드됨; 로컬 API 미연결`,test_id:`connection-ready`},{key:`connection.offline`,en:`Server connection is offline`,ko:`서버 연결이 오프라인입니다`,test_id:`connection-offline`},{key:`connection.prompt.title`,en:`Connect to the local WebUI API`,ko:`로컬 WebUI API에 연결하세요`,test_id:`connection-prompt-title`},{key:`connection.prompt.body`,en:`The shell is loaded, but catalog, chat and activity data wait for the authenticated local server connection.`,ko:`셸은 로드되었지만 카탈로그, 대화, 활동 데이터는 인증된 로컬 서버 연결을 기다립니다.`,test_id:`connection-prompt-body`},{key:`connection.prompt.detail`,en:`Start mlxcel-server with --webui, enter the terminal session key when prompted, then refresh this view.`,ko:`mlxcel-server를 --webui로 시작하고, 요청되면 터미널 세션 키를 입력한 뒤 이 화면을 새로고침하세요.`,test_id:`connection-prompt-detail`},{key:`connection.footer.connected`,en:`{mode} · {status} · v{version}`,ko:`{mode} · {status} · v{version}`,test_id:`connection-footer-connected`},{key:`connection.footer.details`,en:`Connection details`,ko:`연결 세부 정보`,test_id:`connection-footer-details`},{key:`connection.footer.instance`,en:`Server instance: {id}`,ko:`서버 인스턴스: {id}`,test_id:`connection-footer-instance`},{key:`connection.footer.sequence`,en:`Event sequence: {sequence}`,ko:`이벤트 순번: {sequence}`,test_id:`connection-footer-sequence`},{key:`connection.snapshot.pending`,en:`pending`,ko:`대기 중`,test_id:`connection-snapshot-pending`},{key:`connection.authenticated.title`,en:`Authenticated local API session`,ko:`인증된 로컬 API 세션`,test_id:`connection-authenticated-title`},{key:`connection.authenticated.body`,en:`This route is connected to the shared provider. Browsing does not load models or start inference; load, unload and chat actions remain explicit.`,ko:`이 경로는 공유 provider에 연결되어 있습니다. 탐색만으로 모델을 로드하거나 추론을 시작하지 않으며, 로드·언로드·대화 동작은 명시적으로 실행됩니다.`,test_id:`connection-authenticated-body`},{key:`connection.authenticated.detail`,en:`Backend {mode}; build {version}; state {status}; catalog {count}; operations {operations}; snapshot {sequence}.`,ko:`백엔드 {mode}; 빌드 {version}; 상태 {status}; 카탈로그 {count}; 작업 {operations}; 스냅샷 {sequence}.`,test_id:`connection-authenticated-detail`},{key:`connection.error.title`,en:`Provider connection needs attention`,ko:`Provider 연결 확인 필요`,test_id:`connection-error-title`},{key:`connection.error.stale`,en:`The server snapshot changed; retry to take a fresh catalog and operation snapshot before continuing.`,ko:`서버 스냅샷이 바뀌었습니다. 계속하기 전에 다시 시도해 새 카탈로그와 작업 스냅샷을 가져오세요.`,test_id:`connection-error-stale`},{key:`connection.error.forbidden`,en:`The authenticated session is not allowed to access this UI endpoint.`,ko:`인증된 세션이 이 UI 엔드포인트에 접근할 수 없습니다.`,test_id:`connection-error-forbidden`},{key:`connection.error.unauthorized`,en:`The server rejected the current session key. Sign in again with the latest terminal key.`,ko:`서버가 현재 세션 키를 거부했습니다. 터미널에 표시된 최신 키로 다시 로그인하세요.`,test_id:`connection-error-unauthorized`},{key:`connection.error.generic`,en:`Retry the shared provider snapshot before issuing any model control action.`,ko:`모델 제어 동작을 실행하기 전에 공유 provider 스냅샷을 다시 가져오세요.`,test_id:`connection-error-generic`},{key:`connection.status.idle`,en:`idle`,ko:`대기`,test_id:`connection-status-idle`},{key:`connection.status.bootstrapping`,en:`bootstrapping`,ko:`부트스트랩`,test_id:`connection-status-bootstrapping`},{key:`connection.status.ready`,en:`ready`,ko:`준비`,test_id:`connection-status-ready`},{key:`connection.status.streaming`,en:`streaming`,ko:`스트리밍`,test_id:`connection-status-streaming`},{key:`connection.status.polling`,en:`polling`,ko:`폴링`,test_id:`connection-status-polling`},{key:`connection.status.offline`,en:`offline`,ko:`오프라인`,test_id:`connection-status-offline`},{key:`connection.status.stale`,en:`stale`,ko:`낡음`,test_id:`connection-status-stale`},{key:`connection.status.unauthorized`,en:`unauthorized`,ko:`인증 실패`,test_id:`connection-status-unauthorized`},{key:`connection.status.forbidden`,en:`forbidden`,ko:`거부됨`,test_id:`connection-status-forbidden`},{key:`connection.status.schema_mismatch`,en:`schema mismatch`,ko:`스키마 불일치`,test_id:`connection-status-schema-mismatch`},{key:`connection.status.error`,en:`error`,ko:`오류`,test_id:`connection-status-error`},{key:`auth.login`,en:`WebUI session login`,ko:`WebUI 세션 로그인`,test_id:`auth-login`},{key:`models.title`,en:`Model library`,ko:`모델 라이브러리`,test_id:`models-title`},{key:`models.empty.title`,en:`No local models yet`,ko:`아직 로컬 모델이 없습니다`,test_id:`models-empty-title`},{key:`models.empty.body`,en:`Browse, download, and load models explicitly. The shell never autoloads a checkpoint.`,ko:`모델을 명시적으로 탐색, 다운로드, 로드하세요. 셸은 체크포인트를 자동 로드하지 않습니다.`,test_id:`models-empty-body`},{key:`models.long_name`,en:`Qwen3-Very-Long-Local-Checkpoint-Name-With-Mixed-English-and-한국어-모델-이름`,ko:`Qwen3-Very-Long-Local-Checkpoint-Name-With-Mixed-English-and-한국어-모델-이름`,test_id:`models-long-name`},{key:`models.unsupported.reason`,en:`Vision input is unavailable for this backend; chat remains text-only.`,ko:`이 백엔드에서는 비전 입력을 사용할 수 없어 대화는 텍스트 전용입니다.`,test_id:`models-unsupported-reason`},{key:`models.load`,en:`Load`,ko:`로드`,test_id:`models-load`},{key:`models.unload`,en:`Unload`,ko:`언로드`,test_id:`models-unload`},{key:`models.status.ready`,en:`Ready`,ko:`준비됨`,test_id:`models-status-ready`},{key:`models.status.unloaded`,en:`Unloaded`,ko:`언로드됨`,test_id:`models-status-unloaded`},{key:`models.status.failed`,en:`Failed`,ko:`실패`,test_id:`models-status-failed`},{key:`models.status.loading`,en:`Loading`,ko:`로드 중`,test_id:`models-status-loading`},{key:`models.status.draining`,en:`Draining`,ko:`drain 중`,test_id:`models-status-draining`},{key:`models.status.unloading`,en:`Unloading`,ko:`언로드 중`,test_id:`models-status-unloading`},{key:`models.delete.confirm.title`,en:`Delete model from cache?`,ko:`캐시에서 모델을 삭제할까요?`,test_id:`dialog-delete-model-title`},{key:`models.delete.confirm.body`,en:`Delete {model} from the managed cache. Loaded or non-cache models cannot be deleted.`,ko:`관리 캐시에서 {model} 모델을 삭제합니다. 로드 중이거나 캐시 모델이 아니면 삭제할 수 없습니다.`,test_id:`dialog-delete-model-body`},{key:`models.delete.confirm.token_label`,en:`Type DELETE to confirm`,ko:`확인하려면 DELETE를 입력하세요`,test_id:`dialog-delete-model-token`},{key:`models.unload.confirm.body`,en:`Unload {model} after active requests drain; browser tabs will keep their selected model.`,ko:`활성 요청이 비워진 뒤 {model} 모델을 언로드합니다. 브라우저 탭의 선택 모델은 유지됩니다.`,test_id:`dialog-unload-model-body`},{key:`chat.title`,en:`Chat`,ko:`대화`,test_id:`chat-title`},{key:`chat.placeholder`,en:`Type a message; IME composition is preserved.`,ko:`메시지를 입력하세요. IME 조합은 보존됩니다.`,test_id:`chat-placeholder`},{key:`chat.streaming.status`,en:`Streaming tokens from the selected ready model.`,ko:`선택한 준비 모델에서 토큰을 스트리밍 중입니다.`,test_id:`chat-streaming-status`},{key:`chat.reasoning`,en:`Reasoning`,ko:`추론`,test_id:`chat-reasoning`},{key:`chat.tool_call`,en:`Tool call preview`,ko:`도구 호출 미리보기`,test_id:`chat-tool-call`},{key:`downloads.cancel.confirm.body`,en:`Cancel this download only after the server acknowledges worker shutdown.`,ko:`서버가 작업자 중지를 확인한 뒤에만 이 다운로드를 취소합니다.`,test_id:`dialog-cancel-download-body`},{key:`activity.title`,en:`Activity`,ko:`활동`,test_id:`activity-title`},{key:`activity.progress.indeterminate`,en:`Downloaded {bytes}; total size is not known yet.`,ko:`{bytes} 다운로드됨; 전체 크기는 아직 알 수 없습니다.`,test_id:`activity-progress-indeterminate`},{key:`activity.sse_reset`,en:`Event stream reset; take a fresh snapshot before resuming updates.`,ko:`이벤트 스트림이 재설정되었습니다. 업데이트를 재개하기 전에 새 스냅샷을 가져오세요.`,test_id:`activity-sse-reset`},{key:`settings.title`,en:`Settings`,ko:`설정`,test_id:`settings-title`},{key:`settings.appearance`,en:`Appearance preferences`,ko:`화면 표시 설정`,test_id:`settings-appearance`},{key:`settings.theme`,en:`Theme`,ko:`테마`,test_id:`settings-theme`},{key:`settings.color_scheme`,en:`Color scheme`,ko:`색상 모드`,test_id:`settings-color-scheme`},{key:`settings.material`,en:`Material`,ko:`재질`,test_id:`settings-material`},{key:`settings.glass_intensity`,en:`Glass intensity`,ko:`글래스 강도`,test_id:`settings-glass-intensity`},{key:`settings.reduce_motion`,en:`Reduce motion`,ko:`동작 줄이기`,test_id:`settings-reduce-motion`},{key:`settings.reduce_transparency`,en:`Reduce transparency`,ko:`투명도 줄이기`,test_id:`settings-reduce-transparency`},{key:`settings.high_contrast`,en:`High contrast`,ko:`고대비`,test_id:`settings-high-contrast`},{key:`settings.locale`,en:`Language`,ko:`언어`,test_id:`settings-locale`},{key:`settings.partial_success`,en:`Some settings changed; fields controlled by CLI or the loaded worker were left unchanged.`,ko:`일부 설정만 변경되었습니다. CLI 또는 로드된 워커가 제어하는 필드는 변경되지 않았습니다.`,test_id:`settings-partial-success`},{key:`settings.browser_only`,en:`Browser appearance only`,ko:`브라우저 표시 설정 전용`,test_id:`settings-browser-only`},{key:`settings.browser_only.body`,en:`These preferences stay in this browser and do not claim server or worker settings changed.`,ko:`이 설정은 이 브라우저에만 남으며 서버나 워커 설정 변경을 의미하지 않습니다.`,test_id:`settings-browser-only-body`},{key:`settings.clear_history.confirm.body`,en:`Clear only the explicit local browser history store; API keys are never persisted there.`,ko:`명시적으로 켠 로컬 브라우저 기록 저장소만 지웁니다. API 키는 그곳에 저장하지 않습니다.`,test_id:`dialog-clear-history-body`},{key:`state.unauthorized.title`,en:`Authentication required`,ko:`인증이 필요합니다`,test_id:`state-unauthorized-title`},{key:`state.unauthorized.body`,en:`Enter the session key printed by the local server terminal.`,ko:`로컬 서버 터미널에 한 번 표시된 세션 키를 입력하세요.`,test_id:`state-unauthorized-body`},{key:`state.offline.title`,en:`Server offline`,ko:`서버 오프라인`,test_id:`state-offline-title`},{key:`state.offline.body`,en:`The shell is available, but model data waits for the local API.`,ko:`셸은 사용할 수 있지만 모델 데이터는 로컬 API를 기다립니다.`,test_id:`state-offline-body`},{key:`state.schema_mismatch.title`,en:`UI schema mismatch`,ko:`UI 스키마 불일치`,test_id:`state-schema-mismatch-title`},{key:`state.schema_mismatch.body`,en:`The shell loaded, but the server reports a different UI API schema version; refresh after updating the bundle or server.`,ko:`셸은 로드되었지만 서버가 다른 UI API 스키마 버전을 보고했습니다. 번들이나 서버를 업데이트한 뒤 새로고침하세요.`,test_id:`state-schema-mismatch-body`},{key:`command.title`,en:`Command palette`,ko:`명령 팔레트`,test_id:`command-title`},{key:`select.search`,en:`Search options`,ko:`옵션 검색`,test_id:`select-search`},{key:`select.no_options`,en:`No matching options.`,ko:`일치하는 옵션이 없습니다.`,test_id:`select-no-options`},{key:`command.search`,en:`Search commands and models`,ko:`명령과 모델 검색`,test_id:`command-search`},{key:`command.no_results`,en:`No commands or models match this search.`,ko:`검색과 일치하는 명령이나 모델이 없습니다.`,test_id:`command-no-results`},{key:`command.section.commands`,en:`Commands`,ko:`명령`,test_id:`command-section-commands`},{key:`command.section.models`,en:`Models`,ko:`모델`,test_id:`command-section-models`},{key:`help.title`,en:`Keyboard shortcuts`,ko:`키보드 단축키`,test_id:`help-title`},{key:`help.body`,en:`Shortcuts do not fire while you type in a field, inside an open dialog (Esc still closes it) or during IME composition. Cmd/Ctrl+N and Cmd/Ctrl+Enter also work in the chat message field. Brackets move navigation only while the sidebar has focus.`,ko:`입력란에 입력하는 중, 열린 다이얼로그 안(Esc로 닫기는 가능), IME 조합 중에는 단축키가 동작하지 않습니다. Cmd/Ctrl+N과 Cmd/Ctrl+Enter는 대화 메시지 입력란에서도 동작합니다. 대괄호는 사이드바에 포커스가 있을 때만 내비게이션을 이동합니다.`,test_id:`help-body`},{key:`help.shortcut.command`,en:`Open the command palette`,ko:`명령 팔레트 열기`,test_id:`help-shortcut-command`},{key:`help.shortcut.new_chat`,en:`Start a new conversation`,ko:`새 대화 시작`,test_id:`help-shortcut-new-chat`},{key:`help.shortcut.send`,en:`Send the message from the composer`,ko:`입력 중인 메시지 보내기`,test_id:`help-shortcut-send`},{key:`help.shortcut.escape`,en:`Close the open dialog or navigation`,ko:`열린 다이얼로그나 내비게이션 닫기`,test_id:`help-shortcut-escape`},{key:`help.shortcut.navigate`,en:`Move primary navigation while the sidebar has focus`,ko:`사이드바에 포커스가 있을 때 주 내비게이션 이동`,test_id:`help-shortcut-navigate`},{key:`help.shortcut.help`,en:`Show keyboard shortcuts`,ko:`키보드 단축키 보기`,test_id:`help-shortcut-help`},{key:`gallery.title`,en:`Design system gallery`,ko:`디자인 시스템 갤러리`,test_id:`gallery-title`},{key:`gallery.subtitle`,en:`Shared tokens, controls, states, and viewport fixtures for page implementers.`,ko:`페이지 구현자를 위한 공유 토큰, 컨트롤, 상태, 뷰포트 픽스처입니다.`,test_id:`gallery-subtitle`},{key:`gallery.long_cjk`,en:`Long English and 한국어 labels truncate with accessible full labels.`,ko:`긴 English 및 한국어 레이블은 접근 가능한 전체 레이블을 유지하며 줄임표 처리됩니다.`,test_id:`gallery-long-cjk`},{key:`common.unavailable`,en:`Unavailable until server adapters connect`,ko:`서버 어댑터 연결 전에는 사용할 수 없습니다`,test_id:`common-unavailable`},{key:`common.cancel`,en:`Cancel`,ko:`취소`,test_id:`common-cancel`},{key:`common.delete`,en:`Delete`,ko:`삭제`,test_id:`common-delete`},{key:`common.retry`,en:`Retry`,ko:`다시 시도`,test_id:`common-retry`},{key:`common.reload`,en:`Reload`,ko:`새로고침`,test_id:`common-reload`},{key:`common.enter_key`,en:`Enter key`,ko:`키 입력`,test_id:`common-enter-key`},{key:`common.add_model`,en:`Add model`,ko:`모델 추가`,test_id:`common-add-model`},{key:`common.send`,en:`Send`,ko:`보내기`,test_id:`common-send`},{key:`adapters.pending.title`,en:`Local API is not connected yet`,ko:`로컬 API가 아직 연결되지 않았습니다`,test_id:`adapters-pending-title`},{key:`adapters.pending.body`,en:`This route shows the production shell only. Catalog, lifecycle, and runtime data will come from the shared typed client in the integration wave.`,ko:`이 경로는 프로덕션 셸만 보여줍니다. 카탈로그, 수명주기, 런타임 데이터는 통합 웨이브의 공유 typed client에서 제공됩니다.`,test_id:`adapters-pending-body`},{key:`chat.pending.body`,en:`Chat controls stay disabled until an authenticated ready model is selected by the shared state provider.`,ko:`공유 상태 provider가 인증된 준비 모델을 선택하기 전까지 대화 컨트롤은 비활성화됩니다.`,test_id:`chat-pending-body`},{key:`activity.empty.title`,en:`No active operations`,ko:`활성 작업 없음`,test_id:`activity-empty-title`},{key:`activity.empty.body`,en:`Loads, downloads, drains and SSE resets appear here after the lifecycle adapter connects.`,ko:`수명주기 어댑터가 연결된 뒤 로드, 다운로드, drain, SSE reset이 여기에 표시됩니다.`,test_id:`activity-empty-body`},{key:`gallery.tab.sections`,en:`Gallery sections`,ko:`갤러리 섹션`,test_id:`gallery-tab-sections`},{key:`gallery.tab.controls`,en:`Controls`,ko:`컨트롤`,test_id:`gallery-tab-controls`},{key:`gallery.tab.states`,en:`States`,ko:`상태`,test_id:`gallery-tab-states`},{key:`gallery.tab.data`,en:`Data display`,ko:`데이터 표시`,test_id:`gallery-tab-data`},{key:`gallery.controls.title`,en:`Buttons and fields`,ko:`버튼과 필드`,test_id:`gallery-controls-title`},{key:`gallery.controls.primary`,en:`Primary`,ko:`기본`,test_id:`gallery-controls-primary`},{key:`gallery.controls.secondary`,en:`Secondary`,ko:`보조`,test_id:`gallery-controls-secondary`},{key:`gallery.controls.danger`,en:`Danger`,ko:`위험`,test_id:`gallery-controls-danger`},{key:`gallery.controls.busy`,en:`Busy`,ko:`진행 중`,test_id:`gallery-controls-busy`},{key:`gallery.field.repo`,en:`Repository ID`,ko:`저장소 ID`,test_id:`gallery-field-repo`},{key:`gallery.field.repo_error`,en:`Use owner/name without a URL.`,ko:`URL 없이 owner/name 형식을 사용하세요.`,test_id:`gallery-field-repo-error`},{key:`gallery.field.repo_hint`,en:`Sample only; downloads require the later lifecycle adapter.`,ko:`샘플 전용입니다. 다운로드는 후속 수명주기 어댑터가 필요합니다.`,test_id:`gallery-field-repo-hint`},{key:`gallery.progress.measured`,en:`Sample download progress`,ko:`예시 다운로드 진행률`,test_id:`gallery-progress-measured`},{key:`gallery.select.native`,en:`Shared select combobox`,ko:`공통 선택 콤보박스`,test_id:`gallery-select-native`},{key:`gallery.overlays.title`,en:`Overlays`,ko:`오버레이`,test_id:`gallery-overlays-title`},{key:`gallery.tooltip`,en:`Tooltips are descriptive only`,ko:`툴팁은 설명 전용입니다`,test_id:`gallery-tooltip`},{key:`gallery.dialog.open`,en:`Open dialog`,ko:`다이얼로그 열기`,test_id:`gallery-dialog-open`},{key:`gallery.states.load_failed`,en:`Load failed`,ko:`로드 실패`,test_id:`gallery-states-load-failed`},{key:`gallery.states.load_failed_body`,en:`The worker exited before becoming ready; retry after checking logs.`,ko:`워커가 준비되기 전에 종료되었습니다. 로그 확인 후 다시 시도하세요.`,test_id:`gallery-states-load-failed-body`},{key:`gallery.data.title`,en:`Model rows`,ko:`모델 행`,test_id:`gallery-data-title`},{key:`gallery.data.name`,en:`Name`,ko:`이름`,test_id:`gallery-data-name`},{key:`gallery.data.status`,en:`Status`,ko:`상태`,test_id:`gallery-data-status`},{key:`gallery.data.rate`,en:`Rate`,ko:`속도`,test_id:`gallery-data-rate`},{key:`gallery.data.inspector`,en:`Inspector`,ko:`인스펙터`,test_id:`gallery-data-inspector`},{key:`gallery.sample.reasoning_body`,en:`Reasoning text is visually separated from answer content.`,ko:`추론 텍스트는 답변 본문과 시각적으로 분리됩니다.`,test_id:`gallery-sample-reasoning-body`},{key:`gallery.sample.tool_preview`,en:`{ "tool": "display_only" }`,ko:`{ "tool": "표시_전용" }`,test_id:`gallery-sample-tool-preview`},{key:`common.close`,en:`Close`,ko:`닫기`,test_id:`common-close`},{key:`settings.theme.mlxcel`,en:`Standard`,ko:`표준`,test_id:`settings-theme-mlxcel`},{key:`settings.theme.glass`,en:`Glass`,ko:`글래스`,test_id:`settings-theme-glass`},{key:`settings.color_scheme.system`,en:`System`,ko:`시스템`,test_id:`settings-color-scheme-system`},{key:`settings.color_scheme.light`,en:`Light`,ko:`라이트`,test_id:`settings-color-scheme-light`},{key:`settings.color_scheme.dark`,en:`Dark`,ko:`다크`,test_id:`settings-color-scheme-dark`},{key:`settings.material.glass`,en:`Glass`,ko:`글래스`,test_id:`settings-material-glass`},{key:`settings.material.tinted`,en:`Tinted`,ko:`틴트`,test_id:`settings-material-tinted`},{key:`settings.material.opaque`,en:`Opaque`,ko:`불투명`,test_id:`settings-material-opaque`},{key:`settings.locale.en`,en:`English`,ko:`영어`,test_id:`settings-locale-en`},{key:`settings.locale.ko`,en:`Korean`,ko:`한국어`,test_id:`settings-locale-ko`},{key:`settings.high_contrast.system`,en:`Follow system`,ko:`시스템 따르기`,test_id:`settings-high-contrast-system`},{key:`settings.high_contrast.on`,en:`On`,ko:`켬`,test_id:`settings-high-contrast-on`},{key:`settings.high_contrast.off`,en:`Off`,ko:`끔`,test_id:`settings-high-contrast-off`},{key:`gallery.hover_focus`,en:`Hover or focus`,ko:`호버 또는 포커스`,test_id:`gallery-hover-focus`},{key:`gallery.download`,en:`Download`,ko:`다운로드`,test_id:`gallery-download`},{key:`gallery.lifecycle.samples`,en:`Lifecycle sample badges`,ko:`수명주기 샘플 배지`,test_id:`gallery-lifecycle-samples`},{key:`gallery.sample.caption`,en:`Sample model table`,ko:`샘플 모델 표`,test_id:`gallery-sample-caption`},{key:`gallery.sample.label`,en:`Sample fixture`,ko:`샘플 픽스처`,test_id:`gallery-sample-label`},{key:`login.token.label`,en:`Session key`,ko:`세션 키`,test_id:`login-token-label`},{key:`login.token.help`,en:`Use the key printed once by the local server. It stays in memory only.`,ko:`로컬 서버가 한 번 출력한 키를 사용하세요. 키는 메모리에만 유지됩니다.`,test_id:`login-token-help`},{key:`login.submit`,en:`Connect`,ko:`연결`,test_id:`login-submit`},{key:`login.logout`,en:`Clear key`,ko:`키 지우기`,test_id:`login-logout`},{key:`login.error.sample`,en:`Sample error: the key was not accepted by the local API.`,ko:`샘플 오류: 로컬 API가 키를 허용하지 않았습니다.`,test_id:`login-error-sample`},{key:`login.error.wrong_key`,en:`The session key was rejected. Copy the latest key printed by the local server terminal.`,ko:`세션 키가 거부되었습니다. 로컬 서버 터미널에 표시된 최신 키를 복사하세요.`,test_id:`login-error-wrong-key`},{key:`login.error.offline`,en:`Could not reach the local WebUI API. Check that mlxcel-server is still running with --webui.`,ko:`로컬 WebUI API에 연결할 수 없습니다. mlxcel-server가 --webui로 계속 실행 중인지 확인하세요.`,test_id:`login-error-offline`},{key:`login.error.forbidden`,en:`The key authenticated, but this UI endpoint is forbidden for the current server session.`,ko:`키 인증은 되었지만 현재 서버 세션에서 이 UI 엔드포인트가 금지되어 있습니다.`,test_id:`login-error-forbidden`},{key:`login.error.schema`,en:`The server response does not match this bundled UI schema. Reload after updating the server or bundle.`,ko:`서버 응답이 번들된 UI 스키마와 일치하지 않습니다. 서버나 번들을 업데이트한 뒤 새로고침하세요.`,test_id:`login-error-schema`},{key:`login.error.generic`,en:`The local API could not complete authentication. Retry with the latest terminal key.`,ko:`로컬 API 인증을 완료할 수 없습니다. 터미널에 표시된 최신 키로 다시 시도하세요.`,test_id:`login-error-generic`},{key:`gallery.delete_token`,en:`DELETE`,ko:`DELETE`,test_id:`gallery-delete-token`},{key:`models.next_profile.body`,en:`Pending browser profile; not applied yet. Explicit CLI/environment settings take precedence. Read effective values after loading.`,ko:`대기 중인 브라우저 프로필이며 아직 적용되지 않았습니다. 명시적 CLI/환경 설정이 우선합니다. 실제 값은 로드 후 확인하세요.`,test_id:`models-next-profile-body`},{key:`models.next_profile.edit`,en:`Edit profile`,ko:`프로필 편집`,test_id:`models-next-profile-edit`},{key:`format.unknown`,en:`unknown`,ko:`알 수 없음`,test_id:`format-unknown`},{key:`format.not_measured`,en:`not yet measured`,ko:`아직 측정되지 않음`,test_id:`format-not-measured`},...Ce,...we,...Oe,...ke],je=new Map(Ae.map(e=>[e.key,e]));function B(e,t,n={}){let r=je.get(t);return r?r[e].replace(/\{(\w+)\}/g,(e,t)=>Object.hasOwn(n,t)?n[t]:`{${t}}`):t}function V(e){return je.get(e)?.test_id??e.replaceAll(`.`,`-`)}function Me(e){let t=(0,_.useContext)(be),n=(0,_.useId)(),r=`${n}-hint`,i=`${n}-error`,a=[e.hint?r:null,e.error?i:null].filter(Boolean).join(` `)||void 0,o=e.disabled||e.busy,s=(0,x.jsxs)(x.Fragment,{children:[e.hint?(0,x.jsx)(`small`,{id:r,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:i,"data-tone":`error`,children:e.error}):null]});return t?(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:n,"data-disabled":o||void 0,children:[(0,x.jsx)(`span`,{children:e.label}),(0,x.jsx)(`select`,{id:n,value:e.value,disabled:o,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":a,"data-testid":e.testId,onChange:t=>e.onChange(t.currentTarget.value),children:e.options.map(e=>(0,x.jsx)(`option`,{value:e.value,disabled:e.disabled,children:e.description?`${e.label}. ${e.description}`:e.label},e.value))}),s]}):(0,x.jsxs)(`div`,{ref:t=>{let n=t?.querySelector(`.select__trigger`);n?.setAttribute(`role`,`combobox`),e.busy?n?.setAttribute(`aria-busy`,`true`):n?.removeAttribute(`aria-busy`)},onKeyDownCapture:e=>{e.nativeEvent.isComposing&&e.stopPropagation()},className:`ds-field ds-common-select`,"data-disabled":o||void 0,"aria-busy":e.busy||void 0,"data-testid":e.testId,children:[(0,x.jsx)(Se,{value:e.value,options:e.options,onChange:e.onChange,label:e.label,disabled:o,invalid:!!e.error,"aria-describedby":a,fullWidth:!0,noOptionsLabel:B(e.locale??`en`,`select.no_options`),searchPlaceholder:B(e.locale??`en`,`select.search`)}),s]})}var Ne={narrow:`400px`,medium:`520px`,wide:`900px`};function Pe({isOpen:e,onClose:t,title:n,subtitle:r,width:i=`medium`,children:a,footer:o,className:s=``,ariaLabelledBy:c,ariaDescribedBy:l,closeLabel:u=`Close`,preventDismiss:d=!1,onDismissAttempt:f}){let p=(0,_.useRef)(null),[m,h]=(0,_.useState)(!1),g=(0,_.useRef)(d);g.current=d;let v=(0,_.useRef)(f);v.current=f;let y=(0,_.useCallback)(()=>g.current?(h(!0),v.current?.(),!0):!1,[]),b=(0,_.useCallback)(e=>{e.animationName===`drawer-shake`&&h(!1)},[]),S=(0,_.useRef)(null),C=(0,_.useRef)(null),[w,T]=(0,_.useState)(!1),E=(0,_.useRef)(!1);(0,_.useEffect)(()=>{e&&!E.current?(E.current=!0,requestAnimationFrame(()=>{requestAnimationFrame(()=>{T(!0)})})):T(!!e)},[e]);let D=e&&w,O=i in Ne?Ne[i]:i,k=c||`drawer-title`,A=l||(r?`drawer-subtitle`:void 0);(0,_.useEffect)(()=>{if(!e)return;S.current=document.activeElement;let n=p.current;if(!n)return;requestAnimationFrame(()=>{C.current?.focus()});let r=n.querySelectorAll(`button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])`),i=r[0],a=r[r.length-1],o=e=>{if(e.key===`Escape`){if(y())return;t();return}e.key===`Tab`&&(e.shiftKey?document.activeElement===i&&(e.preventDefault(),a?.focus()):document.activeElement===a&&(e.preventDefault(),i?.focus()))};return n.addEventListener(`keydown`,o),()=>{n.removeEventListener(`keydown`,o),S.current&&S.current.focus()}},[e,t,y]);let j=(0,_.useCallback)(e=>{if(e.target===e.currentTarget){if(y())return;t()}},[t,y]);return(0,x.jsx)(`div`,{className:`drawer__backdrop ${D?`drawer__backdrop--open`:``}`,onClick:j,role:`presentation`,"aria-hidden":!e,inert:!e,children:(0,x.jsxs)(`aside`,{ref:p,className:`drawer ${D?`drawer--open`:``} ${m?`drawer--shaking`:``} ${s}`,style:{width:O,maxWidth:`100vw`},role:`dialog`,"aria-modal":`true`,"aria-labelledby":k,"aria-describedby":A,onAnimationEnd:b,children:[(0,x.jsxs)(`header`,{className:`drawer__header`,children:[(0,x.jsxs)(`div`,{className:`drawer__title-row`,children:[(0,x.jsx)(`h2`,{id:k,className:`drawer__title`,children:n}),(0,x.jsx)(`button`,{ref:C,type:`button`,className:`drawer__close-btn`,onClick:t,"aria-label":u,children:(0,x.jsxs)(`svg`,{width:`20`,height:`20`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`2`,strokeLinecap:`round`,strokeLinejoin:`round`,children:[(0,x.jsx)(`path`,{d:`M18 6L6 18`}),(0,x.jsx)(`path`,{d:`M6 6l12 12`})]})})]}),r&&(0,x.jsx)(`p`,{id:A,className:`drawer__subtitle`,children:r})]}),(0,x.jsx)(`div`,{className:`drawer__content`,children:a}),o&&(0,x.jsx)(`footer`,{className:`drawer__footer`,children:o})]})})}var Fe=120;function Ie({content:e,children:t,className:n,contentClassName:r,tooltipId:i,tabIndex:a=0,toggleable:o=!1}){let[s,c]=(0,_.useState)(!1),[l,u]=(0,_.useState)(null),d=(0,_.useRef)(null),f=(0,_.useRef)(null),p=(0,_.useRef)(!0),m=(0,_.useRef)(null),h=(0,_.useId)(),g=i??`tooltip-${h}`;(0,_.useEffect)(()=>(p.current=!0,()=>{p.current=!1,m.current!==null&&clearTimeout(m.current)}),[]),(0,_.useEffect)(()=>{let e=f.current;if(!e)return;let t=()=>{p.current&&c(!0)},n=()=>{p.current&&(c(!1),u(null))};return e.addEventListener(`focusin`,t),e.addEventListener(`focusout`,n),()=>{e.removeEventListener(`focusin`,t),e.removeEventListener(`focusout`,n)}},[]);let v=(0,_.useCallback)(()=>{let e=f.current,t=d.current;if(!p.current||!e||!t)return;let n=e.getBoundingClientRect(),r=t.getBoundingClientRect(),i=n.top,a=window.innerHeight-n.bottom,o=r.height,s=i<o+10&&a>i?`bottom`:`top`,c=n.left+n.width/2,l=r.width,m=c-l/2,h=window.innerWidth;m+l>h-16&&(m=h-l-16),m<16&&(m=16);let g=s===`top`?n.top-o-8:n.bottom+8;u({top:g,left:m,placement:s})},[]);(0,_.useLayoutEffect)(()=>{if(s)return v(),window.addEventListener(`scroll`,v,{capture:!0,passive:!0}),window.addEventListener(`resize`,v),()=>{window.removeEventListener(`scroll`,v,{capture:!0}),window.removeEventListener(`resize`,v)}},[s,v]);let y=(0,_.useCallback)(()=>{m.current!==null&&(clearTimeout(m.current),m.current=null),c(!0)},[]),b=(0,_.useCallback)(()=>{m.current!==null&&(clearTimeout(m.current),m.current=null),c(!1),u(null)},[]),S=(0,_.useCallback)(()=>{m.current!==null&&clearTimeout(m.current),m.current=setTimeout(()=>{m.current=null,p.current&&(c(!1),u(null))},Fe)},[]);(0,_.useEffect)(()=>{if(!s)return;let e=e=>{e.key===`Escape`&&b()};return document.addEventListener(`keydown`,e),()=>{document.removeEventListener(`keydown`,e)}},[s,b]);let C=(0,_.useCallback)(e=>{(e.key===`Enter`||e.key===` `)&&(e.preventDefault(),s?b():y())},[s,b,y]),w=s&&(0,x.jsx)(`div`,{ref:d,id:g,className:`tooltip__content tooltip__content--${l?.placement??`top`} ${r??``}`,style:l?{top:`${String(l.top)}px`,left:`${String(l.left)}px`}:{visibility:`hidden`},role:`tooltip`,children:typeof e==`string`?(0,x.jsx)(`p`,{className:`tooltip__text`,children:e}):e});return(0,x.jsxs)(`div`,{ref:f,className:`tooltip__wrapper ${n??``}`,onMouseEnter:y,onMouseLeave:S,onKeyDown:o?C:void 0,tabIndex:a,role:o?`button`:void 0,"aria-expanded":o?s:void 0,"aria-describedby":s?g:void 0,children:[t,w&&(0,xe.createPortal)(w,document.body)]})}var Le=`min(320px, calc(100vw - 32px))`;function Re(e){let t=(0,_.useId)(),n=(0,_.useRef)(e.onClose);n.current=e.onClose;let r=(0,_.useCallback)(()=>n.current(),[]),i=(0,_.useRef)(null);(0,_.useEffect)(()=>{if(!e.open)return;let t=0,n=0,a=()=>{let e=i.current?.querySelector(`.drawer`);e&&!e.contains(document.activeElement)&&(e.querySelector(`.drawer__close-btn`)?.focus(),!e.contains(document.activeElement)&&n++<10&&(t=requestAnimationFrame(a)))};t=requestAnimationFrame(a);let o=e=>{if(e.key!==`Escape`||e.isComposing||e.defaultPrevented)return;let t=i.current?.querySelector(`.drawer`);t&&e.target instanceof Node&&t.contains(e.target)||e.target instanceof Element&&e.target.closest(`dialog[open]`)||r()};return document.addEventListener(`keydown`,o),()=>{cancelAnimationFrame(t),document.removeEventListener(`keydown`,o)}},[e.open,r]);let a=[e.width===`medium`?`ds-drawer--medium`:``,e.side===`end`?`ds-drawer--end`:``].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{className:`ds-drawer-host`,ref:t=>{i.current=t,e.testId&&t?.querySelector(`.drawer`)?.setAttribute(`data-testid`,e.testId)},onKeyDownCapture:t=>{t.key===`Escape`&&(t.nativeEvent.isComposing||t.keyCode===229)&&t.stopPropagation(),e.open&&Be(i.current?.querySelector(`.drawer`)??null,t)},children:(0,x.jsx)(Pe,{isOpen:e.open,onClose:r,title:e.title,closeLabel:e.closeLabel,ariaLabelledBy:t,width:Le,className:a?`ds-drawer ${a}`:`ds-drawer`,children:e.children})})}var ze=`a[href], button, input, select, textarea, [tabindex]`;function Be(e,t){if(t.key!==`Tab`||t.altKey||t.ctrlKey||t.metaKey||!e||!(t.target instanceof Node)||!e.contains(t.target))return;let n=Array.from(e.querySelectorAll(ze)).filter(e=>e.tabIndex>=0&&!e.matches(`:disabled`)&&!e.closest(`[inert], [hidden]`)&&(typeof e.checkVisibility!=`function`||e.checkVisibility({visibilityProperty:!0}))),r=n.at(0),i=n.at(-1);r&&i&&document.activeElement===(t.shiftKey?r:i)&&(t.preventDefault(),(t.shiftKey?i:r).focus())}function Ve(e){let t=(0,_.useContext)(be),n=(0,_.useId)(),r=[e.children.props[`aria-describedby`],n].filter(Boolean).join(` `),i=_.cloneElement(e.children,{"aria-describedby":r});return t?(0,x.jsxs)(`span`,{className:`ds-tooltip-wrap`,children:[i,(0,x.jsx)(`span`,{className:`ds-tooltip`,role:`tooltip`,id:n,children:e.content})]}):(0,x.jsxs)(Ie,{content:e.content,tabIndex:-1,className:`ds-tooltip-trigger`,children:[i,(0,x.jsx)(`span`,{id:n,hidden:!0,children:e.content})]})}var He=(0,_.forwardRef)(({children:e,className:t=``,style:n,onClick:r,onKeyDown:i,clickable:a,variant:o=`default`,direction:s=`column`,state:c=`idle`,hoverable:l=!0,ariaLabel:u,role:d,ariaChecked:f,tabIndex:p,testId:m},h)=>{let g=a??!!r,_=e=>{i&&i(e),g&&r&&(e.key===`Enter`||e.key===` `)&&(e.preventDefault(),r())},v=[`base-card`,`base-card--${o}`,s===`row`&&`base-card--row`,l&&`base-card--hoverable`,g&&`base-card--clickable`,c!==`idle`&&`base-card--${c}`,t].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{ref:h,className:v,style:n,onClick:r,onKeyDown:_,role:d??(g?`button`:void 0),tabIndex:g?p??0:p,"aria-label":u,"aria-checked":f,"aria-disabled":c===`disabled`||void 0,"data-testid":m,children:e})});He.displayName=`BaseCard`;function Ue({title:e,description:t,actions:n,error:r,onErrorDismiss:i,className:a=``,errorDetail:o,onRetry:s,retryLabel:c=`Retry`,dismissErrorLabel:l=`Dismiss error`}){let u=[`page-header`,a].filter(Boolean).join(` `);return(0,x.jsxs)(`header`,{className:u,children:[(0,x.jsxs)(`div`,{className:`page-header__content`,children:[(0,x.jsxs)(`div`,{className:`page-header__text`,children:[(0,x.jsx)(`h1`,{className:`page-header__title`,children:e}),t&&(0,x.jsx)(`p`,{className:`page-header__description`,children:t})]}),n&&(0,x.jsx)(`div`,{className:`page-header__actions`,children:n})]}),r&&(0,x.jsxs)(`div`,{className:`page-header__error`,role:`alert`,children:[(0,x.jsxs)(`div`,{className:`page-header__error-body`,children:[(0,x.jsx)(`span`,{className:`page-header__error-text`,children:r}),o&&(0,x.jsx)(`span`,{className:`page-header__error-detail`,children:o})]}),(s||i)&&(0,x.jsxs)(`div`,{className:`page-header__error-actions`,children:[s&&(0,x.jsx)(w,{variant:`secondary`,size:`small`,className:`page-header__error-retry`,onClick:s,children:c}),i&&(0,x.jsx)(`button`,{type:`button`,className:`page-header__error-dismiss`,onClick:i,"aria-label":l,children:`×`})]})]})]})}function We({children:e,variant:t=`standard`,className:n=``,...r}){let i=[`page-layout`,`page-layout--${t}`,n].filter(Boolean).join(` `);return(0,x.jsx)(`div`,{className:i,...r,children:e})}var Ge=250;function Ke({active:e,children:t,className:n}){let r=(0,_.useRef)(null),i=(0,_.useRef)(null),a=(0,_.useRef)(null);return(0,_.useLayoutEffect)(()=>{let t=r.current,n=i.current;if(!t||!n||typeof ResizeObserver>`u`)return;if(!e)return a.current=window.setTimeout(()=>{t.style.height=``,a.current=null},Ge),()=>{a.current!==null&&(window.clearTimeout(a.current),a.current=null)};a.current!==null&&(window.clearTimeout(a.current),a.current=null);let o=()=>{t.style.height=`${n.offsetHeight.toFixed(0)}px`};o();let s=new ResizeObserver(o);return s.observe(n),()=>{s.disconnect()}},[e]),(0,x.jsx)(`div`,{ref:r,className:`smooth-height${e?` smooth-height--active`:``}${n?` ${n}`:``}`,children:(0,x.jsx)(`div`,{ref:i,className:`smooth-height__content`,children:t})})}function qe(e){return(0,x.jsx)(He,{className:`ds-card surface-card ${e.className??``}`.trim(),role:e.role??`article`,ariaLabel:e.ariaLabel,tabIndex:e.tabIndex,hoverable:!1,children:e.children})}function Je(e){return(0,x.jsx)(We,{variant:`wide`,className:`ds-page-layout ${e.className??``}`.trim(),children:e.children})}function Ye(e){return(0,x.jsx)(`div`,{className:`ds-page-header ${e.className??``}`.trim(),ref:t=>{let n=t?.querySelector(`.page-header__title`),r=t?.querySelector(`.page-header__description`),i=t?.querySelector(`.page-header__error`);n&&(n.tabIndex=-1,n.setAttribute(`data-dialog-focus-fallback`,``),e.titleTestId&&(n.dataset.testid=e.titleTestId)),r&&e.descriptionTestId&&(r.dataset.testid=e.descriptionTestId),i&&e.errorTestId&&(i.dataset.testid=e.errorTestId)},children:(0,x.jsx)(Ue,{title:e.title,description:e.description,actions:e.actions,error:e.error,errorDetail:e.errorDetail,onRetry:e.onRetry,retryLabel:e.retryLabel})})}var Xe=400;function Ze(e){let[t,n]=(0,_.useState)(e.animate),[r,i]=(0,_.useState)(!1),[a,o]=(0,_.useState)(0);return t!==e.animate&&(n(e.animate),e.animate||(i(!0),o(e=>e+1))),(0,_.useEffect)(()=>{if(!r)return;let e=window.setTimeout(()=>i(!1),Xe);return()=>window.clearTimeout(e)},[r,a]),(0,x.jsx)(Ke,{active:e.animate||r,className:`ds-smooth-height`,children:e.children})}var Qe=`(prefers-reduced-motion: reduce)`;function $e(){return typeof window>`u`||typeof window.matchMedia!=`function`?!1:window.matchMedia(Qe).matches}function et(){let[e,t]=(0,_.useState)($e);return(0,_.useEffect)(()=>{if(typeof window>`u`||typeof window.matchMedia!=`function`)return;let e=window.matchMedia(Qe),n=e=>{t(e.matches)};return e.addEventListener(`change`,n),()=>{e.removeEventListener(`change`,n)}},[]),e}var tt=800;function nt(e,t){let[n,r]=(0,_.useState)(t?0:e),i=(0,_.useRef)(t?0:e),a=(0,_.useRef)(null);return(0,_.useEffect)(()=>{if(!t){i.current=e,r(e);return}let n=i.current,o=performance.now(),s=t=>{let c=Math.min((t-o)/tt,1),l=1-(1-c)**3;r(Math.round(n+(e-n)*l)),c<1?a.current=requestAnimationFrame(s):i.current=e};return a.current=requestAnimationFrame(s),()=>{a.current!==null&&cancelAnimationFrame(a.current)}},[e,t]),t?n:e}var rt={up:`▲`,down:`▼`,flat:`•`};function it(e,t){return typeof e==`number`?t?t(e):e.toLocaleString():e}function at({label:e,labelNode:t,value:n,valueSuffix:r,hint:i,icon:a,tone:o=`default`,trend:s,onClick:c,loading:l=!1,ariaLabel:u,className:d=``,testId:f,format:p,animate:m=!1,sparkline:h,emphasis:g=`default`}){let _=et(),v=typeof n==`number`,y=m&&v&&!l&&!_,b=nt(v?n:0,y),S=it(v&&y?b:n,p),C=u??(l?e:`${e}: ${it(n,p)}${r??``}`),w=[`stat-card`,`corner-accent`,`stat-card--tone-${o}`,g==="default"?``:`stat-card--${g}`,d].filter(Boolean).join(` `);return(0,x.jsxs)(He,{className:w,onClick:c,role:c?void 0:`group`,ariaLabel:C,testId:f,children:[(0,x.jsxs)(`div`,{className:`stat-card__header`,children:[(0,x.jsx)(`span`,{className:`stat-card__label`,children:t??e}),a&&(0,x.jsx)(`span`,{className:`stat-card__icon`,"aria-hidden":`true`,children:a})]}),(0,x.jsxs)(`div`,{className:`stat-card__body`,children:[l?(0,x.jsx)(_e,{width:`60%`,height:`2rem`}):h?(0,x.jsxs)(`span`,{className:`stat-card__value-line`,children:[(0,x.jsxs)(`span`,{className:`stat-card__value-row`,children:[(0,x.jsx)(`span`,{className:`stat-card__value`,children:S}),r&&(0,x.jsx)(`span`,{className:`stat-card__value-suffix`,children:r})]}),(0,x.jsx)(`span`,{className:`stat-card__sparkline`,children:h})]}):(0,x.jsxs)(`span`,{className:`stat-card__value-row`,children:[(0,x.jsx)(`span`,{className:`stat-card__value`,children:S}),r&&(0,x.jsx)(`span`,{className:`stat-card__value-suffix`,children:r})]}),s&&!l&&(0,x.jsxs)(`span`,{className:`stat-card__trend stat-card__trend--${s.direction}`,"aria-label":s.ariaLabel??s.label,children:[(0,x.jsx)(`span`,{"aria-hidden":`true`,children:rt[s.direction]}),(0,x.jsx)(`span`,{children:s.label})]})]}),i!=null&&(0,x.jsx)(`div`,{className:`stat-card__hint`,children:l?(0,x.jsx)(_e,{width:`80%`,height:`0.85rem`}):i})]})}var ot=(0,_.memo)(at);function st(e){return(0,x.jsx)(ot,{label:e.label,value:e.value,hint:e.hint,className:`ds-stat-card ${e.className??``}`.trim(),testId:e.testId})}function ct(e){return(0,x.jsx)(T,{variant:e.tone===`accent`?`primary`:`default`,className:`ds-badge`,children:e.children})}function lt(e){let t=(0,_.useId)(),n=`${t}-label`,r=`${t}-hint`,i=`${t}-error`,a=[e.hint?r:null,e.error?i:null].filter(Boolean).join(` `)||void 0;return(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:t,"data-disabled":e.disabled||e.busy||void 0,children:[(0,x.jsx)(`span`,{id:n,children:e.label}),(0,x.jsx)(`input`,{id:t,"aria-labelledby":n,value:e.value,placeholder:e.placeholder,disabled:e.disabled||e.busy,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":a,"data-testid":e.testId,onChange:t=>e.onChange?.(t.currentTarget.value)}),e.hint?(0,x.jsx)(`small`,{id:r,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:i,"data-tone":`error`,children:e.error}):null]})}function ut(e){let t=(0,_.useRef)(null),n=(0,_.useRef)(null),r=(0,_.useId)(),i=e.labelledBy??r,a=(0,_.useRef)(!1),o=(0,_.useRef)(e.onClose);return o.current=e.onClose,(0,_.useEffect)(()=>{let r=t.current;r&&(e.open&&!r.open&&(n.current=document.activeElement instanceof HTMLElement?document.activeElement:null,r.showModal(),r.querySelector(`[data-autofocus], button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])`)?.focus()),!e.open&&r.open&&(a.current=!0,r.close()))},[e.open]),(0,_.useEffect)(()=>{let e=t.current;if(!e)return;let r=()=>{let t=n.current;dt(e,t),n.current=null},i=()=>{a.current?a.current=!1:o.current(),window.setTimeout(r,0)},s=t=>{if(t.key!==`Tab`)return;let n=Array.from(e.querySelectorAll(`button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])`)).filter(e=>e.offsetParent!==null||e===document.activeElement);if(n.length===0)return;let r=n[0],i=n[n.length-1];t.shiftKey&&document.activeElement===r?(t.preventDefault(),i.focus()):!t.shiftKey&&document.activeElement===i&&(t.preventDefault(),r.focus())};return e.addEventListener(`close`,i),e.addEventListener(`keydown`,s),()=>{e.removeEventListener(`close`,i),e.removeEventListener(`keydown`,s);let t=n.current;window.setTimeout(()=>dt(e,t),0)}},[]),(0,x.jsxs)(`dialog`,{className:`ds-dialog ${e.className??``}`.trim(),"data-position":e.position??`center`,ref:t,"aria-labelledby":i,"data-testid":e.testId,children:[(0,x.jsxs)(`header`,{children:[(0,x.jsx)(`h2`,{id:i,children:e.title}),(0,x.jsx)(F,{label:e.closeLabel??`Close`,icon:`close`,onClick:()=>{let e=n.current;a.current=!0,o.current(),t.current?.close(),window.setTimeout(()=>{let n=t.current;n&&dt(n,e)},0)},"data-testid":`dialog-close`})]}),(0,x.jsx)(`div`,{children:(0,x.jsx)(be.Provider,{value:!0,children:e.children})})]})}function dt(e,t){if(e.isConnected&&e.open)return;let n=document.activeElement;if(n&&n!==document.body&&n!==document.documentElement&&!e.contains(n)||document.querySelector(`dialog[open]`))return;let r=e=>!!e?.isConnected&&e!==document.body&&!e.matches(`:disabled, [aria-disabled="true"]`)&&!e.closest(`[inert], [hidden]`);if(r(t)){t.focus();return}let i=document.querySelector(`[data-dialog-focus-fallback]`)??document.querySelector(`main button:not(:disabled)`);r(i)&&i.focus()}function ft(e){return(0,x.jsx)(ut,{...e})}function pt(e){return(0,x.jsxs)(ft,{open:e.open,title:e.title,onClose:e.onClose,testId:e.testId,closeLabel:e.closeLabel,children:[(0,x.jsx)(`p`,{children:e.body}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(P,{"data-testid":`${e.testId}-cancel`,onClick:e.onClose,children:e.cancelLabel}),(0,x.jsx)(P,{tone:e.tone??`primary`,busy:e.busy,disabled:e.busy,"data-testid":`${e.testId}-confirm`,onClick:e.onConfirm,children:e.confirmLabel})]})]})}function mt(e){return(0,x.jsxs)(`aside`,{className:`ds-inspector`,"aria-label":e.title,children:[(0,x.jsx)(`h2`,{children:e.title}),e.children]})}function ht(e){return(0,x.jsxs)(`table`,{className:`ds-table`,children:[(0,x.jsx)(`caption`,{children:e.caption}),e.children]})}function gt(e){return(0,x.jsx)(`ul`,{className:`ds-list`,"aria-label":e.label,children:e.children})}function _t(e){let[t,n]=_.useState(``),r=_.useId();return _.useEffect(()=>()=>n(``),[]),(0,x.jsxs)(`form`,{className:`ds-login`,"data-testid":e.testId,autoComplete:`off`,onSubmit:r=>{if(r.preventDefault(),e.busy||t.trim().length===0)return;let i=t;n(``),e.onSubmit(i)},children:[(0,x.jsx)(`h2`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:e.tokenLabel}),(0,x.jsx)(`input`,{type:`password`,autoComplete:`off`,spellCheck:!1,autoCapitalize:`none`,autoCorrect:`off`,value:t,disabled:e.busy,"aria-invalid":e.error?`true`:void 0,"aria-describedby":r,onChange:e=>n(e.currentTarget.value)}),(0,x.jsx)(`small`,{id:r,"data-tone":e.error?`error`:`hint`,children:e.error??e.tokenHelp})]}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(P,{tone:`primary`,type:`submit`,busy:e.busy,disabled:t.trim().length===0,children:e.submitLabel}),e.onLogout&&e.logoutLabel?(0,x.jsx)(P,{type:`button`,onClick:()=>{n(``),e.onLogout?.()},children:e.logoutLabel}):null]})]})}function vt(e){return(0,x.jsx)(z,{tone:`error`,title:e.title,body:e.body,action:(0,x.jsxs)(P,{onClick:e.onRecover,children:[(0,x.jsx)(C,{name:`schema`}),e.actionLabel]})})}function yt(e){let t=(0,_.useId)(),n=`${t}-label`,r=`${t}-hint`,i=`${t}-error`,a=[e.hint?r:null,e.error?i:null].filter(Boolean).join(` `)||void 0;return(0,x.jsxs)(`label`,{className:`toggle`,"data-disabled":e.disabled||void 0,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:e.checked,disabled:e.disabled,"aria-labelledby":n,"aria-invalid":e.error?`true`:void 0,"aria-describedby":a,"data-testid":e.testId,onChange:t=>e.onChange(t.currentTarget.checked)}),(0,x.jsx)(`span`,{id:n,children:e.label}),e.hint?(0,x.jsx)(`small`,{id:r,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:i,"data-tone":`error`,children:e.error}):null]})}function bt(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Malformed settings response.`);return e}function xt(e){if(typeof e!=`string`||!/^[0-9a-f]{64}$/.test(e))throw Error(`Malformed settings fingerprint.`);return e}function St(e){let t=bt(e);if(!Array.isArray(t.schema)||t.schema.length>256)throw Error(`Malformed settings schema.`);let n=new Set;return{schema:t.schema.map(e=>{let t=bt(e);if(typeof t.name!=`string`||n.has(t.name)||typeof t.type!=`string`||![`bool`,`int`,`int_or_null`,`float`,`str`,`str_or_null`,`array`,`object`,`object_or_null`].includes(t.type)||typeof t.mutable!=`boolean`||typeof t.help!=`string`||!(t.allowed===null||Array.isArray(t.allowed)&&t.allowed.every(e=>typeof e==`string`))||t.reason!==void 0&&typeof t.reason!=`string`||!Object.hasOwn(t,`default`))throw Error(`Malformed setting specification.`);return n.add(t.name),t}),current:bt(t.current),fingerprint:xt(t.fingerprint)}}function Ct(e){let t=bt(e);if(!Array.isArray(t.rejected)||!t.rejected.every(e=>{let t=bt(e);return typeof t.name==`string`&&typeof t.reason==`string`}))throw Error(`Malformed settings rejection list.`);return{applied:bt(t.applied),rejected:t.rejected,current:bt(t.current),fingerprint:xt(t.fingerprint)}}function wt(e){let t=bt(e),n=bt(t.default_generation_settings),r=e=>typeof e==`number`&&Number.isSafeInteger(e)&&e>0?e:null;return{nCtx:r(n.n_ctx),kvCacheMode:typeof t.kv_cache_mode==`string`&&t.kv_cache_mode.length<=64&&/^[a-z0-9+_-]+$/.test(t.kv_cache_mode)?t.kv_cache_mode:null,totalSlots:r(t.total_slots),geometry:t.geometry===void 0?null:bt(t.geometry)}}function Tt(e,t){if(!e.mutable)throw Error(e.reason??`Restart required.`);let n=t;if(![`str`,`str_or_null`].includes(e.type)||e.type===`str_or_null`&&t===`null`)try{n=JSON.parse(t)}catch{throw Error(`Enter a valid ${e.type} value.`)}let r=e.type.endsWith(`_or_null`),i=e.type.replace(`_or_null`,``);if(!(r&&n===null)&&(i===`bool`?typeof n!=`boolean`:i===`int`?typeof n!=`number`||!Number.isSafeInteger(n):i===`float`?typeof n!=`number`||!Number.isFinite(n):i===`str`?typeof n!=`string`:i===`array`?!Array.isArray(n):typeof n!=`object`||!n||Array.isArray(n)))throw Error(`Expected ${e.type}.`);if(e.allowed!==null&&(typeof n!=`string`||!e.allowed.includes(n)))throw Error(`Allowed values: ${e.allowed.join(`, `)}.`);return n}function Et(e,t){return typeof t==`string`&&[`str`,`str_or_null`].includes(e.type)?t:JSON.stringify(t)??``}var Dt=7;function Ot(e){if(!Number.isFinite(e))return String(e);let t=Math.fround(e);for(let e=1;e<Dt;e+=1){let n=Number(t.toPrecision(e));if(Math.fround(n)===t)return String(n)}return String(Number(t.toPrecision(Dt)))}function kt(e,t){if(t==null)return``;let n=e.type.replace(`_or_null`,``);return n===`float`&&typeof t==`number`?Ot(t):n===`int`&&typeof t==`number`?String(t):Et(e,t)}function At(e){let t=bt(e);if(!Array.isArray(t.tokens)||!t.tokens.every(e=>typeof e==`number`&&Number.isInteger(e)))throw Error(`Malformed tokenization response.`);return t.tokens.length}var jt=1048576,Mt=class{decoder=new TextDecoder(`utf-8`);maxFrameBytes;onMessage;onDone;buffered=``;eventName=``;dataLines=[];id=null;retry=null;frameBytes=0;ended=!1;get done(){return this.ended}constructor(e){this.maxFrameBytes=e.maxFrameBytes??jt,this.onMessage=e.onMessage,this.onDone=e.onDone}push(e){if(!this.ended){if(this.frameBytes+=e.byteLength,this.frameBytes>this.maxFrameBytes)throw Error(`SSE frame exceeded the configured byte limit.`);this.buffered+=this.decoder.decode(e,{stream:!0}),this.drainLines(!1)}}close(){if(this.ended)return;let e=this.decoder.decode();e.length>0&&(this.buffered+=e),this.drainLines(!0),this.buffered.length>0&&(this.consumeLine(this.buffered),this.buffered=``),this.dispatch()}drainLines(e){for(;this.buffered.length>0;){let t=this.buffered.indexOf(`
`),n=this.buffered.indexOf(`\r`),r;if(r=t===-1?n:n===-1?t:Math.min(t,n),r===-1||this.buffered[r]===`\r`&&this.buffered[r+1]===void 0&&!e)break;let i=this.buffered.slice(0,r),a=this.buffered[r]===`\r`&&this.buffered[r+1]===`
`?r+2:r+1;this.buffered=this.buffered.slice(a),this.consumeLine(i)}e&&this.buffered.length===0&&this.dispatch()}consumeLine(e){if(e.length===0){this.dispatch();return}if(e.startsWith(`:`))return;let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?``:e.slice(t+1),i=r.startsWith(` `)?r.slice(1):r;n===`event`?this.eventName=i:n===`data`?this.dataLines.push(i):n===`id`?this.id=i:n===`retry`&&/^\d+$/.test(i)&&(this.retry=Number(i))}dispatch(){if(this.dataLines.length===0){this.eventName=``,this.retry=null,this.frameBytes=0;return}let e=this.dataLines.join(`
`),t=this.eventName.length===0?`message`:this.eventName;if(this.eventName=``,this.dataLines=[],this.frameBytes=0,e===`[DONE]`){this.ended=!0,this.onDone?.();return}this.onMessage({event:t,data:e,id:this.id,retry:this.retry}),this.retry=null}};function Nt(e){return Array.from(e).some(e=>{let t=e.charCodeAt(0);return t<=32||t>=127&&t<=159})}function Pt(e){if(/[\\?#]/.test(e)||Nt(e)||e.includes(`//`))throw Error(`WebUI API base must not contain a query, hash, backslash, whitespace, controls or empty path segments.`);for(let t of e.split(`/`)){let e;try{e=decodeURIComponent(t)}catch{throw Error(`WebUI API base contains an invalid path escape.`)}if(e===`.`||e===`..`)throw Error(`WebUI API base must not contain dot path segments.`);if(/[/\\%?#]/.test(e)||Nt(e))throw Error(`WebUI API base contains an ambiguous encoded path segment.`)}}function Ft(e){let t=e??Lt();if(t===``)return``;if(!t.startsWith(`/`)||t.startsWith(`//`))throw Error(`WebUI API base must be a same-origin absolute path.`);return Pt(t),new URL(t,`http://webui.invalid`).pathname.replace(/\/$/,``)}function It(e){return/^(.*)\/webui(?:\/(?:index\.html)?)?$/.exec(e)?.[1]??null}function Lt(){if(typeof document>`u`)return``;let e=document.querySelector(`meta[name="mlxcel-ui-api-base"]`)?.content;if(e!==void 0&&e.length>0)return Ft(e);let t=document.querySelector(`base`)?.getAttribute(`href`);if(t!=null&&t.length>0){let e=t.replace(/^[a-z][a-z\d+.-]*:\/\/[^/]*/i,``);Pt(e===`.`?``:e.replace(/^\.\//,``));let n=new URL(t,document.URL);if(n.origin!==window.location.origin||n.username||n.password||n.search||n.hash)throw Error(`WebUI document base must stay on the current origin without credentials, query or hash.`);return Ft(It(n.pathname)??n.pathname)}let n=window.location.pathname,r=It(n);return r===null?``:(Pt(n),Ft(r))}function Rt(e,t,n){if(!t.startsWith(`/ui-api/v1/`)&&t!==`/v1/chat/completions`&&t!==`/v1/responses`&&t!==`/settings`&&t!==`/props`&&t!==`/tokenize`)throw Error(`WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.`);let r=new URLSearchParams;for(let[e,t]of Object.entries(n??{}))t!=null&&r.set(e,String(t));return`${e}${t}${r.size===0?``:`?${r.toString()}`}`}function zt(e){if(e.length===0)throw Error(`Opaque path segment must not be empty.`);return encodeURIComponent(e)}var Bt=`{
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
`,H=class extends Error{path;constructor(e,t){super(`${e}: ${t}`),this.path=e,this.name=`ValidationError`}},Vt=Ut(Bt);function U(e,t,n=`$`){let r=tn(Vt,[`components`,`schemas`],`#/components/schemas`)[e];if(r===void 0)throw new H(n,`unknown schema ${e}`);Wt(r,t,n)}function Ht(e){return JSON.parse(e)}function Ut(e){return en(Ht(e),`$contract`)}function Wt(e,t,n){let r=en(e,`${n}#schema`);if(typeof r.$ref==`string`){Wt(Qt(r.$ref),t,n);return}if(r.anyOf!==void 0){if(!Array.isArray(r.anyOf)||!r.anyOf.some(e=>$t(e,t,n)))throw new H(n,`did not match any allowed schema`);return}if(r.oneOf!==void 0){if(!Array.isArray(r.oneOf))throw new H(n,`schema oneOf must be an array`);let e=r.oneOf.filter(e=>$t(e,t,n)).length;if(e!==1)throw new H(n,`matched ${e} oneOf schemas`);return}if(r.const!==void 0&&t!==r.const)throw new H(n,`expected constant ${String(r.const)}`);if(Array.isArray(r.enum)&&!r.enum.some(e=>e===t))throw new H(n,`unexpected enum value ${String(t)}`);let i=r.type;if(Array.isArray(i)){if(!i.some(e=>Zt(e,t)))throw new H(n,`expected one of ${i.join(`, `)}`)}else if(typeof i==`string`&&!Zt(i,t))throw new H(n,`expected ${i}`);typeof t==`string`&&Jt(r,t,n),typeof t==`number`&&Xt(r,t,n),Array.isArray(t)&&Kt(r,t,n),nn(t)&&Gt(r,t,n)}function Gt(e,t,n){let r=rn(e.required,`${n}#schema.required`,!1),i=e.properties===void 0?{}:en(e.properties,`${n}#schema.properties`);for(let e of r)if(!(e in t))throw new H(`${n}.${e}`,`missing required property`);if(e.maxProperties!==void 0&&Object.keys(t).length>an(e.maxProperties,`${n}#schema.maxProperties`))throw new H(n,`too many object properties`);for(let[r,a]of Object.entries(t)){let t=i[r];if(t!==void 0)Wt(t,a,`${n}.${r}`);else if(e.additionalProperties===!1)throw new H(`${n}.${r}`,`unexpected property`);else nn(e.additionalProperties)&&Wt(e.additionalProperties,a,`${n}.${r}`)}}function Kt(e,t,n){if(e.minItems!==void 0&&t.length<an(e.minItems,`${n}#schema.minItems`))throw new H(n,`array shorter than minimum`);if(e.maxItems!==void 0&&t.length>an(e.maxItems,`${n}#schema.maxItems`))throw new H(n,`array longer than maximum`);e.items!==void 0&&t.forEach((t,r)=>Wt(e.items,t,`${n}[${r}]`))}function qt(e){let t=0;for(let n=0;n<e.length;t++)n+=(e.codePointAt(n)??0)>65535?2:1;return t}function Jt(e,t,n){let r=qt(t);if(e.minLength!==void 0&&r<an(e.minLength,`${n}#schema.minLength`))throw new H(n,`string shorter than minimum`);if(e.maxLength!==void 0&&r>an(e.maxLength,`${n}#schema.maxLength`))throw new H(n,`string longer than maximum`);if(typeof e.pattern==`string`&&!new RegExp(e.pattern,`u`).test(t))throw new H(n,`string does not match pattern`);if(e.format===`date-time`&&!Yt(t))throw new H(n,`invalid RFC3339 date-time`)}function Yt(e){return/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(e)?!Number.isNaN(Date.parse(e)):!1}function Xt(e,t,n){if(!Number.isFinite(t))throw new H(n,`expected finite number`);if(e.type===`integer`&&!Number.isInteger(t))throw new H(n,`expected integer`);if(e.minimum!==void 0&&t<an(e.minimum,`${n}#schema.minimum`))throw new H(n,`number below minimum`);if(e.maximum!==void 0&&t>an(e.maximum,`${n}#schema.maximum`))throw new H(n,`number above maximum`)}function Zt(e,t){return e===`null`?t===null:e===`array`?Array.isArray(t):e===`object`?nn(t):e===`integer`?typeof t==`number`&&Number.isInteger(t):e===`number`?typeof t==`number`&&Number.isFinite(t):typeof t===e}function Qt(e){if(!e.startsWith(`#/`))throw new H(`$ref`,`unsupported external ref ${e}`);return e.slice(2).split(`/`).map(e=>e.replace(/~1/g,`/`).replace(/~0/g,`~`)).reduce((t,n)=>en(t,e)[n],Vt)}function $t(e,t,n){try{return Wt(e,t,n),!0}catch(e){if(e instanceof H)return!1;throw e}}function en(e,t){if(!nn(e))throw new H(t,`expected object`);return e}function tn(e,t,n){return t.reduce((e,t)=>en(e,n)[t],e)}function nn(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function rn(e,t,n){if(e===void 0&&!n)return[];if(!Array.isArray(e)||!e.every(e=>typeof e==`string`))throw new H(t,`expected string array`);return e}function an(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw new H(t,`expected numeric schema bound`);return e}function on(e,t,n){return U(e,t,n),t}function sn(e,t=`$`){return on(`ErrorEnvelope`,e,t)}function cn(e,t=`$`){return on(`BootstrapResponse`,e,t)}function ln(e,t=`$`){return on(`CatalogEntry`,e,t)}function un(e,t=`$`){return on(`CatalogListResponse`,e,t)}function dn(e,t=`$`){return on(`OperationsListResponse`,e,t)}function fn(e,t=`$`){return on(`Operation`,e,t)}function pn(e,t=`$`){return on(`OperationAccepted`,e,t)}function mn(e,t=`$`){return on(`RuntimeSnapshot`,e,t)}function hn(e,t=`$`){return on(`UiEvent`,e,t)}var gn=class extends Error{status;envelope;constructor(e,t,n){super(n??t?.error.message??`WebUI request failed with HTTP ${e}`),this.status=e,this.envelope=t,this.name=`WebUiHttpError`}},_n=2097152,vn=class{apiBase;fetchImpl;bearerToken=null;onUnauthorized;controllers=new Set;constructor(e={}){this.apiBase=Ft(e.apiBase),this.fetchImpl=e.fetchImpl??fetch.bind(globalThis),this.onUnauthorized=e.onUnauthorized}setBearerToken(e){this.bearerToken=e}abortAll(){for(let e of this.controllers)e.abort();this.controllers.clear()}async bootstrap(e){return this.request(`/ui-api/v1/bootstrap`,cn,{method:`GET`,signal:e})}async catalog(e={},t){return this.request(`/ui-api/v1/catalog`,un,{method:`GET`,query:{...e},signal:t})}async catalogEntry(e,t){return this.request(`/ui-api/v1/catalog/${zt(e)}`,ln,{method:`GET`,signal:t})}async refreshCatalog(e,t){return this.request(`/ui-api/v1/catalog/refresh`,pn,{method:`POST`,body:{idempotency_key:e},signal:t})}async modelAction(e,t){return this.request(`/ui-api/v1/model-actions`,pn,{method:`POST`,body:e,signal:t})}async download(e,t){return this.request(`/ui-api/v1/downloads`,pn,{method:`POST`,body:e,signal:t})}async removeModel(e,t){return this.request(`/ui-api/v1/model-removals`,pn,{method:`POST`,body:e,signal:t})}async operationsPage(e={},t){return this.request(`/ui-api/v1/operations`,dn,{method:`GET`,query:e,signal:t})}async operations(e){let t=[],n,r=null;for(;;){let i=await this.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null)r=i.server_instance_id;else if(i.server_instance_id!==r)throw Error(`Operations pagination crossed a server restart; refresh the authoritative snapshot.`);if(t.push(...i.items),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operation(e,t){return this.request(`/ui-api/v1/operations/${zt(e)}`,fn,{method:`GET`,signal:t})}async cancelOperation(e,t){return this.request(`/ui-api/v1/operations/${zt(e)}/cancel`,pn,{method:`POST`,body:{},signal:t})}async runtime(e,t){let n=await this.request(`/ui-api/v1/runtime`,mn,{method:`GET`,query:{model_id:e,autoload:!1},signal:t});if(n.model_id!==e)throw Error(`Runtime response model_id did not match the requested model.`);return n}async tokenCount(e,t,n){return this.request(`/tokenize`,At,{method:`POST`,query:{model:e,autoload:!1},body:{content:t,add_special:!1,parse_special:!0,with_pieces:!1},signal:n})}async settings(e,t){return this.request(`/settings`,St,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async patchSettings(e,t,n){return this.request(`/settings`,Ct,{method:`PATCH`,query:{model:e,autoload:!1},body:{op:`merge`,values:t},signal:n})}async modelProps(e,t){return this.request(`/props`,wt,{method:`GET`,query:{model:e,autoload:!1},signal:t})}async events(e,t,n){let r=n?.serverInstanceId!==void 0&&n.serverInstanceId!==null&&n.afterSequence!==null&&n.afterSequence!==void 0,i=r?{server_instance_id:n.serverInstanceId,after_sequence:n.afterSequence}:void 0,a=new Headers({Accept:`text/event-stream`});!r&&n?.lastEventId!==void 0&&n.lastEventId!==null&&a.set(`Last-Event-ID`,n.lastEventId),await this.sse(Rt(this.apiBase,`/ui-api/v1/events`,i),{method:`GET`,signal:t,headers:a},{onDone:e.onDone,onFrame:t=>{t.retry!==null&&e.onRetryAfter?.(t.retry),t.data.length>0&&e.onEvent(hn(Ht(t.data)))}})}async chatCompletions(e,t,n,r){await this.sse(Rt(this.apiBase,`/v1/chat/completions`,{autoload:!1}),{method:`POST`,body:xn(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async responses(e,t,n,r){await this.sse(Rt(this.apiBase,`/v1/responses`,{autoload:!1}),{method:`POST`,body:xn(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async request(e,t,n){let r=await this.fetchWithAuth(Rt(this.apiBase,e,n.query),n);try{return r.response.ok||await this.throwHttp(r.response,r.signal),t(Ht(await this.readBody(r.response,r.signal)))}finally{this.controllers.delete(r.controller),r.cleanup()}}async sse(e,t,n){let r=await this.fetchWithAuth(e,t);try{let e=r.response;if(e.ok||await this.throwHttp(e,r.signal),e.body===null)throw Error(`WebUI event stream response has no body.`);let t=new Mt({onDone:n.onDone,onMessage:n.onFrame}),i=e.body.getReader();try{for(;;){let e=await Sn(i,r.signal);if(e.done)break;t.push(e.value)}if(t.close(),!t.done&&r.signal.aborted!==!0)throw Error(`WebUI event stream ended before a DONE frame.`)}catch(e){throw await i.cancel().catch(()=>void 0),e}finally{i.releaseLock()}}finally{this.controllers.delete(r.controller),r.cleanup()}}async fetchWithAuth(e,t){let n=new AbortController,{signal:r,cleanup:i}=Cn(n,t.signal);this.controllers.add(n);let a=new Headers(t.headers);a.set(`Accept`,a.get(`Accept`)??`application/json`),this.bearerToken!==null&&a.set(`Authorization`,`Bearer ${this.bearerToken}`),t.body!==void 0&&a.set(`Content-Type`,`application/json`);try{return{response:await this.fetchImpl(e,{method:t.method,headers:a,body:t.body===void 0?void 0:JSON.stringify(t.body),signal:r,credentials:`same-origin`,cache:`no-store`,redirect:`error`}),controller:n,signal:r,cleanup:i}}catch(e){throw this.controllers.delete(n),i(),e}}async throwHttp(e,t){let n=this.bearerToken;e.status===401&&(this.bearerToken=null,this.abortAll(),this.onUnauthorized?.());let r=null;try{let n=await this.readBody(e,t);n.length>0&&(r=sn(Ht(n)))}catch{r=null}throw new gn(e.status,yn(r,n),bn(r?.error.message??`WebUI request failed with HTTP ${e.status}`,n))}async readBody(e,t){if(e.body===null)return``;let n=e.body.getReader(),r=[],i=0;try{for(;;){let e=await Sn(n,t);if(e.done)break;if(i+=e.value.byteLength,i>_n)throw Error(`WebUI JSON response exceeded the configured byte limit.`);r.push(e.value)}}catch(e){throw await n.cancel().catch(()=>void 0),e}finally{n.releaseLock()}let a=new Uint8Array(i),o=0;for(let e of r)a.set(e,o),o+=e.byteLength;return new TextDecoder().decode(a)}};function yn(e,t){return e===null?null:{...e,error:{...e.error,message:bn(e.error.message,t),field_errors:e.error.field_errors?.map(e=>({...e,message:bn(e.message,t)}))}}}function bn(e,t){let n=e.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`).replace(/token[=:]\s*\S+/gi,`token=[redacted]`);return t!==null&&t.length>0&&(n=n.split(t).join(`[redacted]`)),n}function xn(e,t){if(t.length===0)throw Error(`Inference model id is required for streaming inference.`);if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Streaming inference bodies must be JSON objects.`);return{...e,model:t,stream:!0}}async function Sn(e,t){if(t.aborted)throw await e.cancel().catch(()=>void 0),wn();let n=null;try{let r=await Promise.race([e.read(),new Promise((r,i)=>{n=()=>{e.cancel().catch(()=>void 0),i(wn())},t.addEventListener(`abort`,n,{once:!0})})]);if(t.aborted)throw wn();return r}finally{n!==null&&t.removeEventListener(`abort`,n)}}function Cn(e,t){if(t===void 0)return{signal:e.signal,cleanup:()=>void 0};let n=()=>e.abort(t.reason);return t.aborted?n():t.addEventListener(`abort`,n,{once:!0}),{signal:e.signal,cleanup:()=>t.removeEventListener(`abort`,n)}}function wn(){return typeof DOMException==`function`?new DOMException(`The WebUI request was aborted.`,`AbortError`):Error(`The WebUI request was aborted.`)}function Tn(e){return e instanceof gn?e.status===401?`wrong-key`:e.status===403?`forbidden`:`generic`:e instanceof H||e instanceof Error&&e.name===`ValidationError`?`schema`:e instanceof TypeError||e instanceof Error&&/fetch|network|offline|connection/i.test(e.message)?`offline`:`generic`}var En=new Set([`ready`,`loading`,`draining`,`unloading`]);function Dn(e){return En.has(e)}function On(e,t){let n=e=>e.lifecycle.state===`ready`?0:Dn(e.lifecycle.state)?1:2;return n(e)-n(t)||e.identity.display_name.localeCompare(t.identity.display_name)||e.identity.id.localeCompare(t.identity.id)}function kn(e){return e.catalog.filter(e=>Dn(e.lifecycle.state)).sort(On)}function An(e,t){return e.catalog.some(e=>e.identity.id===t)?t:null}function jn(e,t){return t.bootstrap===null?B(e,`connection.ready`):B(e,`connection.footer.connected`,{mode:t.bootstrap.server.mode,version:t.bootstrap.server.build.version,status:Nn(e,t.connection)})}function Mn(e,t){if(t.bootstrap===null)return null;let n=B(e,`format.unknown`),r=t.serverInstanceId??t.bootstrap.server.server_instance_id,i=t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot;return{instance:r||n,sequence:i===null?n:String(i)}}function Nn(e,t){return B(e,{idle:`connection.status.idle`,bootstrapping:`connection.status.bootstrapping`,ready:`connection.status.ready`,streaming:`connection.status.streaming`,polling:`connection.status.polling`,offline:`connection.status.offline`,stale:`connection.status.stale`,unauthorized:`connection.status.unauthorized`,forbidden:`connection.status.forbidden`,"schema-mismatch":`connection.status.schema_mismatch`,error:`connection.status.error`}[t])}function Pn(e){return e.snapshot.auth.status===`authenticated`?e.snapshot.connection===`schema-mismatch`?(0,x.jsx)(Ln,{...e}):(0,x.jsx)(In,{...e}):(0,x.jsx)(Fn,{...e})}function Fn(e){return e.authFailure===`schema`?(0,x.jsx)(Ln,{...e}):(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsx)(Ye,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:B(e.locale,`connection.prompt.body`),descriptionTestId:V(`connection.prompt.body`)}),(0,x.jsx)(_t,{title:B(e.locale,`state.unauthorized.title`),body:B(e.locale,`state.unauthorized.body`),tokenLabel:B(e.locale,`login.token.label`),tokenHelp:B(e.locale,`login.token.help`),submitLabel:B(e.locale,`login.submit`),logoutLabel:B(e.locale,`login.logout`),error:e.authFailure===null?void 0:Bn(e.locale,e.authFailure),busy:e.snapshot.auth.status===`authenticating`,onSubmit:e.onLogin,onLogout:e.onLogout,testId:V(`auth.login`)})]})}function In(e){let t=e.snapshot.connection===`offline`||e.snapshot.connection===`stale`||e.snapshot.connection===`error`||e.snapshot.connection===`unauthorized`||e.snapshot.connection===`forbidden`;return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsx)(Ye,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId,description:B(e.locale,`connection.authenticated.body`),descriptionTestId:V(`connection.authenticated.body`)}),(0,x.jsx)(z,{tone:`info`,title:B(e.locale,`connection.authenticated.title`),body:Rn(e.locale,e.snapshot),testId:V(`connection.authenticated.detail`)}),t?(0,x.jsx)(z,{title:B(e.locale,`connection.error.title`),body:zn(e.locale,e.snapshot.connection),action:(0,x.jsx)(P,{onClick:e.onRetry,children:B(e.locale,`common.retry`)}),testId:V(`connection.error.title`)}):null]})}function Ln(e){return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsx)(Ye,{className:`connection-prompt`,title:e.title,titleTestId:e.titleTestId}),(0,x.jsx)(vt,{title:B(e.locale,`state.schema_mismatch.title`),body:B(e.locale,`state.schema_mismatch.body`),actionLabel:B(e.locale,`common.reload`),onRecover:e.onRecoverSchema})]})}function Rn(e,t){let n=t.bootstrap;return n===null?B(e,`connection.prompt.detail`):B(e,`connection.authenticated.detail`,{mode:n.server.mode,version:n.server.build.version,status:Nn(e,t.connection),count:t.catalogSequence===null?B(e,`connection.snapshot.pending`):String(t.catalog.length),operations:t.resourceFences.operationsSnapshot===null?B(e,`connection.snapshot.pending`):String(t.operations.size),sequence:Hn(e,t)})}function zn(e,t){return t===`offline`?B(e,`state.offline.body`):t===`stale`?B(e,`connection.error.stale`):t===`forbidden`?B(e,`connection.error.forbidden`):t===`unauthorized`?B(e,`connection.error.unauthorized`):B(e,`connection.error.generic`)}function Bn(e,t){return t===`wrong-key`?B(e,`login.error.wrong_key`):t===`offline`?B(e,`login.error.offline`):t===`forbidden`?B(e,`login.error.forbidden`):t===`schema`?B(e,`login.error.schema`):B(e,`login.error.generic`)}function Vn(e,t){return B(e,{unloaded:`models.status.unloaded`,loading:`models.status.loading`,ready:`models.status.ready`,draining:`models.status.draining`,unloading:`models.status.unloading`,failed:`models.status.failed`}[t])}function Hn(e,t){return String(t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot??B(e,`connection.snapshot.pending`))}var Un=[{id:`models`,key:`nav.models`,testId:`command-route-models`},{id:`chat`,key:`nav.chat`,testId:`command-route-chat`},{id:`activity`,key:`nav.activity`,testId:`command-route-activity`},{id:`settings`,key:`nav.settings`,testId:`command-route-settings`},{id:`new-chat`,key:`chat.new_conversation`,testId:`command-new-chat`}];function Wn(e,t,n=20){let r=t.trim().toLowerCase();return r===``?[]:e.filter(e=>e.identity.display_name.toLowerCase().includes(r)||e.identity.id.toLowerCase().includes(r)).sort(On).slice(0,Math.max(0,n))}function Gn(e){let[t,n]=(0,_.useState)(``),[r,i]=(0,_.useState)(e.open),a=(0,_.useId)(),o=(0,_.useId)();e.open!==r&&(i(e.open),n(``));let s=t.trim().toLowerCase(),c=Un.filter(t=>s===``||t.id.includes(s)||B(e.locale,t.key).toLowerCase().includes(s)),l=s===``?e.loaded:Wn(e.catalog,s),u=t=>{t.id===`new-chat`?e.onNewChat():e.onNavigate(t.id)};return(0,x.jsxs)(ft,{open:e.open,title:B(e.locale,`command.title`),onClose:e.onClose,testId:`command-dialog`,closeLabel:B(e.locale,`common.close`),children:[(0,x.jsx)(lt,{label:B(e.locale,`command.search`),value:t,onChange:n,testId:V(`command.search`)}),c.length===0&&l.length===0?(0,x.jsx)(`p`,{"data-testid":V(`command.no_results`),children:B(e.locale,`command.no_results`)}):null,c.length>0?(0,x.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":a,children:[(0,x.jsx)(`h3`,{id:a,children:B(e.locale,`command.section.commands`)}),(0,x.jsx)(`div`,{className:`command-list`,children:c.map(t=>(0,x.jsx)(P,{"data-testid":t.testId,onClick:()=>u(t),children:B(e.locale,t.key)},t.id))})]}):null,l.length>0?(0,x.jsxs)(`div`,{className:`command-section`,role:`group`,"aria-labelledby":o,children:[(0,x.jsx)(`h3`,{id:o,children:B(e.locale,`command.section.models`)}),(0,x.jsx)(`div`,{className:`command-list`,children:l.map(t=>(0,x.jsx)(P,{className:`command-model`,title:t.identity.display_name,"data-testid":`command-model`,onClick:()=>e.onOpenModel(t.identity.id),children:`${t.identity.display_name} · ${Vn(e.locale,t.lifecycle.state)}`},t.identity.id))})]}):null]})}var Kn=[{id:`models`,key:`nav.models`,icon:`models`},{id:`chat`,key:`nav.chat`,icon:`chat`},{id:`activity`,key:`nav.activity`,icon:`activity`},{id:`settings`,key:`nav.settings`,icon:`settings`}],qn=[{id:`command`,keys:[`⌘/Ctrl`,`K`],separator:`+`,key:`help.shortcut.command`},{id:`new-chat`,keys:[`⌘/Ctrl`,`N`],separator:`+`,key:`help.shortcut.new_chat`},{id:`send`,keys:[`⌘/Ctrl`,`Enter`],separator:`+`,key:`help.shortcut.send`},{id:`escape`,keys:[`Esc`],separator:`+`,key:`help.shortcut.escape`},{id:`navigate`,keys:[`[`,`]`],separator:`/`,key:`help.shortcut.navigate`},{id:`help`,keys:[`?`],separator:`+`,key:`help.shortcut.help`}];function Jn(e){let t=(0,_.useRef)(null),n=(0,_.useRef)(e.onCommand),r=(0,_.useRef)(e.onHelp),i=(0,_.useRef)(e.onNewChat),[a,o]=(0,_.useState)(!1);(0,_.useEffect)(()=>{n.current=e.onCommand,r.current=e.onHelp,i.current=e.onNewChat},[e.onCommand,e.onHelp,e.onNewChat]),(0,_.useEffect)(()=>{let e=e=>{let t=e.target,a=t instanceof HTMLElement&&(t.isContentEditable||!!t.closest(`[role="combobox"], [role="listbox"]`)||[`INPUT`,`TEXTAREA`,`SELECT`].includes(t.tagName)),s=t instanceof HTMLElement&&!!t.closest(`dialog[open], [role="dialog"][aria-modal="true"]`);if(e.isComposing||e.altKey||a||s)return;let c=e.metaKey||e.ctrlKey;if(c&&e.key.toLowerCase()===`k`&&(e.preventDefault(),o(!1),n.current()),c&&!e.shiftKey&&e.key.toLowerCase()===`n`){if(e.preventDefault(),e.repeat)return;o(!1),i.current()}(e.key===`?`||e.key===`/`&&e.shiftKey)&&(e.preventDefault(),o(!1),r.current())};return window.addEventListener(`keydown`,e),()=>window.removeEventListener(`keydown`,e)},[]),(0,_.useEffect)(()=>{if(!a||typeof window.matchMedia!=`function`)return;let e=window.matchMedia(`(max-width: 960px)`),n=()=>{e.matches||(o(!1),requestAnimationFrame(()=>{let e=document.activeElement;if(e&&e!==document.body&&!e.closest(`[aria-modal="true"]`))return;let n=t.current;(n?.querySelector(`a[aria-current="page"]`)??n?.querySelector(`a`))?.focus()}))};return e.addEventListener(`change`,n),()=>e.removeEventListener(`change`,n)},[a]);let s=n=>{let r=Kn[(Kn.findIndex(t=>t.id===e.route)+n+Kn.length)%Kn.length];e.onRouteChange(r.id),requestAnimationFrame(()=>t.current?.querySelector(`a[href="#${r.id}"]`)?.focus())},c=e=>{(e.key===`[`||e.key===`]`)&&(e.altKey||e.metaKey||e.ctrlKey||e.nativeEvent.isComposing||(e.preventDefault(),s(e.key===`]`?1:-1)))},l=t=>{e.onRouteChange(t),o(!1)};return(0,x.jsxs)(`div`,{className:`app-shell`,children:[(0,x.jsx)(Zn,{locale:e.locale,route:e.route,onRouteChange:l,onKeyDown:c,ref:t,className:`app-sidebar desktop-sidebar material-glass`,connection:e.connection}),(0,x.jsx)(Re,{open:a,title:B(e.locale,`nav.primary`),onClose:()=>o(!1),closeLabel:B(e.locale,`common.close`),testId:`mobile-nav-sheet`,children:(0,x.jsx)(Zn,{locale:e.locale,route:e.route,onRouteChange:l,onKeyDown:c,className:`app-sidebar sheet-sidebar`,connection:e.connection})}),(0,x.jsxs)(`main`,{className:`app-main`,"aria-labelledby":`app-title`,inert:a,children:[(0,x.jsxs)(`header`,{className:`app-toolbar material-glass`,children:[(0,x.jsx)(F,{className:`mobile-menu-button`,label:B(e.locale,`toolbar.menu`),icon:`menu`,onClick:()=>o(!0),"data-testid":V(`toolbar.menu`)}),(0,x.jsxs)(`div`,{className:`toolbar-title`,children:[(0,x.jsx)(`p`,{id:`app-title`,title:B(e.locale,`app.title`),"data-testid":V(`app.title`),children:B(e.locale,`app.title`)}),(0,x.jsx)(`span`,{title:B(e.locale,`app.subtitle`),"data-testid":V(`app.subtitle`),children:B(e.locale,`app.subtitle`)})]}),(0,x.jsx)(Yn,{locale:e.locale,models:e.loadedModels,onOpenModel:e.onOpenModel,onMore:e.onCommand}),(0,x.jsxs)(`div`,{className:`toolbar-actions`,children:[e.sessionAction,(0,x.jsx)(F,{label:B(e.locale,`toolbar.command`),icon:`command`,onClick:e.onCommand,"data-testid":V(`toolbar.command`)}),(0,x.jsx)(F,{label:B(e.locale,`toolbar.help`),icon:`help`,onClick:e.onHelp,"data-testid":V(`toolbar.help`)})]})]}),(0,x.jsxs)(`div`,{className:`app-content-grid ${e.inspector?`has-inspector`:``}`.trim(),children:[(0,x.jsx)(Je,{className:`app-content`,children:e.children}),e.inspector]})]})]})}function Yn(e){let t=e.models??[],n=t.slice(0,3),r=t.length-n.length,i=e.models===null?`toolbar.loaded.unknown`:t.length===0?`toolbar.loaded.none`:null;return(0,x.jsx)(`div`,{className:`toolbar-loaded`,role:`group`,"aria-label":B(e.locale,`toolbar.loaded.label`),"data-testid":V(`toolbar.loaded.label`),onFocus:Xn,children:i===null?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":V(`toolbar.loaded.count`),children:B(e.locale,`toolbar.loaded.count`,{count:String(t.length)})}),n.map(t=>(0,x.jsxs)(P,{tone:`ghost`,className:`toolbar-chip`,title:t.name,"data-testid":`toolbar-loaded-chip`,onClick:()=>e.onOpenModel(t.id),children:[(0,x.jsx)(`span`,{className:`truncate`,children:t.name}),(0,x.jsx)(ue,{state:t.state,children:t.stateLabel})]},t.id)),r>0?(0,x.jsx)(P,{tone:`ghost`,className:`toolbar-chip toolbar-chip-more`,"aria-label":B(e.locale,`toolbar.loaded.more_label`,{count:String(r)}),"data-testid":V(`toolbar.loaded.more`),onClick:e.onMore,children:B(e.locale,`toolbar.loaded.more`,{count:String(r)})}):null]}):(0,x.jsx)(`span`,{className:`toolbar-loaded-count`,"data-testid":V(i),children:B(e.locale,i)})})}function Xn(e){let t=e.target;t instanceof HTMLElement&&typeof t.scrollIntoView==`function`&&t.matches(`:focus-visible`)&&t.scrollIntoView({block:`nearest`,inline:`nearest`})}var Zn=_.forwardRef((e,t)=>(0,x.jsxs)(`aside`,{className:e.className,"aria-label":B(e.locale,`nav.primary`),onKeyDown:e.onKeyDown,ref:t,tabIndex:-1,children:[(0,x.jsxs)(`a`,{className:`brand-mark`,href:`#models`,"aria-label":B(e.locale,`nav.home`),onClick:t=>{t.preventDefault(),e.onRouteChange(`models`)},children:[(0,x.jsx)(`span`,{className:`brand-symbol`,"aria-hidden":`true`,children:`mx`}),(0,x.jsx)(`span`,{className:`brand-name`,children:`mlxcel`})]}),(0,x.jsx)(`nav`,{className:`app-nav`,children:Kn.map(t=>(0,x.jsxs)(`a`,{href:`#${t.id}`,"aria-label":B(e.locale,t.key),"aria-current":e.route===t.id?`page`:void 0,"data-testid":V(t.key),onClick:n=>{n.preventDefault(),e.onRouteChange(t.id)},children:[(0,x.jsx)(C,{name:t.icon}),(0,x.jsx)(`span`,{className:`app-nav-label`,children:B(e.locale,t.key)})]},t.id))}),(0,x.jsxs)(`footer`,{children:[(0,x.jsxs)(`div`,{className:`connection-status`,children:[(0,x.jsx)(`span`,{className:`connection-dot`,"data-state":e.connection.state,"aria-hidden":`true`}),(0,x.jsx)(`span`,{"data-testid":V(`connection.ready`),children:e.connection.label})]}),e.connection.details?(0,x.jsxs)(`details`,{className:`connection-details`,"data-testid":V(`connection.footer.details`),children:[(0,x.jsx)(`summary`,{children:B(e.locale,`connection.footer.details`)}),(0,x.jsx)(`p`,{"data-testid":V(`connection.footer.instance`),children:B(e.locale,`connection.footer.instance`,{id:e.connection.details.instance})}),(0,x.jsx)(`p`,{"data-testid":V(`connection.footer.sequence`),children:B(e.locale,`connection.footer.sequence`,{sequence:e.connection.details.sequence})})]}):null]})]}));Zn.displayName=`Sidebar`;var Qn=[`mlxcel`,`glass`],$n=[`light`,`dark`],er=[`system`,`light`,`dark`],tr=`mlxcel`,nr=`system`;Qn.flatMap(e=>$n.map(t=>`${e}-${t}`));var rr=`(prefers-color-scheme: dark)`;function ir(e){return typeof e==`string`&&Qn.includes(e)}function ar(e){return typeof e==`string`&&er.includes(e)}function or(e,t){return`${e}-${t}`}function sr(){try{return typeof window>`u`||typeof window.matchMedia!=`function`?`light`:window.matchMedia(`(prefers-color-scheme: dark)`).matches?`dark`:`light`}catch{return`light`}}function cr(e,t=sr()){return e===`system`?t:e}function lr(e,t,n=sr()){let r=cr(t,n);return{id:or(e,r),family:e,scheme:r,preference:t}}function ur(e){if(typeof window>`u`||typeof window.matchMedia!=`function`)return()=>void 0;let t;try{t=window.matchMedia(rr)}catch{return()=>void 0}let n=t=>e(t.matches?`dark`:`light`);return t.addEventListener(`change`,n),()=>t.removeEventListener(`change`,n)}var dr={themeFamily:tr,colorScheme:nr,material:`glass`,glassIntensity:35,reduceMotion:!1,reduceTransparency:!1,highContrast:`system`,locale:`en`},fr=`mlxcel.webui.appearance`;function pr(e){let t=typeof e==`number`&&Number.isFinite(e)?e:dr.glassIntensity;return Math.max(0,Math.min(100,Math.round(t)))}function mr(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function hr(e){return e===`glass`||e===`tinted`||e===`opaque`}function gr(e){return e===`system`||e===`on`||e===`off`}function _r(e){return gr(e)?e:e===!0?`on`:e===!1?`off`:dr.highContrast}function vr(e){return ar(e.colorScheme)?e.colorScheme:e.colorScheme===void 0&&ar(e.theme)?e.theme:dr.colorScheme}function yr(e){return e===`en`||e===`ko`}function br(){if(typeof window>`u`)return dr;let e;try{e=window.localStorage.getItem(fr)}catch{return dr}if(!e)return dr;try{let t=JSON.parse(e);return mr(t)?{themeFamily:ir(t.themeFamily)?t.themeFamily:dr.themeFamily,colorScheme:vr(t),material:hr(t.material)?t.material:dr.material,glassIntensity:pr(t.glassIntensity),reduceMotion:t.reduceMotion===!0,reduceTransparency:t.reduceTransparency===!0,highContrast:_r(t.highContrast),locale:yr(t.locale)?t.locale:dr.locale}:dr}catch{return dr}}function xr(e){try{window.localStorage.setItem(fr,JSON.stringify(e))}catch{}}function Sr(){return typeof CSS>`u`||typeof CSS.supports!=`function`?!1:CSS.supports(`backdrop-filter: blur(1px)`)||CSS.supports(`-webkit-backdrop-filter: blur(1px)`)}function Cr(e,t,n=sr()){let r=lr(t.themeFamily,t.colorScheme,n);e.dataset.theme=r.id,e.dataset.themeFamily=r.family,e.dataset.colorScheme=r.preference,e.dataset.material=t.reduceTransparency?`opaque`:t.material,e.dataset.glassIntensity=String(pr(t.glassIntensity)),e.dataset.reduceMotion=String(t.reduceMotion),e.dataset.reduceTransparency=String(t.reduceTransparency),e.dataset.highContrast=t.highContrast,e.dataset.backdropFilter=Sr()?`supported`:`unsupported`,e.lang=t.locale}function wr(e,t,n){let r=e.filter(e=>e.runtime.server_instance_id===t.server_instance_id&&e.runtime.model_id===t.model_id&&e.receivedAt>n-3e5&&e.receivedAt<=n),i=r.at(-1);return i!==void 0&&Math.floor(i.receivedAt/2e3)===Math.floor(n/2e3)&&r.pop(),[...r,{receivedAt:n,runtime:t}].slice(-150)}function Tr(e,t){let n=[...e.values()].sort((e,t)=>t.updated_at.localeCompare(e.updated_at)),r=n.filter(e=>!Er(e)),i=n.filter(e=>Er(e)&&Date.parse(e.updated_at)>t-36e5).slice(0,200);return new Map([...r,...i].map(e=>[e.operation_id,e]))}function Er(e){return[`succeeded`,`failed`,`cancelled`].includes(e.state)}var Dr=`webui.ui-api.v1`;function Or(){return{schemaVersion:Dr,auth:{status:`signed-out`,tokenPresent:!1},connection:`idle`,bootstrap:null,catalog:[],catalogSequence:null,operations:new Map,runtimes:new Map,runtimeHistory:[],selectedModelId:null,serverInstanceId:null,lastEventId:null,lastSequence:null,lastUpdatedAt:null,lastSuccessfulAt:null,error:null,pendingReconciliations:new Map,resourceFences:{catalog:null,operationsSnapshot:null,operations:new Map,models:new Map,runtimes:new Map}}}function kr(e,t){return t.type===`login-start`?{...e,auth:{status:`authenticating`,tokenPresent:!0},connection:`bootstrapping`,error:null}:t.type===`login-success`?Ar(e,t.bootstrap,t.now):t.type===`logout`?{...Or(),lastUpdatedAt:t.now}:t.type===`select-model`?{...e,selectedModelId:t.modelId,runtimeHistory:[]}:t.type===`catalog`?jr(e,t.response,t.now):t.type===`operations-snapshot`?Mr(e,t.response,t.now):t.type===`operation`?Nr(e,t.operation,t.sequence,t.now):t.type===`runtime`?Pr(e,t.runtime,t.sequence,t.now):t.type===`event`?Fr(e,t.event,t.now):t.type===`pending`?{...e,pendingReconciliations:Ur(e.pendingReconciliations,t.item.idempotencyKey,t.item)}:t.type===`reconciled`?{...e,pendingReconciliations:Wr(e.pendingReconciliations,t.idempotencyKey)}:{...e,connection:t.connection,error:t.error??null,lastUpdatedAt:t.now}}function Ar(e,t,n){let r=t.server.server_instance_id;return e.serverInstanceId!==null&&e.serverInstanceId!==r?{...Or(),auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n}:{...e,auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n,error:null}}function jr(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return zr(e,t.server_instance_id,n);if(e.catalogSequence!==null&&t.snapshot_sequence<e.catalogSequence)return e;let r=e.catalogSequence===t.snapshot_sequence;return Br({...e,connection:e.connection===`bootstrapping`?`ready`:e.connection,catalog:Vr(r?e.catalog:[],t),catalogSequence:t.snapshot_sequence,serverInstanceId:t.server_instance_id,lastSequence:Hr({...e.resourceFences,catalog:t.snapshot_sequence}),error:null,resourceFences:{...e.resourceFences,catalog:t.snapshot_sequence}},n)}function Mr(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return zr(e,t.server_instance_id,n);if(e.resourceFences.operationsSnapshot!==null&&t.snapshot_sequence<e.resourceFences.operationsSnapshot)return e;let r=e.resourceFences.operationsSnapshot===t.snapshot_sequence,i=r?new Map(e.operations):new Map,a=r?new Map(e.resourceFences.operations):new Map;for(let e of t.items)i.set(e.operation_id,e),a.set(e.operation_id,t.snapshot_sequence);let o=Tr(i,n),s={...e.resourceFences,operationsSnapshot:t.snapshot_sequence,operations:new Map([...a].filter(([e])=>o.has(e)))};return Br({...e,operations:o,serverInstanceId:t.server_instance_id,lastSequence:Hr(s),error:null,resourceFences:s},n)}function Nr(e,t,n,r){if(n!==null&&e.resourceFences.operationsSnapshot!==null&&n<=e.resourceFences.operationsSnapshot)return e;let i=e.resourceFences.operations.get(t.operation_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=Tr(Ur(e.operations,t.operation_id,t),r),o=n===null?e.resourceFences.operations:Ur(e.resourceFences.operations,t.operation_id,n),s={...e.resourceFences,operations:new Map([...o].filter(([e])=>a.has(e)))};return Br({...e,operations:a,lastSequence:Hr(s),error:null,resourceFences:s},r)}function Pr(e,t,n,r){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return zr(e,t.server_instance_id,r);let i=e.resourceFences.runtimes.get(t.model_id);if(n!==null&&i!==void 0&&n<i)return e;let a=n===null?e.resourceFences.runtimes:Ur(e.resourceFences.runtimes,t.model_id,n),o={...e.resourceFences,runtimes:a};return Br({...e,runtimes:Ur(e.runtimes,t.model_id,t),runtimeHistory:t.model_id===e.selectedModelId?wr(e.runtimeHistory,t,r):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastSequence:Hr(o),error:null,resourceFences:o},r)}function Fr(e,t,n){if(t.schema_version!==`webui.ui-api.v1`)return{...e,connection:`schema-mismatch`,error:{code:`schema_mismatch`,message:`Unsupported WebUI schema ${t.schema_version}`,retryable:!1},lastUpdatedAt:n};if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return zr(e,t.server_instance_id,n);if(t.type===`heartbeat`)return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:e.connection===`polling`?`polling`:`streaming`,lastUpdatedAt:n};if(t.type===`server_restart`||t.type===`gap`||t.type===`reset`)return Rr(e,t,n);if(t.type===`operation`)return{...Nr(e,t.payload.operation,t.sequence,n),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`};if(t.type===`model_revision`)return Ir(e,t,n);if(t.type===`runtime`)return Lr(e,t,n);if(t.type===`snapshot`){let r={...e.resourceFences,catalog:t.payload.catalog_changed?t.sequence:e.resourceFences.catalog};return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Hr(r),connection:`streaming`,lastUpdatedAt:n,resourceFences:r}}return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`,lastUpdatedAt:n}}function Ir(e,t,n){let r=t.payload.model_id,i=e.resourceFences.models.get(r);if(i!==void 0&&t.sequence<=i)return e;let a=e.catalog.map(e=>e.identity.id===r&&t.payload.revision>=e.identity.revision?{...e,identity:{...e.identity,revision:t.payload.revision},lifecycle:t.payload.lifecycle}:e),o={...e.resourceFences,models:Ur(e.resourceFences.models,r,t.sequence)};return Br({...e,catalog:a,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Hr(o),connection:`streaming`,resourceFences:o},n)}function Lr(e,t,n){let r=t.payload.runtime,i=e.resourceFences.runtimes.get(r.model_id);if(i!==void 0&&t.sequence<=i)return e;let a={...e.resourceFences,runtimes:Ur(e.resourceFences.runtimes,r.model_id,t.sequence)};return Br({...e,runtimes:Ur(e.runtimes,r.model_id,r),runtimeHistory:r.model_id===e.selectedModelId?wr(e.runtimeHistory,r,n):e.runtimeHistory,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:Hr(a),connection:`streaming`,resourceFences:a},n)}function Rr(e,t,n){let r=t.type===`server_restart`?null:e.lastSuccessfulAt;return{...Or(),auth:e.auth,connection:`stale`,serverInstanceId:t.server_instance_id,selectedModelId:e.selectedModelId,lastEventId:t.event_id,error:{code:t.type,message:t.payload.reason,retryable:t.payload.resnapshot},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n,lastSuccessfulAt:r}}function zr(e,t,n){return{...Or(),auth:e.auth,connection:`stale`,serverInstanceId:t,selectedModelId:e.selectedModelId,error:{code:`server_restarted`,message:`The mlxcel server restarted; refresh the authoritative snapshot before continuing.`,retryable:!0},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n}}function Br(e,t){return{...e,lastUpdatedAt:t,lastSuccessfulAt:t}}function Vr(e,t){if(e.length>0){let n=new Map(e.map(e=>[e.identity.id,e]));for(let e of t.items)n.set(e.identity.id,e);return Array.from(n.values()).sort((e,t)=>e.identity.display_name.localeCompare(t.identity.display_name))}return t.items}function Hr(e){let t=[e.catalog,e.operationsSnapshot,...e.operations.values(),...e.models.values(),...e.runtimes.values()].filter(e=>e!==null);return t.length===0?null:Math.min(...t)}function Ur(e,t,n){let r=new Map(e);return r.set(t,n),r}function Wr(e,t){let n=new Map(e);return n.delete(t),n}var Gr=2e3,Kr=1e4,qr=3e4,Jr=6e4,Yr=6e4,Xr=new Set([`snapshot`,`model_revision`]),Zr=class{client;dispatch;getSnapshot;clock;visibility;random;timer=null;observationTimer=null;inflight=null;eventAbort=null;stopped=!0;failures=0;generation=0;needsFullSnapshot=!0;lastFullSnapshotAt=null;unsubscribe;constructor(e){this.client=e.client,this.dispatch=e.dispatch,this.getSnapshot=e.getSnapshot,this.clock=e.clock??ri,this.visibility=e.visibility??ii,this.random=e.random??Math.random,this.unsubscribe=this.visibility.subscribe(()=>this.visibilityChanged())}start(){this.stopped&&(this.stopped=!1,this.generation+=1,this.needsFullSnapshot=!0,this.reschedule(0))}stop(){this.generation+=1,this.stopped=!0,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null}selectionChanged(){this.cancelObservation(),this.reschedule(0)}fullSnapshotDue(e){let t=this.getSnapshot();return t.bootstrap===null||t.catalogSequence===null||this.needsFullSnapshot||this.lastFullSnapshotAt===null||this.eventAbort===null?!0:e-this.lastFullSnapshotAt>=Yr}cancelObservation(){this.generation+=1,this.observationTimer!==null&&this.clock.clearTimeout(this.observationTimer),this.observationTimer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null}visibilityChanged(){this.cancelObservation(),this.needsFullSnapshot=!0,!this.stopped&&(this.dispatch({type:`connection`,connection:`stale`,now:this.clock.now()}),this.reschedule(0))}dispose(){this.stop(),this.unsubscribe()}async resnapshot(){this.needsFullSnapshot=!0,await this.refresh()}async refresh(){if(this.inflight!==null)return;let e=new AbortController;this.inflight=e;let t=this.generation,n=this.clock.now(),r=this.clock.setTimeout(()=>{e.abort(),this.generation===t&&this.dispatch({type:`connection`,connection:`stale`,error:{code:`observation_timeout`,message:`Runtime observation timed out; retrying.`,retryable:!0},now:this.clock.now()})},Kr);this.observationTimer=r;try{if(this.getSnapshot().auth.status===`signed-out`)return;let r=this.fullSnapshotDue(n),i=null,a=[];if(r){if(i=await this.client.bootstrap(e.signal),!this.isCurrent(e,t)||this.getSnapshot().auth.status===`signed-out`||(this.dispatch({type:`login-success`,bootstrap:i,now:n}),a=await this.catalogSnapshot(e.signal),!this.isCurrent(e,t)))return;for(let e of a)this.dispatch({type:`catalog`,response:e,now:n})}let o=await this.operationSnapshot(e.signal);if(!this.isCurrent(e,t))return;let s=a.at(-1),c=o[0];if(s!==void 0&&c!==void 0&&c.server_instance_id!==s.server_instance_id){this.dispatch({type:`connection`,connection:`stale`,error:{code:`snapshot_mismatch`,message:`Operation and catalog snapshots came from different server instances.`,retryable:!0},now:n});return}let l=o.flatMap(e=>(this.dispatch({type:`operations-snapshot`,response:e,now:n}),e.items)),u=this.getSnapshot().selectedModelId,d=r&&i!==null&&u!==null&&a.every(e=>e.server_instance_id===i.server.server_instance_id)&&!a.some(e=>e.items.some(e=>e.identity.id===u));if(d&&this.dispatch({type:`select-model`,modelId:null}),d||await this.refreshSelectedRuntime(e.signal,t),r&&(this.needsFullSnapshot=!1,this.lastFullSnapshotAt=n),!this.isCurrent(e,t))return;let f=this.reconcilePending(l,n);this.failures=0,f||this.dispatch({type:`connection`,connection:`ready`,now:n}),this.eventAbort===null&&!this.visibility.hidden()&&this.startEvents()}catch(n){!e.signal.aborted&&this.generation===t&&(this.failures+=1,this.dispatch({type:`connection`,connection:$r(n),error:ei(n),now:this.clock.now()}))}finally{this.clock.clearTimeout(r),this.observationTimer===r&&(this.observationTimer=null),this.inflight===e&&(this.inflight=null),!this.stopped&&this.generation===t&&this.reschedule(this.pollDelay())}}noteUnknownPost(e){this.dispatch({type:`pending`,item:e}),this.reschedule(0)}startEvents(){this.eventAbort?.abort();let e=new AbortController;this.eventAbort=e;let t=this.generation,n=Qr(this.getSnapshot());this.client.events({onEvent:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.handleEvent(n)},onRetryAfter:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.reschedule(n)}},e.signal,n).then(()=>{this.generation===t&&(this.eventAbort===e&&(this.eventAbort=null),!e.signal.aborted&&!this.stopped&&this.reschedule(this.backoffDelay()))}).catch(n=>{e.signal.aborted||this.stopped||this.generation!==t||(this.failures+=1,this.eventAbort=null,this.dispatch({type:`connection`,connection:$r(n),error:ei(n),now:this.clock.now()}),this.reschedule(this.backoffDelay()))})}handleEvent(e){this.dispatch({type:`event`,event:e,now:this.clock.now()}),Xr.has(e.type)&&(this.needsFullSnapshot=!0),(e.type===`server_restart`||e.type===`gap`||e.type===`reset`)&&e.payload.resnapshot&&(this.needsFullSnapshot=!0,this.eventAbort?.abort(),this.eventAbort=null,this.refresh())}async catalogSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.catalog(n===void 0?{}:{cursor:n},e);if(r===null?r=i:ni(`catalog`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operationSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null?r=i:ni(`operations`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async refreshSelectedRuntime(e,t){let n=this.getSnapshot().selectedModelId;if(n===null)return;let r;try{r=await this.client.runtime(n,e)}catch(e){throw typeof e==`object`&&e&&`status`in e&&e.status===404?new ti(`Selected model changed during observation; refreshing the catalog.`):e}e.aborted||this.generation!==t||this.getSnapshot().selectedModelId===n&&this.dispatch({type:`runtime`,runtime:r,sequence:r.snapshot_sequence,now:this.clock.now()})}reconcilePending(e,t){let n=!1,r=new Set(e.map(e=>e.operation_id));for(let e of this.getSnapshot().pendingReconciliations.values())e.operationId!==null&&r.has(e.operationId)?this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}):t-e.createdAt>=Jr&&(this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}),this.dispatch({type:`connection`,connection:`stale`,error:{code:`unknown_post_unresolved`,message:`A previous control request could not be matched to an operation after reconciliation.`,retryable:!0},now:t}),n=!0);return n}reschedule(e){this.stopped||this.visibility.hidden()||(this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=this.clock.setTimeout(()=>{this.timer=null,this.refresh()},e))}pollDelay(){return Gr}backoffDelay(){let e=Math.min(qr,500*2**Math.min(6,this.failures));return Math.round(e/2+this.random()*(e/2))}isCurrent(e,t){return this.inflight===e&&this.generation===t&&!e.signal.aborted}};function Qr(e){let t=Hr(e.resourceFences);if(e.serverInstanceId!==null&&t!==null)return{lastEventId:null,serverInstanceId:e.serverInstanceId,afterSequence:t};if(e.lastEventId!==null)return{lastEventId:e.lastEventId,serverInstanceId:null,afterSequence:null}}function $r(e){if(e instanceof ti)return`stale`;let t=typeof e==`object`&&e&&`status`in e&&typeof e.status==`number`?e.status:null;return t===401||e instanceof Error&&/401|unauthorized/i.test(e.message)?`unauthorized`:t===403||e instanceof Error&&/403|forbidden/i.test(e.message)?`forbidden`:e instanceof DOMException&&e.name===`AbortError`?`stale`:`offline`}function ei(e){return e instanceof ti?{code:e.code,message:e.message,retryable:!0}:{code:`sync_error`,message:e instanceof Error?e.message.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`):`Unknown WebUI client error`,retryable:!0}}var ti=class extends Error{code=`snapshot_mismatch`;constructor(e){super(e),this.name=`SnapshotConsistencyError`}};function ni(e,t,n){if(n.server_instance_id!==t.server_instance_id)throw new ti(`${e} pagination crossed a server restart; refresh the authoritative snapshot.`);if(n.snapshot_sequence!==t.snapshot_sequence)throw new ti(`${e} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`)}var ri={setTimeout:(e,t)=>globalThis.setTimeout(e,t),clearTimeout:e=>globalThis.clearTimeout(e),now:()=>Date.now()},ii={hidden:()=>typeof document<`u`&&document.hidden,subscribe:e=>typeof document>`u`?()=>void 0:(document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e))},ai=_.createContext(null),oi=_.createContext(null);function si({children:e,apiBase:t,fetchImpl:n}){let[r,i]=_.useReducer(kr,void 0,Or),a=_.useRef(r);a.current=r;let o=_.useRef(0),s=_.useRef(null),c=_.useMemo(()=>new vn({apiBase:t,fetchImpl:n,onUnauthorized:()=>{o.current+=1,s.current?.stop(),i({type:`logout`,now:Date.now()})}}),[t,n]);_.useEffect(()=>{let e=new Zr({client:c,dispatch:i,getSnapshot:()=>a.current});return s.current=e,()=>{e.dispose(),s.current=null,c.abortAll()}},[c]);let l=_.useMemo(()=>({getTokenCount:(e,t,n)=>c.tokenCount(d(e),t,n),getSettings:(e,t)=>c.settings(d(e),t),patchSettings:(e,t,n)=>c.patchSettings(d(e),t,n),getModelProps:(e,t)=>c.modelProps(d(e),t),login:async e=>{o.current+=1;let t=o.current;c.abortAll(),c.setBearerToken(e),i({type:`login-start`});try{let e=await c.bootstrap();if(o.current!==t)return;i({type:`login-success`,bootstrap:e,now:Date.now()}),s.current?.start()}catch(e){throw o.current===t&&(c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})),e}},logout:()=>{o.current+=1,s.current?.stop(),c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})},refresh:async()=>{await s.current?.resnapshot()},selectModel:e=>{i({type:`select-model`,modelId:e}),s.current?.selectionChanged()},loadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},unloadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},downloadModel:async e=>{await u(`download`,e.idempotency_key,void 0,()=>c.download(e))},removeModel:async e=>{await u(`removal`,e.idempotency_key,e.model_id,()=>c.removeModel(e))},refreshRuntime:async e=>{let t=o.current,n=await c.runtime(e);return o.current===t&&i({type:`runtime`,runtime:n,sequence:n.snapshot_sequence,now:Date.now()}),n},refreshCatalog:async e=>{await u(`catalog-refresh`,e,void 0,()=>c.refreshCatalog(e))},cancelOperation:async e=>{let t=o.current;await c.cancelOperation(e),o.current===t&&await s.current?.resnapshot()},streamChatCompletions:async(e,t,n,r)=>{await c.chatCompletions(d(e),t,n,r)},streamResponses:async(e,t,n,r)=>{await c.responses(d(e),t,n,r)}}),[c]);async function u(e,t,n,r){let i=o.current;try{let a=await r();if(o.current!==i)return;s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:a.operation_id,modelId:n,createdAt:Date.now()})}catch(r){throw o.current===i&&!(r instanceof gn)&&s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:null,modelId:n,createdAt:Date.now()}),r}}function d(e){let t=a.current.catalog.find(t=>t.identity.id===e);if(t===void 0)throw Error(`Selected model is not present in the catalog snapshot.`);if(t.identity.inference_id.length===0)throw Error(`Selected model does not expose an inference model id.`);return t.identity.inference_id}return(0,x.jsx)(oi.Provider,{value:l,children:(0,x.jsx)(ai.Provider,{value:r,children:e})})}function ci(){let e=_.useContext(ai);if(e===null)throw Error(`useWebUi must be used within WebUiProvider.`);return e}function li(){let e=_.useContext(oi);if(e===null)throw Error(`useWebUiActions must be used within WebUiProvider.`);return e}function ui(e){return e.value===null||!Number.isFinite(e.value)?`N/A`:`${new Intl.NumberFormat(void 0,{maximumFractionDigits:2}).format(e.value)} ${e.unit}`}function di(e){let t=e.progress;return!t.indeterminate&&t.total_bytes!==null&&t.total_bytes>0?Math.min(100,100*t.completed_bytes/t.total_bytes):void 0}function fi(e){let t=e.selectedModelId===null?void 0:e.runtimes.get(e.selectedModelId);return JSON.stringify({format:`mlxcel-webui-diagnostics-v1`,connection:e.connection,last_successful_at:e.lastSuccessfulAt,operation_counts:Object.fromEntries([`queued`,`running`,`cancelling`,`succeeded`,`failed`,`cancelled`].map(t=>[t,[...e.operations.values()].filter(e=>e.state===t).length])),catalog_count:e.catalog.length,runtime_present:t!==void 0,history_samples:e.runtimeHistory.length,measurements:t===void 0?[]:Object.values(t.measurements).map(e=>({value:e.value,available:e.value!==null}))},null,2)}function pi(e){let t=URL.createObjectURL(new Blob([fi(e)],{type:`application/json`})),n=document.createElement(`a`);n.href=t,n.download=`mlxcel-diagnostics.json`,n.click(),URL.revokeObjectURL(t)}function mi({operations:e,locale:t,stale:n}){let r=[...e.values()].sort((e,t)=>Number(Er(e))-Number(Er(t))||t.updated_at.localeCompare(e.updated_at));return(0,x.jsxs)(`section`,{"aria-label":B(t,`activity.operations`),children:[(0,x.jsx)(`h2`,{children:B(t,`activity.operations`)}),(0,x.jsx)(`p`,{role:`status`,"aria-live":`polite`,children:B(t,`activity.operations.counts`,{active:String(r.filter(e=>!Er(e)).length),failed:String(r.filter(e=>e.state===`failed`).length)})}),(0,x.jsx)(`p`,{children:B(t,`activity.session`)}),(0,x.jsx)(Ze,{animate:r.some(e=>!Er(e)),children:r.length===0?(0,x.jsx)(I,{title:B(t,`activity.operations.empty.title`),body:B(t,`activity.operations.empty.body`)}):(0,x.jsx)(`ol`,{className:`activity-operations`,children:r.map(e=>(0,x.jsx)(`li`,{children:(0,x.jsx)(hi,{operation:e,locale:t,stale:n})},e.operation_id))})})]})}function hi({operation:e,locale:t,stale:n}){let r=li(),i=ci(),a=e.target,o=a.target_kind===`model`?i.catalog.find(e=>e.identity.id===a.model_id)?.identity.display_name??a.model_id:a.target_kind===`download`?a.repo_id:a.target_kind,[s,c]=(0,_.useState)(!1),[l,u]=(0,_.useState)(!1),d=Er(e),f=async()=>{c(!0),u(!1);try{await r.cancelOperation(e.operation_id)}catch{u(!0)}finally{c(!1)}};return(0,x.jsxs)(`article`,{className:`activity-operation`,children:[(0,x.jsxs)(`header`,{children:[(0,x.jsx)(`strong`,{children:e.kind.replaceAll(`_`,` `)}),(0,x.jsx)(ue,{state:e.state===`failed`?`failed`:d?`unloaded`:e.state===`cancelling`?`draining`:`loading`,label:B(t,`activity.status`),children:e.state})]}),(0,x.jsx)(`p`,{children:o}),(0,x.jsx)(`p`,{children:(0,x.jsx)(`time`,{dateTime:e.updated_at,children:new Date(e.updated_at).toLocaleString(t)})}),!d&&e.kind===`download`?(0,x.jsx)(de,{label:B(t,`activity.bytes`),value:di(e),detail:B(t,`activity.download_progress`,{completed:e.progress.completed_bytes.toLocaleString(t),total:e.progress.total_bytes?.toLocaleString(t)??B(t,`activity.not_available`)})}):null,e.state===`cancelling`?(0,x.jsx)(`p`,{role:`status`,children:B(t,`activity.cancelling`)}):null,e.state===`failed`?(0,x.jsx)(z,{title:B(t,`activity.failure`),body:e.error?.code??`operation_failed`}):null,(0,x.jsxs)(`details`,{children:[(0,x.jsx)(`summary`,{children:B(t,`activity.details`)}),(0,x.jsx)(`p`,{children:e.operation_id}),(0,x.jsx)(`p`,{children:e.target.target_kind===`model`?e.target.model_id:e.target.target_kind})]}),!d&&e.state!==`cancelling`?e.cancellable?(0,x.jsx)(P,{disabled:n,busy:s,onClick:()=>{f()},children:B(t,`activity.cancel`)}):(0,x.jsx)(`p`,{children:B(t,`activity.cancel_unsupported`)}):null,l?(0,x.jsx)(z,{title:B(t,`activity.cancel_failed`),body:B(t,`activity.refresh`)}):null]})}var gi=new Set(ke.map(e=>e.key).filter(e=>e.startsWith(`activity.metric.`))),_i=e=>gi.has(e);function vi({runtime:e,locale:t,stale:n=!1}){if(e===void 0)return(0,x.jsx)(I,{title:B(t,`activity.runtime`),body:B(t,`activity.unavailable`)});let r=Object.entries(e.measurements),i=new Set([`active_requests`,`queued_requests`,`completed_requests_total`,`completion_tokens_total`]),a=r.filter(([e,t])=>i.has(e)&&t.value!==null&&Number.isFinite(t.value)),o=r.filter(([,e])=>e.value===null||!Number.isFinite(e.value)).length,s=e=>{let n=`activity.metric.${e}`;return _i(n)?B(t,n):e.replaceAll(`_`,` `)},c=B(t,`activity.not_available`);return(0,x.jsxs)(`section`,{"aria-label":B(t,`activity.runtime`),children:[(0,x.jsx)(`h2`,{children:B(t,`activity.runtime`)}),n?(0,x.jsx)(z,{tone:`warning`,title:B(t,`activity.stale`),body:B(t,`activity.observed`)}):null,(0,x.jsx)(`div`,{className:`activity-metrics activity-metrics--summary`,"data-testid":`runtime-summary`,children:a.map(([e,t])=>(0,x.jsx)(st,{className:`activity-metric`,label:s(e),value:ui(t)},e))}),a.length===0?(0,x.jsx)(`p`,{children:B(t,`activity.no_primary`)}):null,o>0?(0,x.jsxs)(`p`,{children:[B(t,`activity.unavailable_count`),`: `,o,`. `,B(t,`activity.unavailable_reason`)]}):null,(0,x.jsx)(`h3`,{children:B(t,`activity.slots`)}),(0,x.jsxs)(`p`,{children:[B(t,`activity.parallel`),`: `,e.slots.effective_parallelism??c,` / `,e.slots.configured_parallelism]}),(0,x.jsx)(`p`,{children:B(t,`activity.context_line`,{context:B(t,`activity.context`),request:e.slots.request_context_tokens?.toString()??c,pool:B(t,`activity.pool`),shared:e.slots.shared_pool_context_tokens?.toString()??c})}),e.slots.reason===null?null:(0,x.jsx)(`p`,{children:e.slots.reason}),(0,x.jsxs)(`p`,{children:[B(t,`activity.observed`),`: `,e.slots.measured_at===null?B(t,`activity.unknown_time`):new Date(e.slots.measured_at).toLocaleString(t)]}),e.slots.available?e.slots.items.map(n=>(0,x.jsxs)(`section`,{"aria-label":`${B(t,`activity.slot`)} ${n.id}`,children:[(0,x.jsxs)(`h4`,{children:[B(t,`activity.slot`),` `,n.id,` · `,n.processing?B(t,`activity.processing`):B(t,`activity.idle`)]}),n.prompt_tokens!==null&&e.slots.request_context_tokens!==null&&e.slots.request_context_tokens>0?(0,x.jsx)(de,{label:`${B(t,`activity.slot`)} ${n.id} ${B(t,`activity.context`)}`,value:n.prompt_tokens/e.slots.request_context_tokens*100,detail:B(t,`activity.slot_progress`,{used:String(n.prompt_tokens),total:String(e.slots.request_context_tokens)})}):(0,x.jsx)(`p`,{children:B(t,`activity.no_context`)}),(0,x.jsx)(`p`,{children:B(t,`activity.slot_tokens`,{decoded:n.decoded_tokens?.toString()??c,cached:n.cached_prompt_tokens?.toString()??c})})]},n.id)):e.slots.reason===null?(0,x.jsx)(`p`,{children:B(t,`activity.unavailable`)}):null,(0,x.jsxs)(`details`,{className:`activity-measurement-details`,children:[(0,x.jsx)(`summary`,{children:B(t,`activity.metric_details`)}),(0,x.jsx)(`p`,{children:B(t,`activity.memory`)}),(0,x.jsx)(`p`,{children:B(t,`activity.timing`)}),(0,x.jsx)(`div`,{className:`activity-metrics`,children:r.map(([e,n])=>(0,x.jsx)(st,{className:`activity-metric`,label:s(e),value:ui(n),hint:(0,x.jsxs)(x.Fragment,{children:[(0,x.jsxs)(`p`,{children:[B(t,`activity.scope`),`: `,n.scope]}),(0,x.jsxs)(`p`,{children:[B(t,`activity.observed`),`: `,n.measured_at===null?B(t,`activity.unknown_time`):new Date(n.measured_at).toLocaleString(t)]}),n.reason===null?null:(0,x.jsx)(`p`,{children:n.reason})]})},e))})]})]})}var yi=(function(){let e=typeof document<`u`&&document.createElement(`link`).relList;return e&&e.supports&&e.supports(`modulepreload`)?`modulepreload`:`preload`})(),bi=function(e,t){return new URL(e,t).href},xi={},Si=function(e,t,n){let r=Promise.resolve();if(t&&t.length>0){let e=document.getElementsByTagName(`link`),i=document.querySelector(`meta[property=csp-nonce]`),a=i?.nonce||i?.getAttribute(`nonce`);function o(e){return Promise.all(e.map(e=>Promise.resolve(e).then(e=>({status:`fulfilled`,value:e}),e=>({status:`rejected`,reason:e}))))}function s(e){return import.meta.resolve?import.meta.resolve(e):new URL(e,import.meta.url).href}r=o(t.map(t=>{if(t=bi(t,n),t=s(t),t in xi)return;xi[t]=!0;let r=t.endsWith(`.css`);for(let n=e.length-1;n>=0;n--){let i=e[n];if(i.href===t&&(!r||i.rel===`stylesheet`))return}let i=document.createElement(`link`);if(i.rel=r?`stylesheet`:yi,r||(i.as=`script`),i.crossOrigin=``,i.href=t,a&&i.setAttribute(`nonce`,a),document.head.appendChild(i),r)return new Promise((e,n)=>{i.addEventListener(`load`,e),i.addEventListener(`error`,()=>n(Error(`Unable to preload CSS for ${t}`)))})}).filter(e=>e!==void 0))}function i(e){let t=new Event(`vite:preloadError`,{cancelable:!0});if(t.payload=e,window.dispatchEvent(t),!t.defaultPrevented)throw e}return r.then(t=>{for(let e of t||[])e.status===`rejected`&&i(e.reason);return e().catch(i)})},Ci=(0,_.lazy)(()=>Si(()=>import(`./history-BYCVBTrB.js`),[],import.meta.url));function wi({locale:e}){let t=ci(),n=li(),[r,i]=(0,_.useState)(!1),a=![`ready`,`streaming`,`polling`].includes(t.connection),o=t.runtimeHistory.at(-1)?.receivedAt??null,s=a||o===null||t.lastUpdatedAt!==null&&t.lastUpdatedAt-o>4e3,c=t.selectedModelId===null?void 0:t.runtimes.get(t.selectedModelId);return(0,x.jsxs)(`div`,{className:`screen-stack`,"data-testid":`activity-page`,children:[(0,x.jsx)(Ye,{title:B(e,`activity.title`),titleTestId:V(`activity.title`),description:B(e,`activity.intro`),actions:(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(P,{onClick:()=>{n.refresh()},"data-testid":V(`activity.refresh`),children:B(e,`activity.refresh`)}),(0,x.jsx)(P,{onClick:()=>pi(t),"data-testid":V(`activity.export`),children:B(e,`activity.export`)})]}),error:a?B(e,`activity.stale`):null,errorTestId:V(`activity.stale`),onRetry:()=>{n.refresh()},retryLabel:B(e,`activity.refresh`)}),(0,x.jsxs)(`p`,{className:`activity-updated`,children:[B(e,`activity.updated`),`: `,t.lastSuccessfulAt===null?B(e,`activity.pending`):(0,x.jsx)(`time`,{children:new Date(t.lastSuccessfulAt).toLocaleString(e)})]}),(0,x.jsx)(Me,{locale:e,label:B(e,`activity.select`),value:t.selectedModelId??``,onChange:e=>n.selectModel(e||null),options:[{value:``,label:B(e,`activity.select`)},...t.catalog.map(e=>({value:e.identity.id,label:e.identity.display_name}))]}),(0,x.jsx)(mi,{operations:t.operations,locale:e,stale:a}),t.selectedModelId===null?(0,x.jsx)(I,{title:B(e,`activity.select`),body:B(e,`activity.select_body`)}):(0,x.jsx)(vi,{runtime:c,locale:e,stale:s}),(0,x.jsx)(P,{onClick:()=>i(!r),children:r?B(e,`activity.history.hide`):B(e,`activity.history.show`)}),r?(0,x.jsx)(_.Suspense,{fallback:(0,x.jsx)(`p`,{children:B(e,`activity.pending`)}),children:(0,x.jsx)(Ci,{samples:t.runtimeHistory,locale:e})}):null]})}var Ti=[`fp16`,`float16`,`int8`,`i8`,`turbo4-asym`,`fp16+turbo4`,`turbo3-asym`,`fp16+turbo3`,`turbo3`,`turbo4`,`turbo4-sym`,`turbo4-delegated`,`fp16+turbo4-delegated`],Ei=Object.freeze({});function Di(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Profile must be an object.`);let t={};for(let[n,r]of Object.entries(e)){if(![`ctx_size`,`n_parallel`,`kv_cache_mode`].includes(n))throw Error(`Unsafe or unsupported profile field: ${n}`);if(r!=null){if(n===`kv_cache_mode`){if(typeof r!=`string`||!Ti.includes(r))throw Error(`Unknown KV cache mode.`);t.kv_cache_mode=r}else{if(typeof r!=`number`||!Number.isInteger(r)||r<1||r>(n===`ctx_size`?262144:32))throw Error(`${n} is outside its supported range.`);n===`ctx_size`?t.ctx_size=r:t.n_parallel=r}}}return Object.freeze(t)}function Oi(e){if(e.length>65536)throw Error(`Profile import exceeds 64 KiB.`);let t=JSON.parse(e);if(typeof t!=`object`||!t||Array.isArray(t))throw Error(`Invalid profile document.`);let n=t;if(n.version===0&&Object.keys(n).every(e=>[`version`,`profile`].includes(e)))return{version:1,reusable:Di(n.profile),models:{}};if(n.version!==1||Object.keys(n).some(e=>![`version`,`reusable`,`models`].includes(e)))throw Error(`Unsupported profile version or fields.`);if(typeof n.models!=`object`||n.models===null||Array.isArray(n.models))throw Error(`Invalid model profiles.`);let r=Object.create(null);if(Object.keys(n.models).length>200)throw Error(`Too many model profiles.`);for(let[e,t]of Object.entries(n.models)){if(!/^mdl_[A-Za-z0-9_-]{43}$/.test(e))throw Error(`Model profiles require opaque catalog IDs.`);r[e]=Di(t)}return{version:1,reusable:Di(n.reusable),models:r}}var ki=`mlxcel.webui.load-profiles.v1`,Ai={version:1,reusable:Ei,models:{}},ji=!1,Mi=new Set;function Ni(e){return Mi.add(e),()=>Mi.delete(e)}function Pi(){if(!ji){ji=!0;try{let e=localStorage.getItem(ki);e!==null&&(Ai=Oi(e))}catch{}}}function Fi(e){let t=JSON.stringify(e),n=Oi(t);localStorage.setItem(ki,t),Ai=n;for(let e of Mi)e()}function Ii(e){Pi();let t=(0,_.useSyncExternalStore)(Ni,()=>Ai);return{reusable:t.reusable,modelProfile:e===null?Ei:t.models[e]??Ei,profile:e!==null&&Object.hasOwn(t.models,e)?t.models[e]:t.reusable,save:(t,n)=>{let r=Di(t);if(n===`model`&&e===null)throw Error(`Select a model first.`);Fi(n===`reusable`?{...Ai,reusable:r}:{...Ai,models:{...Ai.models,[e]:r}})},reset:(t=e===null?`reusable`:`model`)=>{let n=Object.fromEntries(Object.entries(Ai.models).filter(([t])=>t!==e));Fi(t===`reusable`?{...Ai,reusable:Ei}:{...Ai,models:n})},exportJson:()=>JSON.stringify(Ai,null,2),importJson:e=>Fi(Oi(e))}}function Li(e){let t=Di(e);return[`mlxcel-server --webui`,t.ctx_size===void 0?``:`--ctx-size ${t.ctx_size}`,t.n_parallel===void 0?``:`--parallel ${t.n_parallel}`,t.kv_cache_mode===void 0?``:`--kv-cache-mode ${t.kv_cache_mode}`].filter(Boolean).join(` `)}var Ri=e=>[`succeeded`,`failed`,`cancelled`].includes(e.state),zi=e=>e.auth.status===`authenticated`&&e.bootstrap!==null&&e.catalogSequence!==null&&[`ready`,`streaming`,`polling`].includes(e.connection);function Bi(e,t){return zi(e)&&e.bootstrap?.actions[t]?.state===`enabled`}function Vi(e,t){return[...e.operations.values()].some(e=>!Ri(e)&&e.target.target_kind===`model`&&(e.target.model_id===t||e.target.eviction_target_id===t))||[...e.pendingReconciliations.values()].some(e=>e.modelId===t)}function Hi(e,t){return Bi(e,`load`)&&t.supported&&t.complete&&t.metadata.support.runnable_on_backend&&!t.lifecycle.busy&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Vi(e,t.identity.id)}function Ui(e,t){return Bi(e,`unload`)&&t.lifecycle.state===`ready`&&!Vi(e,t.identity.id)}function Wi(e,t){return Bi(e,`cache_delete`)&&t.identity.source===`cache`&&t.removable&&t.removal.eligible&&!t.lifecycle.busy&&t.lifecycle.active_requests===0&&[`unloaded`,`failed`].includes(t.lifecycle.state)&&!Vi(e,t.identity.id)}function Gi(e,t){return zi(e)&&t.lifecycle.state===`ready`&&t.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available)}function Ki(e,t){return e.catalog.filter(n=>n.identity.id!==t&&Ui(e,n)&&!n.lifecycle.busy&&n.lifecycle.active_requests===0&&n.lifecycle.draining_requests===0)}function qi(e,t){if(e===null)return B(t,`models.library.unknown`);if(e===0)return`0 B`;let n=Math.min(4,Math.floor(Math.log(e)/Math.log(1024)));return`${(e/1024**n).toLocaleString(t,{maximumFractionDigits:1})} ${[`B`,`KiB`,`MiB`,`GiB`,`TiB`][n]}`}function Ji(e,t){return e instanceof gn?e.envelope?.error.code===`stale_revision`?B(t,`models.library.stale`):e.message:B(t,`models.library.pending`)}function Yi(e,t){let n=t.query.trim().toLocaleLowerCase();return e.filter(e=>(!n||[e.identity.display_name,e.identity.inference_id,e.metadata.architecture??``].some(e=>e.toLocaleLowerCase().includes(n)))&&(!t.source||e.identity.source===t.source)&&(!t.task||e.capabilities.some(e=>e.task===t.task))&&(!t.status||e.lifecycle.state===t.status)).sort((e,n)=>{let r=e=>t.sort===`status`?e.lifecycle.state:t.sort===`source`?e.identity.source:t.sort===`task`?e.capabilities.map(e=>e.task).join(`,`):e.identity.display_name;return r(e).localeCompare(r(n))||e.identity.id.localeCompare(n.identity.id)})}function Xi(e){let t=e.split(`/`);return t.length===2&&t.every(e=>e.length<=96&&/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(e))}function Zi(e){return e.length===0||e.length<=128&&e.split(`/`).every(e=>e!==`.`&&e!==`..`&&/^[A-Za-z0-9._~+-]+$/.test(e))}function Qi({value:e,state:t,locale:n,busy:r,onClose:i,onConfirm:a}){let[o,s]=(0,_.useState)(``),[c,l]=(0,_.useState)(),u=e.kind===`capacity`?Ki(t,e.entry.identity.id):[],d=e.kind===`cancel`?e.operation.target.target_kind===`download`?e.operation.target.repo_id:e.operation.operation_id:e.entry.identity.display_name,f=B(n,e.kind===`delete`?`models.library.delete`:e.kind===`unload`?`models.unload`:e.kind===`cancel`?`models.library.cancel_download`:`models.library.capacity`),p=e.kind===`capacity`?B(n,`models.library.capacity_body`):e.kind===`cancel`?B(n,`models.library.cancel_body`,{name:d}):B(n,e.kind===`delete`?`models.library.delete_body`:`models.library.unload_body`,{name:d,source:e.entry.identity.source,count:String(e.entry.lifecycle.active_requests)}),m=t.serverInstanceId===e.instance&&(e.kind===`cancel`||t.catalog.some(t=>t.identity.id===e.entry.identity.id&&t.identity.revision===e.entry.identity.revision)),h=m&&(e.kind===`delete`?o===e.entry.identity.id:e.kind!==`capacity`||u.some(e=>e.identity.id===o&&e.identity.revision===c));return(0,x.jsxs)(ft,{open:!0,title:f,onClose:i,closeLabel:B(n,`common.close`),testId:`models-confirm`,children:[(0,x.jsx)(`p`,{className:`models-wrap`,children:p}),e.kind===`delete`?(0,x.jsx)(`code`,{className:`models-wrap`,children:e.entry.identity.id}):null,e.kind===`capacity`&&o&&!h?(0,x.jsx)(`p`,{role:`alert`,children:B(n,`models.library.stale`)}):null,m?null:(0,x.jsx)(`p`,{role:`alert`,children:B(n,`models.library.stale`)}),e.kind===`delete`?(0,x.jsx)(lt,{label:B(n,`models.library.confirm_name`),value:o,onChange:s,testId:`models-confirm-name`}):null,e.kind===`capacity`?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(`ul`,{children:t.catalog.filter(e=>[`ready`,`loading`,`draining`,`unloading`].includes(e.lifecycle.state)).map(e=>(0,x.jsxs)(`li`,{className:`models-wrap`,children:[e.identity.display_name,` · `,e.lifecycle.state,` · `,B(n,`models.library.active`),`:`,` `,e.lifecycle.active_requests]},e.identity.id))}),(0,x.jsx)(Me,{locale:n,label:B(n,`models.library.eviction`),value:o,onChange:e=>{s(e),l(u.find(t=>t.identity.id===e)?.identity.revision)},options:[{value:``,label:B(n,`models.library.choose`)},...u.map(e=>({value:e.identity.id,label:e.identity.display_name}))],testId:`models-eviction-target`})]}):null,(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(P,{onClick:i,children:B(n,`models.library.cancel`)}),(0,x.jsx)(P,{tone:`danger`,"data-testid":`models-confirm-submit`,busy:r,disabled:!h||r,onClick:()=>a(e.kind===`capacity`?o:void 0,c),children:B(n,e.kind===`capacity`?`models.library.evict_load`:`models.library.confirm`)})]})]})}function $i({locale:e,state:t,busy:n,onClose:r,onDownload:i,initial:a,error:o}){let[s,c]=(0,_.useState)(a?.repo??``),[l,u]=(0,_.useState)(a?.revision??``),[d,f]=(0,_.useState)(!1),p=Xi(s.trim())&&Zi(l.trim());return(0,x.jsx)(ft,{open:!0,title:B(e,`models.library.add`),onClose:r,closeLabel:B(e,`common.close`),testId:`models-add-dialog`,children:(0,x.jsxs)(`form`,{onSubmit:e=>{e.preventDefault(),p&&d&&!n&&i(s.trim(),l.trim())},children:[o?(0,x.jsx)(`p`,{role:`alert`,children:o}):null,(0,x.jsx)(`p`,{children:B(e,`models.library.network`)}),(0,x.jsx)(`ul`,{children:t.bootstrap?.roots.filter(e=>e.kind===`cache`).map((e,t)=>(0,x.jsx)(`li`,{children:e.display_name},t))}),(0,x.jsx)(lt,{label:B(e,`models.library.repo`),value:s,onChange:c,testId:`models-repo`,error:s&&!Xi(s.trim())?B(e,`models.library.repo_invalid`):void 0}),(0,x.jsx)(lt,{label:B(e,`models.library.revision`),value:l,onChange:u,testId:`models-revision`}),(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:d,onChange:e=>f(e.currentTarget.checked),"data-testid":`models-public-repo`}),(0,x.jsx)(`span`,{children:B(e,`models.library.public`)})]}),(0,x.jsx)(`p`,{children:B(e,`models.library.private`)}),(0,x.jsx)(`pre`,{className:`models-wrap`,children:`hf download OWNER/REPO --local-dir /path/to/models/checkpoint`}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(P,{onClick:r,children:B(e,`models.library.cancel`)}),(0,x.jsx)(P,{type:`submit`,tone:`primary`,busy:n,disabled:!p||!d||n,"data-testid":`models-download-submit`,children:B(e,`models.library.add`)})]})]})})}var ea=class extends Error{constructor(){super(`The model can no longer be loaded as confirmed.`),this.name=`LoadRefusedError`}};async function W(e,t,n,r,i,a){if(!Hi(t,n)||a&&!Ki(t,n.identity.id).some(e=>e.identity.id===a.id&&e.identity.revision===a.revision))throw new ea;try{await e.loadModel({action:`load`,...Object.keys(r).length?{load_profile:{...r}}:{},model_id:n.identity.id,expected_revision:n.identity.revision,idempotency_key:crypto.randomUUID(),...a?{eviction_target_id:a.id,eviction_target_expected_revision:a.revision}:{}})}catch(e){throw e instanceof gn&&e.envelope?.error.code===`conflict`&&i(),e}}function ta(e,t){return e instanceof ea?B(t,`models.library.stale`):Ji(e,t)}function na({locale:e}){return(0,x.jsxs)(`p`,{"data-testid":`models-pending-profile`,children:[B(e,`models.next_profile.body`),` `,(0,x.jsx)(`a`,{href:`#settings/model`,children:B(e,`models.next_profile.edit`)})]})}function ra({entry:e,profile:t,state:n,locale:r,busy:i,onAction:a,onChat:o}){let s=e.metadata,c=n.runtimes.get(e.identity.id),l=c?.revision===e.identity.revision?c.settings.effective.ctx_size:null,u=e=>e==null?B(r,`models.library.unknown`):String(e),d=e=>B(r,e?`models.library.yes`:`models.library.no`),f=[[B(r,`models.library.source`),e.identity.source],[B(r,`models.library.architecture`),u(s.architecture)],[B(r,`models.library.backend`),d(s.support.runnable_on_backend)],[B(r,`models.library.tested`),d(s.support.tested_checkpoint)],[B(r,`models.library.disk`),qi(s.disk_bytes,r)],[B(r,`models.library.memory`),qi(s.memory_estimate_bytes,r)],[B(r,`models.library.context`),u(typeof l==`number`?l:null)],[B(r,`models.library.profile`),Object.keys(t).length?JSON.stringify(t):B(r,`models.library.defaults`)],[B(r,`models.library.active`),String(e.lifecycle.active_requests)],[B(r,`models.library.worker`),d(e.lifecycle.worker_exit_observed)]],p=[s.support.reason,s.support.architecturally_supported_reason,s.support.runnable_on_backend_reason,s.support.complete_reason,s.support.tested_checkpoint_reason,...Object.values(s.unknown_reasons),...e.capabilities.map(e=>e.reason)].filter(e=>!!e);return(0,x.jsxs)(mt,{title:B(r,`models.library.details`),children:[(0,x.jsx)(`h3`,{children:e.identity.display_name}),(0,x.jsx)(`code`,{className:`models-wrap`,children:e.identity.id}),(0,x.jsx)(ue,{state:e.lifecycle.state,children:Vn(r,e.lifecycle.state)}),(0,x.jsx)(`dl`,{children:f.map(([e,t])=>(0,x.jsxs)(_.Fragment,{children:[(0,x.jsx)(`dt`,{children:e}),(0,x.jsx)(`dd`,{children:t})]},e))}),Object.keys(t).length?(0,x.jsx)(na,{locale:r}):null,(0,x.jsx)(`p`,{children:B(r,`models.library.tested_help`)}),[...new Set(p)].map(e=>(0,x.jsx)(`p`,{children:e},e)),e.lifecycle.last_error?(0,x.jsxs)(`p`,{role:`alert`,children:[B(r,`models.library.last_error`),`: `,e.lifecycle.last_error]}):null,(0,x.jsxs)(`div`,{className:`button-row`,children:[(0,x.jsx)(P,{"data-testid":`models-load`,disabled:i||!Hi(n,e),onClick:()=>a(`load`),children:B(r,`models.load`)}),(0,x.jsx)(P,{"data-testid":`models-use-chat`,disabled:i||!Gi(n,e),onClick:o,children:B(r,`models.library.chat`)}),(0,x.jsx)(P,{"data-testid":`models-unload`,disabled:i||!Ui(n,e),onClick:()=>a(`unload`),children:B(r,`models.unload`)}),e.identity.source===`cache`?(0,x.jsx)(P,{tone:`danger`,"data-testid":`models-delete`,disabled:i||!Wi(n,e),onClick:()=>a(`delete`),children:B(r,`models.library.delete`)}):null]}),Gi(n,e)?null:(0,x.jsx)(`p`,{children:B(r,`models.library.chat_reason`)}),e.removal.eligible?null:(0,x.jsxs)(`p`,{children:[e.removal.reason,` `,e.removal.instructions]}),Object.entries(n.bootstrap?.actions??{}).filter(([,e])=>e.state!==`enabled`).map(([e,t])=>(0,x.jsxs)(`p`,{children:[e,`: `,t.reason,` `,t.instructions]},e)),(0,x.jsx)(`a`,{href:`https://github.com/lablup/mlxcel/blob/main/docs/llama-server-compat.md`,target:`_blank`,rel:`noreferrer`,children:B(r,`models.library.api`)}),(0,x.jsx)(`ul`,{children:e.capabilities.map(e=>(0,x.jsxs)(`li`,{children:[e.task,` · `,e.phase,` · `,d(e.available)]},`${e.task}-${e.phase}`))})]})}function ia({state:e,locale:t,busy:n,downloadPending:r,onCancel:i,onRetry:a,onCapacity:o}){let s=[...e.operations.values()].filter(e=>[`download`,`model_load`,`model_unload`,`model_removal`,`catalog_refresh`].includes(e.kind)).sort((e,t)=>Number(Ri(e))-Number(Ri(t))||t.created_at.localeCompare(e.created_at));return!s.length&&!e.pendingReconciliations.size?(0,x.jsx)(x.Fragment,{}):(0,x.jsxs)(`section`,{className:`models-operations`,"aria-label":B(t,`models.library.operations`),children:[(0,x.jsx)(`h2`,{children:B(t,`models.library.operations`)}),e.pendingReconciliations.size?(0,x.jsx)(`p`,{role:`status`,"data-testid":`models-pending`,children:B(t,`models.library.pending`)}):null,(0,x.jsx)(`ul`,{className:`ds-list`,children:s.slice(0,64).map(s=>(0,x.jsx)(`li`,{"data-testid":`models-operation`,children:(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`strong`,{children:oa(s,e)}),(0,x.jsxs)(`p`,{children:[(0,x.jsx)(`code`,{children:s.operation_id}),s.target.target_kind===`model`?(0,x.jsxs)(`span`,{children:[` `,`· `,(0,x.jsx)(`code`,{children:s.target.model_id})]}):null,` `,`· `,B(t,`models.library.operation_state`),`: `,s.state]}),s.kind===`download`?(0,x.jsx)(de,{label:B(t,`models.library.progress`),value:!s.progress.indeterminate&&s.progress.total_bytes!==null&&s.progress.total_bytes>0?s.progress.completed_bytes/s.progress.total_bytes*100:void 0,detail:`${qi(s.progress.completed_bytes,t)} / ${qi(s.progress.total_bytes,t)}`}):null,s.error?(0,x.jsxs)(`p`,{role:`alert`,children:[s.error.code,`: `,s.error.message]}):null,s.result&&`lifecycle`in s.result?(0,x.jsxs)(`p`,{children:[s.result.lifecycle.state,` · `,B(t,`models.library.worker`),`:`,` `,B(t,s.result.lifecycle.worker_exit_observed?`models.library.yes`:`models.library.no`)]}):null,s.cancel_reason?(0,x.jsx)(`p`,{children:s.cancel_reason}):null,(0,x.jsxs)(`div`,{className:`button-row`,children:[s.kind===`model_load`&&s.state===`failed`&&s.error?.code===`conflict`&&s.target.target_kind===`model`?(0,x.jsx)(P,{disabled:n||!e.catalog.some(t=>t.identity.id===aa(s)&&Hi(e,t)),onClick:()=>o(aa(s)),children:B(t,`models.library.capacity`)}):null,s.kind===`download`&&!Ri(s)?(0,x.jsx)(P,{disabled:n||!zi(e)||!s.cancellable||s.state===`cancelling`,onClick:()=>i(s),children:B(t,`models.library.cancel_download`)}):null,s.kind===`download`&&(s.state===`failed`||s.state===`cancelled`)?(0,x.jsx)(P,{disabled:n||r||!Bi(e,`download`),onClick:()=>a(s),children:B(t,`models.library.retry_download`)}):null]})]})},s.operation_id))})]})}function aa(e){return e.target.target_kind===`model`?e.target.model_id:``}function oa(e,t){let n=e.target;return n.target_kind===`download`?n.repo_id:n.target_kind===`model`?t.catalog.find(e=>e.identity.id===n.model_id)?.identity.display_name??n.model_id:e.kind}var sa=25;function ca({locale:e}){let t=ci(),n=li(),[r,i]=(0,_.useState)({query:``,source:``,task:``,status:``,sort:`name`}),[a,o]=(0,_.useState)(0),[s,c]=(0,_.useState)(!1),l=(0,_.useRef)(!1),[u,d]=(0,_.useState)(null),[f,p]=(0,_.useState)(null),[m,h]=(0,_.useState)(null),[g,v]=(0,_.useState)(`copy`),y=t.catalog.find(e=>e.identity.id===t.selectedModelId),{profile:b}=Ii(y?.identity.id??null),{profile:S}=Ii(f?.kind===`capacity`?f.entry.identity.id:null),C=Yi(t.catalog,r),w=Math.max(1,Math.ceil(C.length/sa)),T=Math.min(a,w-1),E=[...t.operations.values()].some(e=>e.kind===`download`&&!Ri(e))||[...t.pendingReconciliations.values()].some(e=>e.kind===`download`),D=[...t.operations.values()].some(e=>e.kind===`catalog_refresh`&&!Ri(e))||[...t.pendingReconciliations.values()].some(e=>e.kind===`catalog-refresh`),O=e=>{i({...r,...e}),o(0)},k=async(n,r)=>{if(l.current||!zi(t))return!1;l.current=!0,c(!0),d(null);try{return await n(),!0}catch(t){return d(Ji(t,e)),r?.(t),!1}finally{l.current=!1,c(!1)}},A=(r,i,a=b)=>{if(!Hi(t,r)){d(B(e,`models.library.stale`));return}if(i&&!Ki(t,r.identity.id).some(e=>e.identity.id===i.id&&e.identity.revision===i.revision)){d(B(e,`models.library.stale`));return}k(()=>W(n,t,r,a,()=>p({kind:`capacity`,entry:r,instance:t.serverInstanceId}),i))},j=e=>{y&&(e===`load`?A(y):p({kind:e,entry:y,instance:t.serverInstanceId}))},ee=(r,i)=>{let a=f;if(a&&!l.current){if(a.instance!==t.serverInstanceId){d(B(e,`models.library.stale`)),p(null);return}if(a.kind===`cancel`){let e=t.operations.get(a.operation.operation_id);if(!e||!e.cancellable||Ri(e)||e.state===`cancelling`){p(null);return}k(()=>n.cancelOperation(e.operation_id))}else{let o=t.catalog.find(e=>e.identity.id===a.entry.identity.id);if(!o||o.identity.revision!==a.entry.identity.revision){d(B(e,`models.library.stale`)),p(null);return}if(a.kind===`capacity`){if(!Ki(t,o.identity.id).some(e=>e.identity.id===r&&e.identity.revision===i)){d(B(e,`models.library.stale`));return}A(o,r&&i!==void 0?{id:r,revision:i}:void 0,S)}else a.kind===`unload`&&Ui(t,o)?k(()=>n.unloadModel({action:`unload`,model_id:o.identity.id,expected_revision:o.identity.revision,idempotency_key:crypto.randomUUID()})):a.kind===`delete`&&Wi(t,o)?k(()=>n.removeModel({model_id:o.identity.id,expected_revision:o.identity.revision,idempotency_key:crypto.randomUUID()})):d(B(e,`models.library.stale`))}p(null)}},te=e=>{e.target.target_kind===`download`&&h({repo:e.target.repo_id,revision:e.target.revision??``})},ne=`mlxcel-server --webui --models-dir /path/to/models --no-models-autoload`,M=[{id:`name`,header:B(e,`models.library.name`),noResize:!0,render:r=>(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(P,{tone:`ghost`,className:pe,"aria-label":B(e,`models.library.inspect`,{name:r.identity.display_name}),onClick:()=>{r.identity.id!==t.selectedModelId&&n.selectModel(r.identity.id)},children:r.identity.display_name}),(0,x.jsxs)(`span`,{className:`models-row-meta`,children:[r.identity.id===t.selectedModelId?(0,x.jsx)(ct,{tone:`accent`,children:B(e,`models.library.selected`)}):null,` `,(0,x.jsx)(ct,{children:r.identity.source}),` `,(0,x.jsx)(ct,{children:r.metadata.quantization??B(e,`models.library.unknown`)})]})]})},{id:`task`,header:B(e,`models.library.task`),noResize:!0,render:t=>t.capabilities.map(e=>e.task).filter((e,t,n)=>n.indexOf(e)===t).join(`, `)||B(e,`models.library.unknown`)},{id:`status`,header:B(e,`models.library.status`),noResize:!0,render:t=>(0,x.jsx)(ue,{state:t.lifecycle.state,children:Vn(e,t.lifecycle.state)})},{id:`support`,header:B(e,`models.library.support`),noResize:!0,render:t=>(0,x.jsxs)(x.Fragment,{children:[B(e,t.metadata.support.architecturally_supported?`models.library.supported`:`models.library.unsupported`),(0,x.jsx)(`br`,{}),B(e,t.complete?`models.library.complete`:`models.library.incomplete`)]})}];return(0,x.jsxs)(`div`,{className:`screen-stack models-library`,"data-testid":`models-library`,children:[(0,x.jsx)(Ye,{title:B(e,`models.title`),titleTestId:V(`models.title`),description:B(e,`models.library.subtitle`),actions:(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(P,{tone:`primary`,onClick:()=>h({repo:``,revision:``}),disabled:s||E||!Bi(t,`download`),"data-testid":`models-add`,children:B(e,`models.library.add`)}),(0,x.jsx)(P,{disabled:s||D||!zi(t)||t.bootstrap?.server.mode===`single_model`,onClick:()=>{k(()=>n.refreshCatalog(crypto.randomUUID()))},"data-testid":`models-rescan`,children:B(e,`models.library.rescan`)}),(0,x.jsx)(P,{onClick:()=>{n.refresh()},disabled:s,children:B(e,`models.library.refresh`)})]}),error:zi(t)?u===null?null:B(e,`models.library.error`):B(e,`models.library.stale`),errorDetail:zi(t)?u:t.error?.message??B(e,`models.library.waiting`),errorTestId:zi(t)?`models-action-error`:`connection-error-title`,onRetry:()=>{d(null),n.refresh()},retryLabel:B(e,`models.library.refresh`)}),t.bootstrap?.server.mode===`single_model`?(0,x.jsx)(z,{tone:`info`,title:B(e,`models.library.single`),body:`mlxcel-server --webui`,testId:`models-read-only`}):null,Object.entries(t.bootstrap?.actions??{}).filter(([,e])=>e.state!==`enabled`).map(([e,t])=>(0,x.jsxs)(`p`,{children:[e,`: `,t.reason,` `,t.instructions]},e)),(0,x.jsx)(`p`,{"data-testid":`connection-authenticated-detail`,children:Rn(e,t)}),(0,x.jsxs)(`details`,{open:t.catalog.length===0,children:[(0,x.jsx)(`summary`,{children:B(e,`models.library.roots`)}),(0,x.jsx)(`ul`,{children:t.bootstrap?.roots.map((t,n)=>(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`strong`,{children:t.display_name}),` · `,t.kind,t.error?(0,x.jsxs)(`p`,{role:`alert`,children:[B(e,`models.library.permission`),` `,t.error]}):null]},n))}),(0,x.jsx)(`p`,{children:B(e,`models.library.roots_help`)}),(0,x.jsx)(`pre`,{className:`models-wrap`,children:ne}),(0,x.jsx)(P,{onClick:()=>{navigator.clipboard?.writeText(ne).then(()=>v(`copied`)).catch(()=>v(`copy_failed`)),navigator.clipboard||v(`copy_failed`)},children:B(e,`models.library.${g}`)}),(0,x.jsx)(`span`,{role:`status`,children:g===`copy`?``:B(e,`models.library.${g}`)})]}),(0,x.jsxs)(`div`,{className:`models-toolbar`,children:[(0,x.jsx)(lt,{label:B(e,`models.library.search`),value:r.query,onChange:e=>O({query:e}),testId:`models-search`}),(0,x.jsx)(Me,{locale:e,label:B(e,`models.library.source`),value:r.source,onChange:e=>O({source:e}),options:[{value:``,label:B(e,`models.library.all`)},...[...new Set(t.catalog.map(e=>e.identity.source))].sort().map(e=>({value:e,label:e}))]}),(0,x.jsx)(Me,{locale:e,label:B(e,`models.library.task`),value:r.task,onChange:e=>O({task:e}),options:[{value:``,label:B(e,`models.library.all`)},...[...new Set(t.catalog.flatMap(e=>e.capabilities.map(e=>e.task)))].sort().map(e=>({value:e,label:e}))]}),(0,x.jsx)(Me,{locale:e,label:B(e,`models.library.status`),value:r.status,onChange:e=>O({status:e}),options:[{value:``,label:B(e,`models.library.all`)},...[`unloaded`,`loading`,`ready`,`draining`,`unloading`,`failed`].map(t=>({value:t,label:Vn(e,t)}))]}),(0,x.jsx)(Me,{locale:e,label:B(e,`models.library.sort`),value:r.sort,onChange:e=>O({sort:e}),options:[`name`,`status`,`source`,`task`].map(t=>({value:t,label:B(e,`models.library.sort_${t}`)}))})]}),(0,x.jsxs)(`div`,{className:`models-layout`,children:[(0,x.jsxs)(`section`,{children:[(0,x.jsx)(me,{columns:M,rows:C.slice(T*sa,(T+1)*sa),getRowKey:e=>e.identity.id,ariaLabel:B(e,`models.title`),testId:`models-table`,activateRowPrimary:!0,rowClassName:e=>e.identity.id===t.selectedModelId?`models-selected`:void 0,loading:t.catalogSequence===null,loadingState:(0,x.jsx)(ye,{label:B(e,`models.library.waiting`)}),emptyState:(0,x.jsx)(I,{title:B(e,t.catalog.length===0?`models.empty.title`:`models.library.filtered`),body:B(e,t.catalog.length===0?`models.empty.body`:`models.library.filtered_body`),testId:`models-empty`})}),(0,x.jsxs)(`nav`,{className:`models-pagination`,"aria-label":B(e,`models.title`),children:[(0,x.jsx)(P,{disabled:T===0,onClick:()=>o(T-1),children:B(e,`models.library.previous`)}),(0,x.jsx)(`span`,{role:`status`,children:B(e,`models.library.page`,{page:String(T+1),pages:String(w),count:String(C.length)})}),(0,x.jsx)(P,{disabled:T>=w-1,onClick:()=>o(T+1),children:B(e,`models.library.next`)})]})]}),y?(0,x.jsx)(ra,{entry:y,profile:b,state:t,locale:e,busy:s,onAction:j,onChat:()=>{Gi(t,y)&&(n.selectModel(y.identity.id),window.location.hash=`chat`)}}):null]}),(0,x.jsx)(ia,{state:t,locale:e,busy:s,downloadPending:E,onCancel:e=>p({kind:`cancel`,operation:e,instance:t.serverInstanceId}),onRetry:te,onCapacity:e=>{let n=t.catalog.find(t=>t.identity.id===e);n&&Hi(t,n)&&p({kind:`capacity`,entry:n,instance:t.serverInstanceId})}}),f?(0,x.jsx)(Qi,{value:f,state:t,locale:e,busy:s,onClose:()=>p(null),onConfirm:ee},f.kind):null,m?(0,x.jsx)($i,{locale:e,state:t,initial:m,error:u,busy:s||E||!Bi(t,`download`),onClose:()=>h(null),onDownload:(e,r)=>{Bi(t,`download`)&&!E&&k(()=>n.downloadModel({repo_id:e,revision:r||null,idempotency_key:crypto.randomUUID()})).then(e=>{e&&h(null)})}}):null]})}var la={en:`en-US`,ko:`ko-KR`},ua=e=>la[e];function da(e,t){if(!Number.isFinite(e)||e<0)return B(t,`format.unknown`);let n=[`B`,`KiB`,`MiB`,`GiB`,`TiB`],r=e,i=0;for(;r>=1024&&i<n.length-1;)r/=1024,i+=1;let a=i===0?0:1;return`${new Intl.NumberFormat(ua(t),{maximumFractionDigits:a}).format(r)} ${n[i]}`}function fa(e,t){return e===null?B(t,`format.not_measured`):`${new Intl.NumberFormat(ua(t),{maximumFractionDigits:1}).format(e)} tokens/s`}function pa(e){let[t,n]=(0,_.useState)(!1),[r,i]=(0,_.useState)(`controls`),[a,o]=(0,_.useState)(`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`),[s,c]=(0,_.useState)(`ready`);return(0,x.jsxs)(`div`,{className:`screen-stack gallery-screen`,children:[(0,x.jsx)(Ye,{title:B(e.locale,`gallery.title`),titleTestId:V(`gallery.title`),description:B(e.locale,`gallery.subtitle`),descriptionTestId:V(`gallery.subtitle`)}),(0,x.jsx)(fe,{active:r,onChange:i,label:B(e.locale,`gallery.tab.sections`),tabs:[{id:`controls`,label:B(e.locale,`gallery.tab.controls`),panel:(0,x.jsx)(ma,{locale:e.locale,value:a,onValueChange:o,selectValue:s,onSelectChange:c,onOpenDialog:()=>n(!0)})},{id:`states`,label:B(e.locale,`gallery.tab.states`),panel:(0,x.jsx)(ha,{locale:e.locale})},{id:`data`,label:B(e.locale,`gallery.tab.data`),panel:(0,x.jsx)(ga,{locale:e.locale})}]}),(0,x.jsxs)(ft,{open:t,title:B(e.locale,`models.delete.confirm.title`),onClose:()=>n(!1),testId:`gallery-dialog`,closeLabel:B(e.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":V(`models.delete.confirm.body`),children:B(e.locale,`models.delete.confirm.body`,{model:a})}),(0,x.jsx)(lt,{label:B(e.locale,`models.delete.confirm.token_label`),value:``,placeholder:B(e.locale,`gallery.delete_token`)}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(P,{onClick:()=>n(!1),children:B(e.locale,`common.cancel`)}),(0,x.jsx)(P,{tone:`danger`,children:B(e.locale,`common.delete`)})]})]})]})}function ma(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsxs)(qe,{children:[(0,x.jsx)(`h2`,{children:B(e.locale,`gallery.controls.title`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(P,{tone:`primary`,children:B(e.locale,`gallery.controls.primary`)}),(0,x.jsx)(P,{children:B(e.locale,`gallery.controls.secondary`)}),(0,x.jsx)(P,{tone:`danger`,children:B(e.locale,`gallery.controls.danger`)}),(0,x.jsx)(P,{busy:!0,children:B(e.locale,`gallery.controls.busy`)})]}),(0,x.jsx)(lt,{label:B(e.locale,`gallery.field.repo`),value:e.value,onChange:e.onValueChange,error:B(e.locale,`gallery.field.repo_error`),hint:B(e.locale,`gallery.field.repo_hint`),testId:`gallery-field`}),(0,x.jsx)(Me,{locale:e.locale,label:B(e.locale,`gallery.select.native`),value:e.selectValue,onChange:e.onSelectChange,options:[{value:`ready`,label:B(e.locale,`models.status.ready`)},{value:`unloaded`,label:B(e.locale,`models.status.unloaded`)}]})]}),(0,x.jsxs)(qe,{children:[(0,x.jsx)(`h2`,{children:B(e.locale,`gallery.overlays.title`)}),(0,x.jsx)(`p`,{children:B(e.locale,`gallery.long_cjk`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(Ve,{content:B(e.locale,`gallery.tooltip`),children:(0,x.jsx)(P,{children:B(e.locale,`gallery.hover_focus`)})}),(0,x.jsx)(P,{tone:`primary`,onClick:e.onOpenDialog,children:B(e.locale,`gallery.dialog.open`)})]})]})]})}function ha(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsx)(_t,{title:B(e.locale,`state.unauthorized.title`),body:B(e.locale,`state.unauthorized.body`),tokenLabel:B(e.locale,`login.token.label`),tokenHelp:B(e.locale,`login.token.help`),submitLabel:B(e.locale,`login.submit`),logoutLabel:B(e.locale,`login.logout`),onSubmit:()=>void 0,onLogout:()=>void 0,testId:`gallery-login`}),(0,x.jsx)(vt,{title:B(e.locale,`state.schema_mismatch.title`),body:B(e.locale,`state.schema_mismatch.body`),actionLabel:B(e.locale,`common.reload`),onRecover:()=>void 0}),(0,x.jsx)(I,{title:B(e.locale,`models.empty.title`),body:B(e.locale,`models.empty.body`),action:(0,x.jsx)(P,{tone:`primary`,children:B(e.locale,`common.add_model`)}),testId:`gallery-empty`}),(0,x.jsx)(z,{tone:`warning`,title:B(e.locale,`gallery.states.load_failed`),body:B(e.locale,`gallery.states.load_failed_body`),action:(0,x.jsx)(P,{children:B(e.locale,`common.retry`)})}),(0,x.jsxs)(`div`,{className:`gallery-progress-samples`,children:[(0,x.jsx)(de,{label:B(e.locale,`gallery.download`),detail:B(e.locale,`activity.progress.indeterminate`,{bytes:da(67108864,e.locale)})}),(0,x.jsx)(de,{label:B(e.locale,`gallery.progress.measured`),value:37})]}),(0,x.jsxs)(gt,{label:B(e.locale,`gallery.lifecycle.samples`),children:[(0,x.jsx)(`li`,{children:(0,x.jsx)(ue,{state:`loading`,children:B(e.locale,`models.status.loading`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(ue,{state:`draining`,children:B(e.locale,`models.status.draining`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(ue,{state:`unloading`,children:B(e.locale,`models.status.unloading`)})})]})]})}function ga(e){return(0,x.jsxs)(`div`,{className:`gallery-grid dense-gallery`,children:[(0,x.jsxs)(qe,{className:`table-card`,role:`region`,tabIndex:0,ariaLabel:B(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`h2`,{children:B(e.locale,`gallery.data.title`)}),(0,x.jsxs)(ht,{caption:B(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`thead`,{children:(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`th`,{children:B(e.locale,`gallery.data.name`)}),(0,x.jsx)(`th`,{children:B(e.locale,`gallery.data.status`)}),(0,x.jsx)(`th`,{children:B(e.locale,`gallery.data.rate`)})]})}),(0,x.jsxs)(`tbody`,{children:[(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:B(e.locale,`models.long_name`),"aria-label":B(e.locale,`models.long_name`),children:B(e.locale,`models.long_name`)})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(ue,{state:`ready`,children:B(e.locale,`models.status.ready`)})}),(0,x.jsx)(`td`,{children:fa(39.4,e.locale)})]}),(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:`granite-4.0-h-tiny-4bit`,"aria-label":`granite-4.0-h-tiny-4bit`,children:`granite-4.0-h-tiny-4bit`})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(ue,{state:`unloaded`,children:B(e.locale,`models.status.unloaded`)})}),(0,x.jsx)(`td`,{children:fa(null,e.locale)})]})]})]})]}),(0,x.jsxs)(mt,{title:B(e.locale,`gallery.data.inspector`),children:[(0,x.jsx)(`p`,{className:`sample-label`,children:B(e.locale,`gallery.sample.label`)}),(0,x.jsx)(`p`,{children:B(e.locale,`models.unsupported.reason`)}),(0,x.jsx)(ue,{state:`failed`,children:B(e.locale,`models.status.failed`)}),(0,x.jsx)(`hr`,{}),(0,x.jsx)(`p`,{children:B(e.locale,`gallery.sample.reasoning_body`)}),(0,x.jsx)(`code`,{children:B(e.locale,`gallery.sample.tool_preview`)})]})]})}function _a(){if(typeof navigator>`u`)return!1;let e=navigator,t=e.userAgentData?.platform??e.platform??e.userAgent??``;return/mac|iphone|ipad|ipod/i.test(t)}function va(e){return _a()?e.metaKey:e.ctrlKey}var ya={count:4,bytes:8388608,dimension:4096,pixels:16e6},ba=new Set([`image/png`,`image/jpeg`,`image/webp`]);function xa(e,t){let n=new DataView(e.buffer,e.byteOffset,e.byteLength),r=(t,n)=>String.fromCharCode(...e.subarray(t,t+n));if(t===`image/png`&&e.length>=33&&e[0]===137&&r(1,7)===`PNG\r

`&&n.getUint32(8)===13&&r(12,4)===`IHDR`)return[n.getUint32(16),n.getUint32(20)];if(t===`image/jpeg`&&e.length>=4&&e[0]===255&&e[1]===216){let t=2;for(;t+4<=e.length&&e[t++]===255;){for(;e[t]===255;)t++;let r=e[t++];if(r===217||r===218)break;if(r===1||r>=208&&r<=215)continue;if(t+2>e.length)break;let i=n.getUint16(t);if(i<2||t+i>e.length)break;if([192,193,194,195,197,198,199,201,202,203,205,206,207].includes(r)&&i>=8)return[n.getUint16(t+5),n.getUint16(t+3)];t+=i}}if(t===`image/webp`&&e.length>=25&&r(0,4)===`RIFF`&&r(8,4)===`WEBP`&&n.getUint32(4,!0)+8===e.length){let t=r(12,4),i=n.getUint32(16,!0);if(i<5||20+i>e.length)throw Error(`Invalid WebP chunk length.`);let a=t=>e[t]|e[t+1]<<8|e[t+2]<<16;if(t===`VP8X`&&e.length>=30&&n.getUint32(16,!0)>=10&&!(e[20]&2))return[a(24)+1,a(27)+1];if(t===`VP8 `&&e.length>=30&&e[23]===157&&e[24]===1&&e[25]===42)return[n.getUint16(26,!0)&16383,n.getUint16(28,!0)&16383];if(t===`VP8L`&&e[20]===47){let e=n.getUint32(21,!0);return[(e&16383)+1,(e>>>14&16383)+1]}}throw Error(`Image content does not match a supported PNG, JPEG or non-animated WebP image.`)}function Sa(e){return new Promise((t,n)=>{let r=new FileReader;r.onerror=()=>n(Error(`Could not read the selected image.`)),r.onabort=()=>n(Error(`Image reading was cancelled.`)),r.onload=()=>r.result instanceof ArrayBuffer?t(new Uint8Array(r.result)):n(Error(`Invalid image data.`)),r.readAsArrayBuffer(e)})}function Ca(e){if(!e||[e.max_images,e.max_image_bytes,e.max_width,e.max_height,e.max_decoded_bytes,e.max_body_bytes].some(e=>!Number.isSafeInteger(e)||e<=0))throw Error(`Image limits are unavailable.`)}function wa(e,t,n){if(!e||!t||e>Math.min(ya.dimension,n.max_width)||t>Math.min(ya.dimension,n.max_height)||e*t>ya.pixels||e*t*4>n.max_decoded_bytes)throw Error(`Image dimensions exceed the server or browser decode budget.`)}function Ta(e,t){if(e.length===0){if(!t||!Number.isSafeInteger(t.max_body_bytes)||t.max_body_bytes<=0)throw Error(`Request body limit is unavailable.`);return}if(Ca(t),e.length>t.max_images)throw Error(`Conversation images exceed the server image count limit.`);let n=0,r=Math.min(ya.bytes,t.max_image_bytes);for(let i of e){if(!i||!ba.has(i.type)||typeof i.dataUrl!=`string`)throw Error(`Invalid local image attachment.`);if(n+=i.dataUrl.length,n>t.max_body_bytes)throw Error(`Images exceed the server request body limit.`);let e=`data:${i.type};base64,`;if(!i.dataUrl.startsWith(e))throw Error(`Invalid image data URL.`);if(i.dataUrl.length-e.length>4*Math.ceil(r/3))throw Error(`Image exceeds the current server or browser byte limit.`)}for(let n of e){let e=n.dataUrl.slice(`data:${n.type};base64,`.length);if(!e||e.length%4!=0||!/^[A-Za-z0-9+/]*={0,2}$/.test(e))throw Error(`Invalid image base64 encoding.`);let i=atob(e);if(btoa(i)!==e)throw Error(`Image base64 encoding must be canonical.`);if(!i.length||i.length>r)throw Error(`Image exceeds the current server or browser byte limit.`);let a=new Uint8Array(i.length);for(let e=0;e<i.length;e++)a[e]=i.charCodeAt(e);let[o,s]=xa(a,n.type);wa(o,s,t)}}async function Ea(e,t,n,r=[]){Ca(n);let i=Math.min(ya.count,n.max_images),a=Math.min(ya.bytes,n.max_image_bytes),o=Array.from(e);if(!Number.isInteger(t)||t<0||t+o.length>i||t!==r.length)throw Error(`Attach at most ${i} images per message; existing attachment count must match.`);for(let e of o){if(!(e instanceof File)||!ba.has(e.type))throw Error(`Choose local PNG, JPEG or WebP image files.`);if(!e.size||e.size>a)throw Error(`Each image must be nonempty and no larger than ${a} bytes.`)}if(r.reduce((e,t)=>e+t.dataUrl.length,0)+o.reduce((e,t)=>e+4*Math.ceil(t.size/3)+`data:${t.type};base64,`.length,0)>n.max_body_bytes)throw Error(`Images exceed the server request body limit.`);let s=[];for(let e of o){let t=await Sa(e),[r,i]=xa(t,e.type);if(wa(r,i,n),typeof createImageBitmap==`function`){let t;try{t=await createImageBitmap(e,{imageOrientation:`none`})}catch{throw Error(`The image could not be decoded.`)}try{if(t.width!==r||t.height!==i)throw Error(`Decoded image dimensions do not match its header.`)}finally{t.close()}}let a=``;for(let e=0;e<t.length;e+=8192)a+=String.fromCharCode(...t.subarray(e,e+8192));s.push({name:e.name,type:e.type,dataUrl:`data:${e.type};base64,${btoa(a)}`})}return s}var Da=32e3,Oa=e=>{if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Malformed chat stream.`);return e};function ka(e){if(e==null)return``;if(typeof e!=`string`)throw Error(`Malformed text delta.`);return e}function Aa(e){if(typeof e!=`number`||!Number.isSafeInteger(e)||e<0)throw Error(`Malformed usage.`);return e}function ja(e,t,n){if(t.length>256e3)throw Error(`Chat frame exceeds the display limit.`);let r=Oa(JSON.parse(t));if(r.error!==void 0)throw Error(`The server reported a generation error. No automatic retry was made.`);let i={...e};if(r.usage!==void 0&&r.usage!==null){let e=Oa(r.usage);i={...i,usage:{prompt_tokens:Aa(e.prompt_tokens),completion_tokens:Aa(e.completion_tokens),total_tokens:Aa(e.total_tokens)}}}if(!Array.isArray(r.choices)||r.choices.length>1)throw Error(`Malformed chat choices.`);for(let e of r.choices){let t=Oa(e);if(t.index!==0)throw Error(`Unexpected chat choice index.`);let r=Oa(t.delta),a=ka(r.content),o=ka(r.reasoning_content??r.reasoning),s=i.tools.map(e=>({...e}));if(r.tool_calls!==void 0&&r.tool_calls!==null){if(!Array.isArray(r.tool_calls)||r.tool_calls.length>32)throw Error(`Too many tool calls.`);for(let e of r.tool_calls){let t=Oa(e),n=Aa(t.index);if(n>=32)throw Error(`Tool index exceeds the display limit.`);let r=t.function===void 0?{}:Oa(t.function),i=s.find(e=>e.index===n)??{index:n,id:``,name:``,arguments:``},a={index:n,id:i.id+ka(t.id),name:i.name+ka(r.name),arguments:i.arguments+ka(r.arguments)};s=[...s.filter(e=>e.index!==n),a].sort((e,t)=>e.index-t.index)}}let c=a.length>0||o.length>0||s.length>0;if(i={...i,content:i.content+a,reasoning:i.reasoning+o,tools:s,ttftMs:i.ttftMs??(c?n:null)},t.finish_reason!==null&&t.finish_reason!==void 0){let e=ka(t.finish_reason);if(e.length>64)throw Error(`Invalid finish reason.`);i={...i,finishReason:e}}}if(i.content.length+i.reasoning.length+JSON.stringify(i.tools).length>256e3)throw Error(`Response exceeds the bounded display limit; generation stopped.`);return i}function Ma(e,t){if(e.finishReason===null||!e.content&&!e.reasoning&&e.tools.length===0)throw Error(`The server returned an empty or unfinished response.`);return{...e,status:`complete`,elapsedMs:t}}function Na(e,t){let n=e?[{role:`system`,content:e}]:[];for(let e of t)n.push({role:`user`,content:e.images.length?[{type:`text`,text:e.prompt},...e.images.map(e=>({type:`image_url`,image_url:{url:e.dataUrl}}))]:e.prompt}),e.status===`complete`&&e.content&&n.push({role:`assistant`,content:e.content});return n}function Pa(e){return e.usage===null||e.ttftMs===null||e.elapsedMs===null||e.elapsedMs<=e.ttftMs||e.usage.completion_tokens<=1?null:(e.usage.completion_tokens-1)*1e3/(e.elapsedMs-e.ttftMs)}var Fa=Object.freeze({conversations:50,turnsPerConversation:200,totalTurns:1e3,textCharacters:262144,jsonBytes:16777216,imageBytes:8388608,imagesPerTurn:8,toolsPerTurn:128}),Ia=`mlxcel.webui.chat.v1`,La=`history`,Ra=`conversations`,za=1,Ba=2,Va=[`id`,`modelId`,`modelRevision`,`inferenceId`,`modelName`,`prompt`,`content`,`reasoning`,`tools`,`status`,`finishReason`,`usage`,`ttftMs`,`elapsedMs`,`error`,`parameters`,`images`],Ha=[...Va,`startedAt`],Ua=new Set([`temperature`,`top_p`,`top_k`,`min_p`,`max_tokens`,`seed`,`repetition_penalty`,`presence_penalty`,`frequency_penalty`]),Wa=class extends Error{constructor(e=`Conversation history is invalid or exceeds the local history limits.`){super(e),this.name=`HistoryValidationError`}},Ga=class extends Error{constructor(e,t){super(e,t),this.name=`HistoryStorageError`}};function Ka(){throw new Wa}function qa(e,t){if(!e||typeof e!=`object`||Array.isArray(e))return Ka();let n=Object.getPrototypeOf(e);if(n!==Object.prototype&&n!==null)return Ka();let r=Object.keys(e);return r.length!==t.length||r.some(e=>!t.includes(e))?Ka():e}function Ja(e,t=Fa.textCharacters){return typeof e!=`string`||e.length>t?Ka():e}function Ya(e){let t=Ja(e,512);return!t||Array.from(t).some(e=>e.charCodeAt(0)<32||e.charCodeAt(0)===127)?Ka():t}function Xa(e,t=!1){return typeof e!=`number`||!Number.isFinite(e)||e<0||t&&!Number.isSafeInteger(e)?Ka():e}var Za=864e13;function Qa(e){let t=Xa(e,!0);return t>Za?Ka():t}function $a(e,t){return e===null?null:t(e)}function eo(e,t){return!Array.isArray(e)||e.length>t?Ka():e}function to(e){return new TextEncoder().encode(e).byteLength}function no(e,t,n){let r=0,i=0,a=0,o=new Set;return eo(e,Fa.conversations).map(e=>{let s=qa(e,[`id`,`title`,`systemPrompt`,`turns`,`updatedAt`]),c=Ya(s.id);if(o.has(c))return Ka();o.add(c);let l=new Set,u=eo(s.turns,Fa.turnsPerConversation).map(e=>{if(++r>Fa.totalTurns)return Ka();let o=qa(e,n===1?Va:Ha),s=Ya(o.id);if(l.has(s))return Ka();l.add(s);let c=Ja(o.status,32);if(![`streaming`,`complete`,`cancelled`,`interrupted`,`error`].includes(c))return Ka();let u={};if(!o.parameters||typeof o.parameters!=`object`||Array.isArray(o.parameters)||![Object.prototype,null].includes(Object.getPrototypeOf(o.parameters)))return Ka();for(let[e,t]of Object.entries(o.parameters)){if(!Ua.has(e)||typeof t!=`number`||!Number.isFinite(t))return Ka();u[e]=t}let d=new Set,f=eo(o.tools,Fa.toolsPerTurn).map(e=>{let t=qa(e,[`index`,`id`,`name`,`arguments`]),n=Xa(t.index,!0);return d.has(n)?Ka():(d.add(n),{index:n,id:Ja(t.id,512),name:Ja(t.name,512),arguments:Ja(t.arguments)})}),p=eo(o.images,Fa.imagesPerTurn).map(e=>{let t=qa(e,[`name`,`type`,`dataUrl`]),n=Ja(t.type,64);if(![`image/png`,`image/jpeg`,`image/webp`].includes(n))return Ka();let r=Ja(t.dataUrl,Fa.imageBytes*2),a=`data:${n};base64,`;if(!r.startsWith(a))return Ka();let o=r.slice(a.length);return!o||o.length%4!=0||!/^[A-Za-z0-9+/]*={0,2}$/u.test(o)||(i+=o.length*3/4-(o.endsWith(`==`)?2:+!!o.endsWith(`=`)),i>Fa.imageBytes)?Ka():{name:Ja(t.name,512),type:n,dataUrl:r}}),m={id:s,modelId:Ya(o.modelId),modelRevision:Xa(o.modelRevision,!0),inferenceId:Ya(o.inferenceId),modelName:Ja(o.modelName,512),prompt:Ja(o.prompt),content:Ja(o.content),reasoning:Ja(o.reasoning),tools:f,status:c===`streaming`?`interrupted`:c,finishReason:$a(o.finishReason,Ja),usage:$a(o.usage,e=>{let t=qa(e,[`prompt_tokens`,`completion_tokens`,`total_tokens`]);return{prompt_tokens:Xa(t.prompt_tokens,!0),completion_tokens:Xa(t.completion_tokens,!0),total_tokens:Xa(t.total_tokens,!0)}}),ttftMs:$a(o.ttftMs,Xa),elapsedMs:$a(o.elapsedMs,Xa),error:$a(o.error,Ja),parameters:u,images:t.includeImages?p:[],startedAt:n===1?0:Qa(o.startedAt)};return m.status===`complete`&&(!m.finishReason||!m.content&&!m.reasoning&&!m.tools.length)||(a+=to(JSON.stringify(m)),a>Fa.jsonBytes)?Ka():m});return{id:c,title:Ja(s.title,512),systemPrompt:Ja(s.systemPrompt),turns:u,updatedAt:Xa(s.updatedAt,!0)}})}function ro(e,t={}){return io(e,t,Ba)}function io(e,t,n){let r=no(e,t,n);return to(JSON.stringify(r))>Fa.jsonBytes?Ka():r}function ao(e,t={}){let n=JSON.stringify({version:Ba,conversations:ro(e,t)});return to(n)>Fa.jsonBytes?Ka():n}function oo(e,t={}){if(to(e)>Fa.jsonBytes)return Ka();let n;try{n=JSON.parse(e)}catch{return Ka()}let r=qa(n,[`version`,`conversations`]);return r.version!==1&&r.version!==2?Ka():io(r.conversations,t,r.version)}function so(e){return new Ga(e instanceof DOMException&&e.name===`QuotaExceededError`?`Local history storage is full. Export or clear history, or disable persistence; the current conversation remains in memory.`:`Local history storage is unavailable. The current conversation remains in memory.`,{cause:e})}function co(e={}){let t=!1,n=!1,r=0,i,a,o=new Set,s=()=>{let t=e.indexedDB??globalThis.indexedDB;if(!t)throw new Ga(`This browser does not support local history storage.`);return t},c=()=>{r++;for(let e of o)try{e.abort()}catch{}o.clear(),i?.close(),i=void 0,a=void 0},l=()=>{if(i)return Promise.resolve(i);if(a)return a;let e=r;return a=new Promise((n,a)=>{let o=s().open(Ia,za);o.onupgradeneeded=()=>{o.result.objectStoreNames.contains(La)||o.result.createObjectStore(La)},o.onerror=()=>a(so(o.error));let l=!1;o.onblocked=()=>{l=!0,a(new Ga(`Close other WebUI tabs to access local history.`))},o.onsuccess=()=>{if(l||!t||e!==r){o.result.close(),a(new Ga(`Local history persistence was disabled.`));return}i=o.result,i.onversionchange=c,n(i)}}),a.catch(t=>{throw e===r&&(a=void 0),t})};async function u(e,i){if(!t||n)return;let a=r;try{let s=await l();return!t||n||r!==a?void 0:await new Promise((t,n)=>{let r=s.transaction(La,e);o.add(r);let a=i(r.objectStore(La));r.oncomplete=()=>{o.delete(r),t(a.result)},r.onabort=r.onerror=()=>{o.delete(r),n(so(r.error??a.error))}})}catch(e){throw e instanceof Ga?e:so(e)}}return{setEnabled(e){t!==e&&(t=e,e||c())},async load(e={}){let t=await u(`readonly`,e=>e.get(Ra));return t===void 0?[]:oo(Ja(t,Fa.jsonBytes),e)},async save(e,n={}){if(!t)return;let r=ao(e,n);await u(`readwrite`,e=>e.put(r,Ra))},async clear(){if(n)throw new Ga(`Local history is already being cleared.`);n=!0;let e=!1;c();try{await new Promise((t,r)=>{let i=s().deleteDatabase(Ia);i.onsuccess=()=>{n=!1,t()},i.onerror=()=>{n=!1,r(so(i.error))},i.onblocked=()=>{e=!0,r(new Ga(`Close other WebUI tabs to clear local history.`))}})}catch(t){throw e||(n=!1),t instanceof Ga?t:so(t)}},close:c}}var lo=[],uo=0,fo=0,po=0,mo=new Set;function ho(e){return mo.add(e),()=>{mo.delete(e)}}function go(){return uo}function _o(e){return{id:crypto.randomUUID(),title:e,systemPrompt:``,turns:[],updatedAt:Date.now()}}var vo=new WeakMap;function yo(e){let t=vo.get(e);if(t!==void 0)return t;let n=new TextEncoder().encode(JSON.stringify(e)).byteLength;return vo.set(e,n),n}function bo(e){return e.length>Fa.conversations||e.reduce((e,t)=>e+t.turns.length,0)>Fa.totalTurns?!1:2+Math.max(0,e.length-1)+e.reduce((e,t)=>e+new TextEncoder().encode(JSON.stringify({...t,turns:[]})).byteLength+t.turns.reduce((e,t)=>e+yo(t),0)+Math.max(0,t.turns.length-1),0)<=Fa.jsonBytes}function xo(){for(let e of mo)e()}function So(e){lo=e,xo()}function Co(e){if(!bo(e))throw Error(`Conversation memory limit exceeded.`);uo++,fo=0,So(e)}function wo(){po++,fo=po,xo()}function To(){return(0,_.useSyncExternalStore)(ho,()=>fo)}function Eo(){let e=fo!==0;return fo=0,e}function Do(e){let t=lo.some(t=>t.id===e.id)?lo.map(t=>t.id===e.id?e:t):[...lo,e];return bo(t)?(So(t),!0):!1}function Oo(){return(0,_.useSyncExternalStore)(ho,()=>lo)}var ko=(0,_.lazy)(()=>Si(()=>import(`./code-highlight-CtSU6YzY.js`),[],import.meta.url)),Ao=65536,jo=512;function Mo(e){let t=[];for(let n=0;n<e.length;n+=1024)t.push((0,x.jsx)(`span`,{children:No(e.slice(n,n+1024))},n));return t}function No(e){let t=[],n=/!\[([^\]\n]*)\]\(([^)\n]*)\)|\[([^\]\n]*)\]\(([^)\n]*)\)|`([^`\n]+)`|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*/g,r=0;for(let i of e.matchAll(n)){t.push(e.slice(r,i.index));let n=i.index;if(i[1]!==void 0)t.push((0,x.jsxs)(`span`,{children:[`[Image omitted: `,i[1]||`image`,`]`]},n));else if(i[3]!==void 0){let e;try{let t=new URL(i[4]);/^https?:$/.test(t.protocol)&&!t.username&&!t.password&&(e=t.href)}catch{}t.push(e?(0,x.jsx)(`a`,{href:e,target:`_blank`,rel:`noopener noreferrer`,children:i[3]},n):(0,x.jsx)(`span`,{children:i[3]},n))}else i[5]===void 0?i[6]===void 0?t.push((0,x.jsx)(`em`,{children:i[7]},n)):t.push((0,x.jsx)(`strong`,{children:i[6]},n)):t.push((0,x.jsx)(`code`,{children:i[5]},n));r=i.index+i[0].length}return t.push(e.slice(r)),t}var Po=(0,_.memo)(function({code:e,language:t,complete:n}){let[r,i]=(0,_.useState)(`Copy code`);async function a(){try{await navigator.clipboard.writeText(e),i(`Copied`)}catch{i(`Copy failed`)}}return(0,x.jsxs)(`div`,{className:`chat-code-block`,children:[(0,x.jsxs)(`div`,{className:`chat-code-toolbar`,children:[(0,x.jsx)(`span`,{children:t||`Plain text`}),(0,x.jsx)(P,{tone:`ghost`,onClick:()=>void a(),children:r})]}),(0,x.jsx)(`pre`,{children:n?(0,x.jsx)(_.Suspense,{fallback:(0,x.jsx)(`code`,{children:e}),children:(0,x.jsx)(ko,{code:e,language:t})}):(0,x.jsx)(`code`,{children:e})})]})}),Fo=(0,_.memo)(function({text:e,streaming:t=!1}){let n=e.slice(0,Ao).split(`
`),r=[],i=0;for(;i<n.length&&r.length<jo;){let a=i,o=n[i++],s=/^ {0,3}(`{3,}|~{3,})([\w+-]*)\s*$/.exec(o);if(s){let o=[],c=!1;for(;i<n.length;){let e=n[i++],t=e.trim();if(t.length>=s[1].length&&[...t].every(e=>e===s[1][0])){c=!0;break}o.push(e)}r.push((0,x.jsx)(Po,{code:o.join(`
`),language:s[2].slice(0,40),complete:c||!t&&e.length<=Ao},a))}else if(/^#{1,6} /.test(o))r.push((0,x.jsx)(`p`,{className:`chat-markdown-heading`,children:(0,x.jsx)(`strong`,{children:Mo(o.replace(/^#{1,6} /,``))})},a));else if(/^[-*] /.test(o)){let e=[o.slice(2)];for(;i<n.length&&/^[-*] /.test(n[i])&&e.length<jo;)e.push(n[i++].slice(2));r.push((0,x.jsx)(`ul`,{children:e.map((e,t)=>(0,x.jsx)(`li`,{children:Mo(e)},t))},a))}else o.trim()&&r.push((0,x.jsx)(`p`,{children:Mo(o)},a))}return(0,x.jsxs)(`div`,{className:`chat-markdown`,children:[r,(e.length>Ao||i<n.length)&&(0,x.jsx)(`p`,{children:`Display truncated for performance. Copy the message to read its full text.`})]})});function Io(e){let[t,n]=(0,_.useState)(()=>Date.now());return(0,_.useEffect)(()=>{let t=window.setInterval(()=>n(Date.now()),e);return()=>window.clearInterval(t)},[e]),t}var Lo=[[`year`,31536e6],[`month`,2592e6],[`week`,6048e5],[`day`,864e5],[`hour`,36e5],[`minute`,6e4]],Ro=new Map,G=new Map;function K(e,t,n){let r=e.get(t);if(r!==void 0)return r;let i=n(ua(t));return e.set(t,i),i}function zo(e,t,n){let r=K(Ro,n,e=>new Intl.RelativeTimeFormat(e,{numeric:`auto`})),i=e-t;for(let[e,t]of Lo)if(Math.abs(i)>=t)return r.format(Math.round(i/t),e);return r.format(0,`second`)}function Bo(e){return!Number.isNaN(new Date(e).getTime())}function Vo(e,t){return Bo(e)?K(G,t,e=>new Intl.DateTimeFormat(e,{hour:`numeric`,minute:`2-digit`})).format(e):``}function Ho(e){return Bo(e)?new Date(e).toISOString():void 0}var Uo={streaming:`chat.transcript.status.streaming`,complete:`chat.transcript.status.complete`,cancelled:`chat.transcript.status.cancelled`,interrupted:`chat.transcript.status.interrupted`,error:`chat.transcript.status.error`};function Wo(e,t){return t&&(e.status===`error`||e.status===`interrupted`)}function Go({at:e,locale:t}){let n=e>0?Ho(e):void 0;return n===void 0?null:(0,x.jsx)(`time`,{className:`chat-time`,dateTime:n,children:Vo(e,t)})}function Ko({startedAt:e,locale:t}){let n=Io(1e3),r=e>0?Math.max(0,Math.floor((n-e)/1e3)):0;return(0,x.jsx)(ct,{children:B(t,`chat.message.elapsed`,{status:B(t,Uo.streaming),seconds:String(r)})})}function qo(){let[e,t]=(0,_.useState)(null);return[e,(e,n,r)=>{(async()=>{try{await navigator.clipboard.writeText(e),t(n)}catch{t(r)}})()}]}function Jo({turn:e,index:t,busy:n,locale:r,onEdit:i}){let[a,o]=qo();return(0,x.jsxs)(`div`,{className:`chat-message chat-message--user`,"data-role":`user`,children:[(0,x.jsxs)(`div`,{className:`chat-message-meta`,children:[(0,x.jsx)(`span`,{className:`chat-role`,children:B(r,`chat.message.you`)}),(0,x.jsx)(Go,{at:e.startedAt,locale:r})]}),(0,x.jsxs)(`div`,{className:`chat-bubble chat-bubble--user`,children:[(0,x.jsx)(`p`,{className:`chat-prompt`,children:e.prompt}),e.images.length>0?(0,x.jsx)(`p`,{className:`chat-attachments`,children:B(r,`chat.transcript.images`,{count:String(e.images.length)})}):null]}),(0,x.jsxs)(`div`,{className:`chat-message-tools`,children:[(0,x.jsxs)(`div`,{className:`chat-message-actions`,role:`group`,"aria-label":B(r,`chat.message.prompt_actions`),children:[(0,x.jsx)(F,{label:B(r,`chat.message.copy_prompt`),icon:`copy`,onClick:()=>o(e.prompt,`chat.message.copied_prompt`,`chat.message.copy_prompt_failed`)}),(0,x.jsx)(F,{label:B(r,`chat.transcript.edit`),icon:`edit`,disabled:n,onClick:()=>i(t)})]}),(0,x.jsx)(`span`,{className:`chat-copy-status`,role:`status`,children:a===null?``:B(r,a)})]})]})}function Yo({turn:e,last:t,busy:n,canSend:r,locale:i,onRetry:a}){let[o,s]=qo(),[c,l]=(0,_.useState)(!1),u=(0,_.useId)(),d=e.status===`streaming`,f=Pa(e),p=B(i,`chat.transcript.unknown`),m=e.startedAt>0&&e.elapsedMs!==null?e.startedAt+Math.round(e.elapsedMs):e.startedAt;return(0,x.jsxs)(`div`,{className:`chat-message chat-message--assistant`,"data-role":`assistant`,children:[(0,x.jsxs)(`div`,{className:`chat-message-meta`,children:[(0,x.jsx)(ct,{tone:`accent`,children:e.modelName}),(0,x.jsx)(`span`,{className:`chat-status`,"data-status":e.status,children:d?(0,x.jsx)(Ko,{startedAt:e.startedAt,locale:i}):(0,x.jsx)(ct,{children:B(i,Uo[e.status])})}),(0,x.jsx)(Go,{at:m,locale:i})]}),(0,x.jsxs)(`div`,{className:`chat-bubble chat-bubble--assistant`,children:[e.reasoning?(0,x.jsxs)(`details`,{className:`chat-reasoning`,children:[(0,x.jsx)(`summary`,{children:d&&!e.content?B(i,`chat.transcript.reasoning_thinking`):B(i,`chat.reasoning`)}),(0,x.jsx)(`p`,{className:`chat-reasoning-text`,children:e.reasoning})]}):null,d&&!e.content?(0,x.jsx)(`p`,{className:`chat-waiting`,children:e.reasoning?B(i,`chat.transcript.thinking`):B(i,`chat.transcript.waiting`)}):null,(0,x.jsx)(Fo,{text:e.content,streaming:d}),e.tools.map(e=>{let t=e.name||B(i,`chat.transcript.tool_name_pending`);return(0,x.jsxs)(`div`,{className:`chat-tool`,role:`group`,"aria-label":B(i,`chat.transcript.tool_call`,{name:t}),children:[(0,x.jsxs)(`div`,{className:`chat-tool-head`,children:[(0,x.jsx)(`span`,{className:`chat-tool-label`,children:B(i,`chat.tool_call`)}),(0,x.jsx)(`code`,{children:t}),(0,x.jsx)(ct,{children:B(i,`chat.message.not_executed`)})]}),(0,x.jsx)(`pre`,{children:e.arguments})]},e.index)}),e.error?(0,x.jsx)(`p`,{className:`chat-turn-error`,role:`alert`,children:e.error}):null]}),(0,x.jsxs)(`div`,{className:`chat-message-tools`,children:[(0,x.jsxs)(`div`,{className:`chat-message-actions`,role:`group`,"aria-label":B(i,`chat.message.response_actions`),children:[(0,x.jsx)(F,{label:B(i,`chat.transcript.copy`),icon:`copy`,onClick:()=>s(e.content,`chat.transcript.copied`,`chat.transcript.copy_failed`)}),Wo(e,t)?(0,x.jsx)(F,{label:B(i,`chat.message.retry`),icon:`retry`,disabled:n||!r,onClick:()=>a(e.id)}):null,(0,x.jsx)(F,{label:B(i,`chat.transcript.details`),icon:`details`,"aria-expanded":c,"aria-controls":u,onClick:()=>l(!c)})]}),(0,x.jsx)(`span`,{className:`chat-copy-status`,role:`status`,children:o===null?``:B(i,o)})]}),(0,x.jsxs)(`div`,{className:`chat-details`,id:u,hidden:!c,children:[(0,x.jsxs)(`dl`,{className:`chat-metrics`,children:[(0,x.jsx)(`dt`,{children:B(i,`chat.transcript.finish_reason`)}),(0,x.jsx)(`dd`,{children:e.finishReason??p}),(0,x.jsx)(`dt`,{children:B(i,`chat.transcript.usage`)}),(0,x.jsx)(`dd`,{children:e.usage?B(i,`chat.transcript.usage_value`,{prompt:String(e.usage.prompt_tokens),completion:String(e.usage.completion_tokens),total:String(e.usage.total_tokens)}):p}),(0,x.jsx)(`dt`,{children:B(i,`chat.transcript.first_delta`)}),(0,x.jsx)(`dd`,{children:e.ttftMs===null?p:`${Math.round(e.ttftMs)} ms`}),(0,x.jsx)(`dt`,{children:B(i,`chat.transcript.decode_rate`)}),(0,x.jsx)(`dd`,{children:f===null?p:`${f.toFixed(1)} tokens/s`})]}),(0,x.jsx)(`p`,{children:B(i,`chat.transcript.metrics_note`)}),(0,x.jsx)(`pre`,{children:JSON.stringify(e.parameters,null,2)})]})]})}var Xo=(0,_.memo)(function({turn:e,index:t,last:n,busy:r,canSend:i,locale:a,onEdit:o,onRetry:s}){return(0,x.jsxs)(`article`,{className:`chat-turn`,"aria-label":B(a,`chat.transcript.turn_label`,{model:e.modelName}),children:[(0,x.jsx)(Jo,{turn:e,index:t,busy:r,locale:a,onEdit:o}),(0,x.jsx)(Yo,{turn:e,last:n,busy:r,canSend:i,locale:a,onRetry:s})]})});function Zo({turns:e,onEdit:t,onRetry:n,busy:r,canSend:i=!0,locale:a}){let o=(0,_.useRef)(t);o.current=t;let s=(0,_.useRef)(n);s.current=n;let c=(0,_.useCallback)(e=>o.current(e),[]),l=(0,_.useCallback)(e=>s.current(e),[]),u=(0,_.useRef)(null),d=(0,_.useRef)(!0),[f,p]=(0,_.useState)(!1),m=e.at(-1);return(0,_.useEffect)(()=>{d.current&&u.current&&window.getSelection()?.isCollapsed!==!1&&(u.current.scrollTop=u.current.scrollHeight)},[m?.content,m?.reasoning,e.length]),(0,x.jsxs)(`section`,{className:`chat-transcript`,"aria-label":B(a,`chat.transcript.label`),children:[(0,x.jsx)(`div`,{className:`chat-messages`,ref:u,tabIndex:0,onScroll:()=>{let e=u.current;e&&(d.current=e.scrollHeight-e.scrollTop-e.clientHeight<80,p(!d.current))},children:e.length?e.map((t,n)=>(0,x.jsx)(Xo,{turn:t,index:n,last:n===e.length-1,busy:r,canSend:i,locale:a,onEdit:c,onRetry:l},t.id)):(0,x.jsx)(`p`,{className:`chat-empty`,children:B(a,`chat.transcript.empty`)})}),f?(0,x.jsx)(P,{className:`chat-jump`,onClick:()=>{d.current=!0,u.current&&(u.current.scrollTop=u.current.scrollHeight),p(!1)},children:B(a,`chat.transcript.jump`)}):null]})}var Qo=3e4,$o=(0,_.memo)(function({item:e,current:t,disabled:n,now:r,locale:i,onSelect:a,onRename:o,onDelete:s}){let[c,l]=(0,_.useState)(!1),[u,d]=(0,_.useState)(``),f=(0,_.useRef)(!1),p=(0,_.useRef)(!1),m=(0,_.useRef)(null),h=e.turns.at(-1),g=()=>{f.current=!0,d(e.title),l(!0)},v=(t,n)=>{f.current&&(f.current=!1,l(!1),t&&o(e.id,u),n&&window.setTimeout(()=>m.current?.focus(),0))};return(0,x.jsxs)(`li`,{className:`chat-row`,"data-current":t||void 0,children:[c?(0,x.jsx)(`input`,{className:`chat-row-rename`,"aria-label":B(i,`chat.list.rename_field`,{title:e.title}),value:u,maxLength:120,autoFocus:!0,onChange:e=>d(e.currentTarget.value),onCompositionStart:()=>{p.current=!0},onCompositionEnd:()=>{p.current=!1},onBlur:()=>v(!0,!1),onKeyDownCapture:e=>{(e.key===`Escape`||e.key===`Enter`)&&(p.current||e.nativeEvent.isComposing||e.keyCode===229||(e.preventDefault(),e.stopPropagation(),v(e.key===`Enter`,!0)))}}):(0,x.jsxs)(`button`,{ref:m,type:`button`,className:`chat-row-select`,"aria-current":t?`true`:void 0,disabled:n,onClick:()=>a(e.id),children:[(0,x.jsx)(`span`,{className:`chat-row-title`,children:e.title}),(0,x.jsxs)(`span`,{className:`chat-row-meta`,children:[(0,x.jsx)(`span`,{children:h?h.modelName:B(i,`chat.list.no_messages`)}),(0,x.jsx)(`time`,{dateTime:Ho(e.updatedAt),children:zo(e.updatedAt,r,i)})]})]}),(0,x.jsxs)(`div`,{className:`chat-row-actions`,children:[(0,x.jsx)(F,{label:B(i,`chat.list.rename`,{title:e.title}),icon:`edit`,disabled:n||c,onClick:g}),(0,x.jsx)(F,{label:B(i,`chat.list.delete`,{title:e.title}),icon:`trash`,disabled:n,onClick:()=>s(e.id)})]})]})});function es(e){if(![`ArrowDown`,`ArrowUp`,`Home`,`End`].includes(e.key)||e.altKey||e.ctrlKey||e.metaKey||e.nativeEvent.isComposing)return;let t=Array.from(e.currentTarget.querySelectorAll(`button.chat-row-select:not(:disabled)`)),n=t.findIndex(t=>t===e.target);if(n===-1)return;let r=e.key===`Home`?0:e.key===`End`?t.length-1:Math.min(t.length-1,Math.max(0,n+(e.key===`ArrowDown`?1:-1)));e.preventDefault(),t[r]?.focus()}function ts(e){let t=Io(Qo),n=(0,_.useMemo)(()=>[...e.conversations].sort((e,t)=>t.updatedAt-e.updatedAt),[e.conversations]),r=(0,_.useRef)(e);r.current=e;let i=(0,_.useCallback)(e=>r.current.onSelect(e),[]),a=(0,_.useCallback)((e,t)=>r.current.onRename(e,t),[]),o=(0,_.useCallback)(e=>r.current.onDelete(e),[]);return(0,x.jsxs)(`div`,{className:`chat-list`,children:[e.atLimit?(0,x.jsx)(`p`,{className:`chat-list-limit`,"data-testid":`chat-list-limit`,children:B(e.locale,`chat.list.limit`)}):null,n.length===0?(0,x.jsx)(I,{title:B(e.locale,`chat.list.empty.title`),body:B(e.locale,`chat.list.empty.body`),testId:`chat-list-empty`}):(0,x.jsx)(`ul`,{className:`chat-list-rows`,"aria-label":B(e.locale,`chat.list.label`),ref:e.listRef,onKeyDown:es,children:n.map(n=>(0,x.jsx)($o,{item:n,current:n.id===e.currentId,disabled:e.disabled,now:t,locale:e.locale,onSelect:i,onRename:a,onDelete:o},n.id))})]})}var ns=1,rs=8;function is(e,t){let[n,r]=(0,_.useState)(ns);return(0,_.useLayoutEffect)(()=>{let t=e.current;if(!t)return;let n=window.getComputedStyle(t),i=Number.parseFloat(n.lineHeight),a=Number.parseFloat(n.paddingTop)+Number.parseFloat(n.paddingBottom);if(!Number.isFinite(i)||i<=0||!Number.isFinite(a))return;let o=t.rows;t.rows=ns;let s=Math.round((t.scrollHeight-a)/i);t.rows=o,r(Math.min(rs,Math.max(ns,s)))},[e,t]),n}function as(e){let{locale:t}=e,n=(0,_.useRef)(null),r=(0,_.useId)(),i=is(e.textareaRef,e.draft),a=e.draft.length>=Da,o=(0,_.useMemo)(()=>new Intl.NumberFormat(ua(t)),[t]);return(0,x.jsxs)(`div`,{className:`chat-composer`,children:[e.images.length?(0,x.jsx)(`div`,{className:`chat-images`,children:e.images.map((n,r)=>(0,x.jsxs)(`figure`,{children:[(0,x.jsx)(`img`,{src:n.dataUrl,alt:n.name}),(0,x.jsx)(P,{disabled:e.disabled,onClick:()=>e.onRemoveImage(r),children:B(t,`chat.images.remove`,{name:n.name})})]},`${n.name}-${r}`))}):null,(0,x.jsxs)(`div`,{className:`chat-composer-row`,children:[e.canImage?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(F,{label:B(t,`chat.images.attach`),icon:`attach`,disabled:e.disabled,onClick:()=>n.current?.click()}),(0,x.jsx)(`input`,{ref:n,className:`chat-file-input`,type:`file`,"aria-label":B(t,`chat.images.label`),accept:`image/png,image/jpeg,image/webp`,multiple:!0,disabled:e.disabled,onChange:t=>{let n=t.target.files;n&&e.onAddImages(n),t.target.value=``}})]}):null,(0,x.jsx)(`textarea`,{ref:e.textareaRef,rows:i,"aria-label":B(t,`chat.composer.label`),"aria-describedby":r,value:e.draft,maxLength:Da,disabled:e.disabled,onChange:t=>e.onDraftChange(t.target.value),onCompositionStart:e.onCompositionStart,onCompositionEnd:e.onCompositionEnd,onKeyDown:e.onKeyDown}),e.running?(0,x.jsx)(P,{onClick:e.onStop,children:B(t,`chat.stop`)}):(0,x.jsx)(P,{tone:`primary`,disabled:!e.canSend,onClick:e.onSend,children:B(t,`common.send`)})]}),(0,x.jsxs)(`div`,{className:`chat-composer-help`,children:[(0,x.jsxs)(`span`,{className:`chat-count`,id:r,"data-full":a||void 0,children:[B(t,`chat.composer.count`,{count:o.format(e.draft.length),max:o.format(Da)}),a?` · ${B(t,`chat.composer.limit`)}`:``]}),(0,x.jsx)(Ve,{content:B(t,`chat.composer.hint`),children:(0,x.jsx)(F,{label:B(t,`chat.composer.keys`),icon:`info`})})]})]})}var os=e=>e.capabilities.some(e=>e.task===`chat`),ss=e=>e.lifecycle.state===`ready`&&e.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available),cs=(e,t)=>e.identity.display_name.localeCompare(t.identity.display_name)||e.identity.id.localeCompare(t.identity.id);function ls(e,t,n){let r=e.filter(ss).sort(cs),i=e.filter(e=>[`loading`,`draining`,`unloading`].includes(e.lifecycle.state)&&os(e)).sort(cs),a=e.filter(e=>[`unloaded`,`failed`].includes(e.lifecycle.state)&&os(e)).sort(cs),o=(e,t)=>({value:e.identity.id,label:e.identity.display_name,description:t}),s=[...r.map(e=>o(e,B(n,`chat.model.group.ready`))),...i.map(e=>o(e,Vn(n,e.lifecycle.state))),...a.map(e=>o(e,B(n,e.lifecycle.state===`failed`?`chat.model.group.failed`:`chat.model.group.unloaded`)))];return t&&!s.some(e=>e.value===t.identity.id)&&s.push(o(t,B(n,`chat.model.group.unavailable`))),t?s:[{value:``,label:B(n,`chat.model.choose`)},...s]}function us(e,t){return t===void 0?`chat.model.hint.choose`:ss(t)?null:[`unloaded`,`failed`].includes(t.lifecycle.state)&&os(t)?Hi(e,t)?`chat.model.hint.load`:t.lifecycle.busy||Vi(e,t.identity.id)?`chat.model.hint.wait`:`chat.model.hint.cannot_load`:[`loading`,`draining`,`unloading`].includes(t.lifecycle.state)&&os(t)?`chat.model.hint.wait`:`chat.model.hint.unavailable`}function ds({locale:e}){let t=ci(),n=li(),r=t.catalog.find(e=>e.identity.id===t.selectedModelId),[i,a]=(0,_.useState)(null),[o,s]=(0,_.useState)(null),[c,l]=(0,_.useState)(!1),u=(0,_.useRef)(!1),[d,f]=(0,_.useState)(null),p=(0,_.useRef)(e);p.current=e;let{profile:m}=Ii((i??r)?.identity.id??null),{profile:h}=Ii(o?.kind===`capacity`?o.entry.identity.id:null),g=e=>s({kind:`capacity`,entry:e,instance:t.serverInstanceId}),v=e=>f({modelId:e,error:B(p.current,`models.library.stale`)}),y=(e,t)=>{u.current=!0,l(!0),f(null),t().then(()=>f({modelId:e.identity.id,error:null})).catch(t=>f({modelId:e.identity.id,error:ta(t,p.current)})).finally(()=>{u.current=!1,l(!1)})},b=()=>{let e=i;if(a(null),e===null||u.current)return;let r=t.catalog.find(t=>t.identity.id===e.identity.id);if(!r||r.identity.revision!==e.identity.revision||!Hi(t,r)){v(e.identity.id);return}y(r,()=>W(n,t,r,m,()=>g(r)))},S=(e,r)=>{let i=o;if(i===null||i.kind!==`capacity`||u.current)return;let a=t.catalog.find(e=>e.identity.id===i.entry.identity.id);if(i.instance!==t.serverInstanceId||!a||a.identity.revision!==i.entry.identity.revision){v(i.entry.identity.id),s(null);return}if(e===void 0||r===void 0||!Ki(t,a.identity.id).some(t=>t.identity.id===e&&t.identity.revision===r)){v(a.identity.id);return}s(null),y(a,()=>W(n,t,a,h,()=>g(a),{id:e,revision:r}))},C=(0,_.useMemo)(()=>ls(t.catalog,r,e),[t.catalog,r,e]),w=us(t,r),T=r!==void 0&&[`unloaded`,`failed`].includes(r.lifecycle.state)&&os(r)?r:void 0,E=d!==null&&r!==void 0&&d.modelId===r.identity.id?d:null,D=E!==null&&E.error===null&&r!==void 0&&(r.lifecycle.state===`loading`||r.lifecycle.busy||Vi(t,r.identity.id));return(0,x.jsxs)(`div`,{className:`chat-model`,children:[(0,x.jsx)(Me,{locale:e,label:B(e,`chat.model.label`),value:r?.identity.id??``,options:C,onChange:e=>n.selectModel(e||null),testId:`chat-model-picker`}),T?(0,x.jsx)(P,{tone:`primary`,busy:c,disabled:c||!Hi(t,T),"data-testid":`chat-load`,onClick:()=>a(T),children:B(e,`models.load`)}):null,(0,x.jsx)(`p`,{className:`chat-model-status`,role:`status`,"data-testid":`chat-model-status`,children:D?B(e,`chat.load.requested`):w&&!E?.error?(0,x.jsxs)(x.Fragment,{children:[B(e,w),w===`chat.model.hint.cannot_load`?(0,x.jsxs)(x.Fragment,{children:[` `,(0,x.jsx)(`a`,{href:`#models`,children:B(e,`chat.no_model.action`)})]}):null]}):null}),E?.error?(0,x.jsx)(`p`,{className:`chat-model-status chat-model-error`,role:`alert`,children:E.error}):null,i?(0,x.jsx)(pt,{open:!0,title:B(e,`chat.load.confirm.title`,{model:i.identity.display_name}),body:[B(e,`chat.load.confirm.body`),Object.keys(m).length?B(e,`models.next_profile.body`):``].filter(Boolean).join(` `),confirmLabel:B(e,`models.load`),cancelLabel:B(e,`common.cancel`),closeLabel:B(e,`common.close`),testId:`chat-load-dialog`,onConfirm:b,onClose:()=>a(null)}):null,o?(0,x.jsx)(Qi,{value:o,state:t,locale:e,busy:c,onClose:()=>s(null),onConfirm:S}):null]})}function fs(e,t,n){let r=(0,_.useRef)(0);(0,_.useEffect)(()=>{if(!e||t===r.current)return;let i=0,a=0,o=0,s=()=>{let e=n.current;if(e){if(document.activeElement!==e&&e.focus(),document.activeElement===e){r.current=t,o++===0&&typeof e.scrollIntoView==`function`&&e.scrollIntoView({block:`start`}),o<3&&(i=requestAnimationFrame(s));return}a++<20&&(i=requestAnimationFrame(s))}};return i=requestAnimationFrame(s),()=>cancelAnimationFrame(i)},[e,t,n])}function ps(e){let{locale:t,conversation:n}=e,r=(0,_.useId)(),i=(0,_.useId)(),a=(0,_.useId)(),o=(0,_.useRef)(null);fs(e.open,e.parametersRequest,o);let[s,c=``]=B(t,`chat.settings.sampling`).split(`{link}`);return(0,x.jsx)(Re,{open:e.open,onClose:e.onClose,title:B(t,`chat.settings.title`),closeLabel:B(t,`common.close`),testId:`chat-settings-drawer`,width:`medium`,side:`end`,children:(0,x.jsxs)(`div`,{className:`chat-settings`,children:[(0,x.jsxs)(`section`,{className:`chat-settings-section`,"aria-labelledby":r,children:[(0,x.jsx)(`h3`,{id:r,children:B(t,`chat.settings.summary`)}),n?(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(lt,{label:B(t,`chat.settings.name`),value:n.title,disabled:e.disabled,onChange:t=>e.onConversationChange({...n,title:t.slice(0,120)})}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:B(t,`chat.settings.system_prompt`)}),(0,x.jsx)(`textarea`,{value:n.systemPrompt,maxLength:Da,disabled:e.disabled,onChange:t=>e.onConversationChange({...n,systemPrompt:t.target.value})})]}),(0,x.jsxs)(`p`,{children:[s,(0,x.jsx)(`a`,{href:`#settings/requests`,children:B(t,`nav.settings`)}),c]})]}):(0,x.jsx)(`p`,{className:`chat-settings-hint`,children:B(t,`chat.settings.none`)})]}),(0,x.jsxs)(`section`,{className:`chat-settings-section`,"aria-labelledby":i,children:[(0,x.jsx)(`h3`,{id:i,ref:o,tabIndex:-1,children:B(t,`chat.params.summary`)}),e.parameters]}),(0,x.jsxs)(`section`,{className:`chat-settings-section`,"aria-labelledby":a,children:[(0,x.jsx)(`h3`,{id:a,children:B(t,`chat.privacy.summary`)}),e.history]})]})})}function ms({conversations:e,busy:t,onPending:n,limits:r,onReplace:i,locale:a}){let o=(0,_.useRef)(co()),[s,c]=(0,_.useState)(!1),l=(0,_.useRef)(t);l.current=t;let u=e=>{c(e),n(e)},[d,f]=(0,_.useState)(!1),[p,m]=(0,_.useState)(!1),[h,g]=(0,_.useState)(null),[v,y]=(0,_.useState)(null),b=(0,_.useRef)(null),S=e=>{b.current=e,y(e)},C=(0,_.useRef)(null),[w,T]=(0,_.useState)(!1),E=(0,_.useRef)(0);(0,_.useEffect)(()=>{if(!d)return;let t=setTimeout(()=>{o.current.save(e,{includeImages:p}).catch(()=>g(`chat.privacy.save_failed`))},300);return()=>clearTimeout(t)},[e,d,p]),(0,_.useEffect)(()=>{let e=o.current;return()=>{E.current++,e.close(),n(!1)}},[]),(0,_.useEffect)(()=>{if(!w||s||t)return;T(!1);let e=document.activeElement;(e===null||e===document.body)&&C.current?.focus()},[w,s,t]);let D=async(t,n)=>{let a=++E.current;if(o.current.setEnabled(t),!t){f(!1);return}u(!0);try{let t=await o.current.load({includeImages:p});if(await hs(t,r),a!==E.current||l.current)return;if(t.length&&e.length){S({kind:`replace-saved`,saved:t,operation:a,trigger:n});return}t.length&&i(t),f(!0)}catch{o.current.setEnabled(!1),a===E.current&&g(`chat.privacy.open_failed`)}finally{a===E.current&&u(!1)}},O=()=>{let e=++E.current;u(!0),f(!1),m(!1),o.current.setEnabled(!1),o.current.clear().then(()=>{e===E.current&&!l.current&&(i([]),g(`chat.privacy.cleared`))}).catch(()=>{e===E.current&&g(`chat.privacy.clear_failed`)}).finally(()=>{e===E.current&&(u(!1),T(!0))})},k=e=>{let t=b.current;if(b.current=null,y(null),t===null)return;if(t.kind===`clear`){e&&!l.current&&O();return}if(t.operation!==E.current)return;if(t.kind===`replace-saved`){if(!e)o.current.setEnabled(!1);else if(!l.current)try{i(t.saved),f(!0)}catch{o.current.setEnabled(!1),g(`chat.privacy.open_failed`)}}else if(e&&!l.current)try{i(t.imported)}catch{g(`chat.privacy.import_rejected`)}let n=t.trigger;window.setTimeout(()=>{let e=document.activeElement;n.isConnected&&!n.matches(`:disabled`)&&(e===null||e===document.body)&&n.focus()},0)},A=(e,t,n,r,i)=>v?.kind===e?(0,x.jsx)(pt,{open:!0,title:B(a,t),body:B(a,n),confirmLabel:B(a,r),cancelLabel:B(a,`common.cancel`),closeLabel:B(a,`common.close`),tone:`danger`,testId:i,onConfirm:()=>k(!0),onClose:()=>k(!1)}):null;return(0,x.jsxs)(x.Fragment,{children:[(0,x.jsxs)(`div`,{className:`chat-privacy`,children:[(0,x.jsx)(`p`,{children:B(a,`chat.privacy.body`)}),(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:d,disabled:t||s,onChange:e=>{D(e.target.checked,e.currentTarget)}}),B(a,`chat.privacy.save`)]}),(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:p,disabled:t||s,onChange:e=>m(e.target.checked)}),B(a,`chat.privacy.include_images`)]}),(0,x.jsx)(`p`,{children:B(a,`chat.privacy.note`)}),(0,x.jsxs)(`div`,{className:`chat-toolbar`,children:[(0,x.jsx)(P,{disabled:t||s,onClick:()=>{try{let t=new Blob([ao(e,{includeImages:p})],{type:`application/json`}),n=URL.createObjectURL(t),r=document.createElement(`a`);r.href=n,r.download=`mlxcel-conversations.json`,r.click(),setTimeout(()=>URL.revokeObjectURL(n),1e3)}catch{g(`chat.privacy.export_failed`)}},children:B(a,`chat.privacy.export`)}),(0,x.jsx)(P,{ref:C,disabled:t||s,onClick:()=>S({kind:`clear`}),children:B(a,`chat.privacy.clear`)})]}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[B(a,`chat.privacy.import`),(0,x.jsx)(`input`,{type:`file`,accept:`application/json,.json`,disabled:t||s,onChange:e=>{let t=e.currentTarget,n=e.target.files?.[0];if(e.target.value=``,!n)return;if(n.size>16777216){g(`chat.privacy.import_too_large`);return}let i=++E.current;u(!0),n.text().then(async e=>{let n=oo(e,{includeImages:p});await hs(n,r),!(i!==E.current||l.current)&&S({kind:`replace-import`,imported:n,operation:i,trigger:t})}).catch(()=>{i===E.current&&g(`chat.privacy.import_rejected`)}).finally(()=>{i===E.current&&u(!1)})}})]}),h?(0,x.jsx)(z,{tone:`info`,title:B(a,`chat.privacy.title`),body:B(a,h)}):null]}),v===null?null:(0,xe.createPortal)((0,x.jsxs)(x.Fragment,{children:[A(`replace-saved`,`chat.privacy.replace_saved.confirm.title`,`chat.privacy.replace_saved.confirm.body`,`chat.privacy.replace_saved.confirm`,`chat-replace-saved-dialog`),A(`clear`,`chat.privacy.clear.confirm.title`,`settings.clear_history.confirm.body`,`chat.privacy.clear.confirm`,`chat-clear-history-dialog`),A(`replace-import`,`chat.privacy.replace_import.confirm.title`,`chat.privacy.replace_import.confirm.body`,`chat.privacy.replace_import.confirm`,`chat-replace-import-dialog`)]}),document.body)]})}async function hs(e,t){for(let n of e)for(let e of n.turns)if(e.images.length){if(!t)throw Error(`Server image limits unavailable.`);e.images=await Ea(e.images.map(e=>{let t=atob(e.dataUrl.slice(e.dataUrl.indexOf(`,`)+1)),n=Uint8Array.from(t,e=>e.charCodeAt(0));return new File([n],e.name,{type:e.type})}),0,t)}}var gs=Object.freeze({}),_s=[`max_tokens`,`temperature`,`top_p`,`top_k`,`min_p`,`repetition_penalty`,`seed`],vs=new Set([`max_tokens`,`top_k`,`seed`]);function ys(e){if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Request defaults must be an object.`);let t={};for(let[n,r]of Object.entries(e)){if(!_s.includes(n))throw Error(`Unsupported request default: ${n}`);if(r!==void 0){if(typeof r!=`number`||!Number.isFinite(r))throw Error(`${n} must be a finite number; leave blank to inherit.`);if(vs.has(n)&&!Number.isSafeInteger(r))throw Error(`${n} must be a safe integer.`);if(n===`max_tokens`&&r<1||n===`temperature`&&r<0||n===`top_k`&&r<0||[`top_p`,`min_p`].includes(n)&&(r<0||r>1)||n===`repetition_penalty`&&r<=0||n===`seed`&&r<0)throw Error(`${n} is outside its supported range.`);t[n]=r}}return Object.freeze(t)}function bs(e,t,n,r){return typeof t==`number`&&Number.isFinite(t)?{value:t,source:`override`}:typeof n==`number`&&Number.isFinite(n)?{value:n,source:`session`}:r===null?{value:null,source:`server`}:typeof r!=`number`||!Number.isFinite(r)?{value:null,source:`unknown`}:vs.has(e)?Number.isSafeInteger(r)?{value:r,source:`server`}:{value:null,source:`unknown`}:{value:Number(Ot(r)),source:`server`}}var xs={override:`settings.requests.source.override`,session:`settings.requests.source.session`,server:`settings.requests.source.server`};function Ss(e,t){return t.source===`unknown`?B(e,`settings.requests.unknown`):B(e,`settings.requests.value_source`,{value:t.value===null?B(e,`settings.value.unset`):String(t.value),source:B(e,xs[t.source])})}var Cs=gs,ws=new Set;function Ts(e){return ws.add(e),()=>ws.delete(e)}function Es(e){Cs=ys(e);for(let e of ws)e()}function Ds(){Es(gs)}function Os(){return{defaults:(0,_.useSyncExternalStore)(Ts,()=>Cs,()=>gs),setDefaults:Es,reset:Ds}}function ks(e,t){let n=e.bootstrap?.server.server_instance_id;return n===void 0||t===null||t.lifecycle.state!==`ready`?null:{instance:n,modelId:t.identity.id,revision:t.identity.revision}}var As=32,js=new Map,Ms=new Set;function Ns(e){return Ms.add(e),()=>Ms.delete(e)}var Ps=e=>JSON.stringify([e.instance,e.modelId,e.revision]);function Fs(e){let t=new Set(e.schema.map(e=>e.name)),n={};for(let r of _s){let i=`default_${r}`;t.has(i)&&Object.hasOwn(e.current,i)&&(n[r]=e.current[i])}return Object.freeze(n)}function Is(e,t){let n=Ps(e);for(js.delete(n),js.set(n,Fs(t));js.size>As;){let e=js.keys().next();if(e.done)break;js.delete(e.value)}for(let e of Ms)e()}function Ls(e){let t=e===null?null:Ps(e);return(0,_.useSyncExternalStore)(Ns,()=>t===null?null:js.get(t)??null,()=>null)}function Rs(){let e=ci();return Ls(ks(e,e.catalog.find(t=>t.identity.id===e.selectedModelId)??null))}function zs(e,t){let n=Object.fromEntries(Object.entries(t).filter(([,e])=>e!==void 0&&e.trim()!==``).map(([e,t])=>[e,Number(t)]));return{...ys({...e,...n})}}function Bs({defaults:e,draft:t,onChange:n,locale:r}){let i=Rs();return(0,x.jsxs)(`div`,{className:`chat-parameters`,children:[(0,x.jsx)(`p`,{children:B(r,`chat.params.body`)}),(0,x.jsx)(`div`,{className:`chat-parameter-grid`,children:_s.map(a=>(0,x.jsx)(lt,{label:B(r,`chat.params.field`,{name:a}),value:t[a]??``,onChange:e=>n({...t,[a]:e}),hint:B(r,`chat.params.inherited`,{value:Ss(r,bs(a,void 0,e[a],i?.[a]))})},a))}),(0,x.jsx)(P,{onClick:()=>n({}),children:B(r,`chat.params.clear`)})]})}var Vs=new Set([`en`,`ko`].map(e=>B(e,`chat.conversation.default_title`))),Hs=50,Us=`(max-width: 960px)`;function Ws(e){if(typeof window.matchMedia!=`function`)return()=>void 0;let t=window.matchMedia(Us);return t.addEventListener(`change`,e),()=>t.removeEventListener(`change`,e)}function Gs(){return(0,_.useSyncExternalStore)(Ws,()=>typeof window.matchMedia==`function`&&window.matchMedia(Us).matches,()=>!1)}function Ks({locale:e}){let t=ci(),n=li(),{defaults:r}=Os(),[i,a]=(0,_.useState)({}),o=Oo(),[s,c]=(0,_.useState)(null),[l,u]=(0,_.useState)(``),[d,f]=(0,_.useState)([]),[p,m]=(0,_.useState)(null),[h,g]=(0,_.useState)(``),[v,y]=(0,_.useState)(!1),[b,S]=(0,_.useState)(!1),[C,w]=(0,_.useState)(!1),[T,E]=(0,_.useState)(null),[D,O]=(0,_.useState)(null),[k,A]=(0,_.useState)(!1),[j,ee]=(0,_.useState)(!1),[te,ne]=(0,_.useState)(0),M=Gs(),re=(0,_.useId)(),ie=(0,_.useRef)(e);ie.current=e;let N=(e,t)=>B(ie.current,e,t),ae=(0,_.useRef)(null),oe=(0,_.useRef)(null),se=(0,_.useRef)(null),ce=(0,_.useRef)(0),le=(0,_.useRef)(!1),ue=(0,_.useRef)(0),de=(0,_.useRef)(null),I=o.find(e=>e.id===s)??null,fe=t.catalog.find(e=>e.identity.id===t.selectedModelId),pe=[`ready`,`streaming`,`polling`].includes(t.connection)&&fe?.lifecycle.state===`ready`&&fe.capabilities.some(e=>e.task===`chat`&&e.phase===`provider_ready`&&e.available),L=pe&&t.bootstrap!==null&&Object.values(t.bootstrap.media_limits).every(e=>e>0)&&fe?.capabilities.some(e=>e.task===`vision_input`&&e.phase===`provider_ready`&&e.available),R=v||b||C,me=o.length>=Hs,he=M&&k;(0,_.useEffect)(()=>()=>{ce.current++,ue.current++;let e=de.current;e!==null&&(e.turn={...e.turn,status:`interrupted`,error:B(ie.current,`chat.error.view_closed`)},e.controller.abort(),e.flush())},[]),(0,_.useEffect)(()=>{!M&&k&&(A(!1),requestAnimationFrame(()=>{let e=document.activeElement;(!e||e===document.body||e.closest(`[aria-modal="true"]`))&&(oe.current?.querySelector(`[aria-current="true"]`)??oe.current?.querySelector(`button`))?.focus()}))},[M,k]);let ge=()=>{if(R||me)return;ce.current++;let e=_o(N(`chat.conversation.default_title`));Do(e),c(e.id),u(``),f([])},_e=To(),ve=(0,_.useRef)(ge);ve.current=ge,(0,_.useEffect)(()=>{Eo()&&ve.current()},[_e]);let ye=e=>{R||(e!==s&&(c(e),u(``),f([])),A(!1))},be=(e,t)=>{let n=o.find(t=>t.id===e);n!==void 0&&!R&&t.trim()&&Do({...n,title:t.slice(0,120)})},xe=()=>{let e=D;O(null),e!==null&&!R&&o.some(t=>t.id===e.id)&&(Co(o.filter(t=>t.id!==e.id)),s===e.id&&c(null),window.setTimeout(()=>{let e=document.activeElement;(e===null||e===document.body)&&(oe.current?.querySelector(`[aria-current="true"]`)??oe.current?.querySelector(`button`))?.focus()},0))},Se=()=>{let e=de.current;if(e===null)return;e.turn={...e.turn,status:`cancelled`,error:null},e.controller.abort(),e.flush();let t=++ue.current;g(N(`chat.announce.stopped`)),n.refreshRuntime(e.turn.modelId).then(e=>{ue.current===t&&g(N(`chat.announce.aborted_observed`,{time:e.measurements.active_requests?.measured_at??N(`chat.announce.unknown_time`)}))}).catch(()=>{ue.current===t&&g(N(`chat.announce.aborted_unobserved`))})},Ce=async(e={prompt:l,images:d,conversation:I,fromComposer:!0})=>{let{prompt:s,fromComposer:p}=e;if(de.current!==null||b||C||!pe||fe===void 0||!s.trim()||s.length>32e3)return;if((e.images.length||e.conversation?.turns.some(e=>e.images.length))&&!L){m(N(`chat.error.vision_unsupported`));return}if(e.conversation===null&&o.length>=Hs){m(N(`chat.error.conversation_limit`));return}let h;try{h=zs(r,i)}catch(e){m(e instanceof Error?e.message:N(`chat.error.invalid_parameters`));return}let _=e.conversation??_o(N(`chat.conversation.default_title`));if(_.turns.length>=100){m(N(`chat.error.turn_limit`));return}c(_.id);let v=new AbortController,x=performance.now(),S={id:crypto.randomUUID(),modelId:fe.identity.id,inferenceId:fe.identity.inference_id,modelName:fe.identity.display_name,modelRevision:fe.identity.revision,prompt:s,content:``,reasoning:``,tools:[],status:`streaming`,finishReason:null,usage:null,ttftMs:null,elapsedMs:null,error:null,parameters:h,images:e.images.map(e=>({...e})),startedAt:Date.now()};_={..._,title:_.turns.length===0&&Vs.has(_.title)?s.slice(0,80):_.title,turns:[..._.turns,S],updatedAt:Date.now()};try{if(!t.bootstrap)throw Error(`Limits unavailable`);Ta(_.turns.flatMap(e=>e.images),t.bootstrap.media_limits)}catch{m(N(`chat.error.images_exceed`));return}let w={...S.parameters,model:S.inferenceId,stream:!0,messages:Na(_.systemPrompt,_.turns),stream_options:{include_usage:!0}},T=Math.min(t.bootstrap?.media_limits.max_body_bytes??0,16777216);if(new TextEncoder().encode(JSON.stringify(w)).byteLength>T){m(N(`chat.error.body_limit`));return}if(!Do(_)){m(N(`chat.error.memory_budget`));return}let E=null,D=go(),O={controller:v,turn:S,conversation:_,flush:()=>{if(E!==null&&clearTimeout(E),E=null,go()!==D)return;let e=O.conversation;O.conversation={...O.conversation,turns:O.conversation.turns.map(e=>e.id===S.id?O.turn:e),updatedAt:Date.now()},Do(O.conversation)||(v.abort(),O.conversation=e,O.turn={...e.turns.find(e=>e.id===S.id)??S,status:`error`,error:null},O.conversation={...e,turns:e.turns.map(e=>e.id===S.id?O.turn:e)},Do(O.conversation),m(N(`chat.error.memory_budget_stream`)))}};a({}),de.current=O,ue.current++,y(!0),m(null),p&&(u(``),f([])),O.flush(),g(N(`chat.announce.generating`));try{await n.streamChatCompletions(S.modelId,w,{onFrame:e=>{v.signal.aborted||(O.turn=ja(O.turn,e.data,performance.now()-x),E===null&&(E=setTimeout(O.flush,50)))}},v.signal),v.signal.aborted||(O.turn=Ma(O.turn,performance.now()-x),g(N(`chat.announce.complete`)))}catch{v.signal.aborted||(v.abort(),O.turn={...O.turn,status:`error`,elapsedMs:performance.now()-x,error:N(`chat.error.generation_failed`)},g(N(`chat.announce.failed`)))}finally{O.flush(),de.current===O&&(de.current=null),y(!1)}},we=e=>{if(R||I===null)return;let t=I.turns.length-1,n=I.turns[t];n!==void 0&&n.id===e&&Wo(n,!0)&&Ce({prompt:n.prompt,images:n.images,conversation:{...I,turns:I.turns.slice(0,t)},fromComposer:!1})},Te=()=>{let e=T;if(E(null),e===null||R||I===null||I.id!==e.conversationId||I.turns[e.index]?.id!==e.turnId)return;let{index:t}=e;u(I.turns[t].prompt),f(I.turns[t].images),Do({...I,turns:I.turns.slice(0,t)}),window.setTimeout(()=>ae.current?.focus(),0)},Ee=e=>{if(!(le.current||e.nativeEvent.isComposing||e.keyCode===229)){if(va(e)&&!e.altKey&&!e.shiftKey&&e.key.toLowerCase()===`n`){e.preventDefault(),e.repeat||ge();return}e.key===`Enter`&&!e.shiftKey&&(e.preventDefault(),Ce())}},De=e=>{if(!t.bootstrap)return;let n=++ce.current;w(!0),Ea(e,d.length,t.bootstrap.media_limits,d).then(e=>{n===ce.current&&f(t=>[...t,...e])}).catch(()=>m(N(`chat.error.images_invalid`))).finally(()=>{n===ce.current&&w(!1)})},Oe=_s.filter(e=>(i[e]??``).trim()!==``),ke=(0,x.jsx)(ts,{locale:e,conversations:o,currentId:s,disabled:R,atLimit:me,onSelect:ye,onRename:be,onDelete:e=>{let t=o.find(t=>t.id===e);t&&!R&&O({id:e,title:t.title})},listRef:oe}),Ae=(0,x.jsx)(P,{onClick:ge,disabled:R||me,"data-testid":V(`chat.new_conversation`),children:B(e,`chat.new_conversation`)}),je=(0,x.jsxs)(x.Fragment,{children:[Oe.length?(0,x.jsx)(P,{tone:`ghost`,className:`chat-overrides`,"data-testid":`chat-overrides`,onClick:()=>{ne(e=>e+1),ee(!0)},children:B(e,`chat.overrides`,{keys:Oe.join(`, `)})}):null,M?(0,x.jsx)(F,{label:B(e,`chat.list.open`),icon:`list`,"aria-haspopup":`dialog`,"aria-expanded":he,onClick:()=>A(!0),"data-testid":`chat-list-open`}):null,(0,x.jsx)(F,{ref:se,label:B(e,`chat.settings.title`),icon:`settings`,"aria-haspopup":`dialog`,"aria-expanded":j,onClick:()=>ee(!0),"data-testid":`chat-settings-open`}),me?(0,x.jsx)(Ve,{content:B(e,`chat.list.limit`),children:Ae}):Ae]});return(0,x.jsxs)(`div`,{className:`chat-screen`,children:[(0,x.jsxs)(`div`,{className:`chat-body`,inert:he||j,children:[(0,x.jsx)(Ye,{title:B(e,`chat.title`),titleTestId:V(`chat.title`),description:B(e,`chat.intro`),actions:je}),(0,x.jsxs)(`div`,{className:`chat-panes`,children:[M?null:(0,x.jsx)(`nav`,{className:`chat-list-pane`,"aria-label":B(e,`chat.list.label`),children:ke}),(0,x.jsxs)(`section`,{className:`chat-conversation`,"aria-labelledby":re,children:[(0,x.jsxs)(`header`,{className:`chat-conversation-header`,children:[(0,x.jsx)(`h2`,{className:`chat-conversation-title`,id:re,children:I?.title??B(e,`chat.conversation.default_title`)}),(0,x.jsx)(ds,{locale:e})]}),(0,x.jsx)(Zo,{turns:I?.turns??[],onEdit:e=>{if(R||I===null)return;let t=I.turns[e];t&&E({conversationId:I.id,turnId:t.id,index:e})},onRetry:we,busy:R,canSend:pe,locale:e}),(0,x.jsxs)(`div`,{className:`chat-footer`,children:[p?(0,x.jsx)(z,{title:B(e,`chat.error.title`),body:p}):null,(0,x.jsx)(as,{locale:e,textareaRef:ae,draft:l,onDraftChange:u,images:d,onRemoveImage:e=>f(d.filter((t,n)=>e!==n)),canImage:!!L,onAddImages:De,disabled:R,running:v,canSend:pe&&!R&&l.trim()!==``,onSend:()=>{Ce()},onStop:Se,onKeyDown:Ee,onCompositionStart:()=>{le.current=!0},onCompositionEnd:()=>{le.current=!1}}),(0,x.jsx)(`p`,{className:`chat-announcement`,role:`status`,"aria-live":`polite`,"aria-atomic":`true`,children:h})]})]})]})]}),M?(0,x.jsx)(Re,{open:k,onClose:()=>A(!1),title:B(e,`chat.list.label`),closeLabel:B(e,`common.close`),testId:`chat-list-drawer`,children:ke}):null,(0,x.jsx)(ps,{open:j,onClose:()=>{ee(!1),window.setTimeout(()=>{let e=document.activeElement;(e===null||e===document.body)&&se.current?.focus()},0)},locale:e,parametersRequest:te,conversation:I,disabled:R,onConversationChange:Do,parameters:(0,x.jsx)(Bs,{defaults:r,draft:i,onChange:a,locale:e}),history:(0,x.jsx)(ms,{conversations:o,busy:v||C,onPending:S,limits:t.bootstrap?.media_limits,onReplace:e=>{Co(e),c(null)},locale:e})}),T===null?null:(0,x.jsx)(pt,{open:!0,title:B(e,`chat.transcript.edit.confirm.title`),body:B(e,`chat.transcript.edit.confirm.body`),confirmLabel:B(e,`chat.transcript.edit.confirm`),cancelLabel:B(e,`common.cancel`),closeLabel:B(e,`common.close`),tone:`danger`,testId:`chat-edit-dialog`,onConfirm:Te,onClose:()=>E(null)}),D===null?null:(0,x.jsx)(pt,{open:!0,title:B(e,`chat.list.delete.confirm.title`,{title:D.title}),body:B(e,`chat.list.delete.confirm.body`),confirmLabel:B(e,`chat.list.delete.confirm`),cancelLabel:B(e,`common.cancel`),closeLabel:B(e,`common.close`),tone:`danger`,testId:`chat-delete-conversation-dialog`,onConfirm:xe,onClose:()=>O(null)})]})}function qs(e){let{appearance:t}=e,n=t.locale,r=n=>e.setAppearance({...t,...n});return(0,x.jsxs)(`div`,{className:`settings-grid`,children:[(0,x.jsx)(Me,{locale:n,label:B(n,`settings.theme`),value:t.themeFamily,onChange:e=>r({themeFamily:e}),options:[{value:`mlxcel`,label:B(n,`settings.theme.mlxcel`)},{value:`glass`,label:B(n,`settings.theme.glass`)}],testId:V(`settings.theme`)}),(0,x.jsx)(Me,{locale:n,label:B(n,`settings.color_scheme`),value:t.colorScheme,onChange:e=>r({colorScheme:e}),options:[{value:`system`,label:B(n,`settings.color_scheme.system`)},{value:`light`,label:B(n,`settings.color_scheme.light`)},{value:`dark`,label:B(n,`settings.color_scheme.dark`)}],testId:V(`settings.color_scheme`)}),(0,x.jsx)(Me,{locale:n,label:B(n,`settings.material`),value:t.material,onChange:e=>r({material:e}),options:[{value:`glass`,label:B(n,`settings.material.glass`)},{value:`tinted`,label:B(n,`settings.material.tinted`)},{value:`opaque`,label:B(n,`settings.material.opaque`)}],testId:V(`settings.material`)}),(0,x.jsx)(Me,{locale:n,label:B(n,`settings.locale`),value:t.locale,onChange:e=>r({locale:e}),options:[{value:`en`,label:B(n,`settings.locale.en`)},{value:`ko`,label:B(n,`settings.locale.ko`)}],testId:V(`settings.locale`)}),(0,x.jsx)(Me,{locale:n,label:B(n,`settings.high_contrast`),value:t.highContrast,onChange:e=>r({highContrast:e}),options:[{value:`system`,label:B(n,`settings.high_contrast.system`)},{value:`on`,label:B(n,`settings.high_contrast.on`)},{value:`off`,label:B(n,`settings.high_contrast.off`)}],testId:V(`settings.high_contrast`)}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsxs)(`span`,{children:[B(n,`settings.glass_intensity`),`: `,t.glassIntensity]}),(0,x.jsx)(`input`,{type:`range`,min:`0`,max:`100`,value:t.glassIntensity,onChange:e=>r({glassIntensity:Number(e.currentTarget.value)}),"data-testid":V(`settings.glass_intensity`)})]}),(0,x.jsx)(yt,{label:B(n,`settings.reduce_motion`),checked:t.reduceMotion,onChange:e=>r({reduceMotion:e}),testId:V(`settings.reduce_motion`)}),(0,x.jsx)(yt,{label:B(n,`settings.reduce_transparency`),checked:t.reduceTransparency,onChange:e=>r({reduceTransparency:e}),testId:V(`settings.reduce_transparency`)})]})}function Js(e,t){let n=`settings.live.${t}`;return De.has(n)?{label:B(e,n),key:t}:{label:t,key:null}}function Ys(e,t){let n=`settings.live.${t.name}.help`;return De.has(n)?B(e,n):t.help}function Xs(e){return(0,x.jsxs)(`div`,{className:`ds-field setting-field`,"data-inner-disabled":e.innerDisabled||void 0,children:[(0,x.jsxs)(`div`,{className:`setting-label-row`,children:[(0,x.jsx)(`label`,{htmlFor:e.id,id:`${e.id}-label`,children:e.label}),e.aside]}),e.children,e.hint?(0,x.jsx)(`small`,{id:`${e.id}-hint`,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:`${e.id}-error`,"data-tone":`error`,children:e.error}):null]})}var Zs=(e,t,n)=>[t?`${e}-hint`:null,n?`${e}-error`:null].filter(Boolean).join(` `)||void 0;function Qs(e){let t=(0,_.useId)();return(0,x.jsx)(Xs,{id:t,label:e.label,aside:e.aside,hint:e.hint,error:e.error,innerDisabled:e.disabled,children:(0,x.jsx)(`input`,{id:t,type:`number`,step:e.step,min:e.min,max:e.max,"aria-labelledby":`${t}-label`,value:e.value,disabled:e.disabled,"aria-invalid":e.error?`true`:void 0,"aria-describedby":Zs(t,e.hint,e.error),"data-testid":e.testId,onChange:t=>e.onChange(t.currentTarget.value)})})}function $s(e){let t=(0,_.useId)(),[n,r]=(0,_.useState)(!1),i=e.error??(n?B(e.locale,`settings.control.invalid_json`):void 0),a=t=>{if(!e.disabled)try{JSON.parse(t),r(!1)}catch{r(!0)}};return(0,x.jsx)(Xs,{id:t,label:e.label,aside:e.aside,hint:e.hint,error:i,innerDisabled:e.disabled,children:(0,x.jsx)(`textarea`,{id:t,rows:2,spellCheck:!1,"aria-labelledby":`${t}-label`,value:e.value,disabled:e.disabled,"aria-invalid":i?`true`:void 0,"aria-describedby":Zs(t,e.hint,i),"data-testid":e.testId,onChange:t=>{r(!1),e.onChange(t.currentTarget.value)},onBlur:e=>a(e.currentTarget.value)})})}function ec({spec:e,value:t,draft:n,onChange:r,error:i,disabled:a=!1,locale:o}){let{label:s,key:c}=Js(o,e.name),l=e.type.endsWith(`_or_null`),u=e.type.replace(`_or_null`,``),d=l&&(n===void 0?t===null:n===`null`),f=n??kt(e,t),p=t===null?B(o,`settings.value.unset`):kt(e,t),m=n!==void 0&&n!==kt(e,t)&&(n!==`null`||t!==null),h=[c,Ys(o,e),m?B(o,`settings.control.server_value`,{value:p}):null].filter(Boolean).join(` · `)||void 0,g=l?(0,x.jsx)(yt,{label:B(o,`settings.control.unset`),checked:d,onChange:e=>{r(e?`null`:t===null?``:void 0)},disabled:a,testId:`setting-${e.name}-unset`}):null,_=d?``:f,v=a||d,y=`setting-${e.name}`,b=t=>(0,x.jsx)(`div`,{className:`setting-control`,"data-kind":e.type,role:l?`group`:void 0,"aria-label":l?s:void 0,children:t});if(u===`int`||u===`float`)return b((0,x.jsx)(Qs,{label:s,value:_,step:u===`int`?`1`:`any`,onChange:e=>r(e),disabled:v,aside:g,hint:h,error:i,testId:y}));if(u===`array`||u===`object`)return b((0,x.jsx)($s,{label:s,value:_,onChange:e=>r(e),disabled:v,aside:g,hint:h,error:i,locale:o,testId:y}));let S;if(u===`bool`)S=(0,x.jsx)(yt,{label:s,checked:f===`true`,onChange:e=>r(String(e)),disabled:v,hint:h,error:i,testId:y});else if(e.allowed!==null){let t=e.allowed.map(e=>({value:e,label:e}));_!==``&&!e.allowed.includes(_)&&t.push({value:_,label:B(o,`settings.control.not_allowed`,{value:_}),disabled:!0}),S=(0,x.jsx)(Me,{locale:o,label:s,value:_,options:t,onChange:e=>r(e),disabled:v,hint:h,error:i,testId:y})}else S=(0,x.jsx)(lt,{label:s,value:_,onChange:e=>r(e),disabled:v,hint:h,error:i,testId:y});return b((0,x.jsxs)(x.Fragment,{children:[S,g]}))}function tc({label:e,content:t}){return(0,x.jsx)(Ve,{content:t,children:(0,x.jsx)(F,{label:e,icon:`help`,className:`settings-help`})})}function nc(){let e=ci(),t=li(),n=e.catalog.find(t=>t.identity.id===e.selectedModelId)??null,r=e.auth.status===`authenticated`&&[`ready`,`streaming`,`polling`].includes(e.connection),i=r&&n?.lifecycle.state===`ready`?n.identity.id:null;return{snapshot:e,actions:t,selected:n,connected:r,readyId:i,revision:i===null||n===null?null:n.identity.revision,scope:i===null?null:ks(e,n),liveEnabled:e.bootstrap?.features.includes(`settings`)===!0,single:e.bootstrap?.server.mode===`single_model`}}function rc(e){let{actions:t,readyId:n,scope:r,liveEnabled:i}=e,[a,o]=(0,_.useState)(`idle`);(0,_.useEffect)(()=>{if(n===null||r===null||!i){o(`idle`);return}let e=new AbortController;return o(`loading`),t.getSettings(n,e.signal).then(t=>{e.signal.aborted||(Is(r,t),o(`ready`))}).catch(()=>{e.signal.aborted||o(`failed`)}),()=>e.abort()},[t,n,r?.instance,r?.revision,i]);let s=e.connected?n===null?`none`:i?null:`disabled`:`offline`;return{state:a,model:e.selected?.identity.display_name??null,reason:s}}function ic({locale:e}){let{defaults:t,setDefaults:n,reset:r}=Os(),i=nc(),a=Ls(i.scope),o=rc(i),[s,c]=(0,_.useState)(()=>Object.fromEntries(Object.entries(t).map(([e,t])=>[e,String(t)]))),[l,u]=(0,_.useState)(``),[d,f]=(0,_.useState)(!1),p=(0,_.useId)(),m=()=>{try{n(ys(Object.fromEntries(Object.entries(s).filter(([,e])=>e.trim()!==``).map(([e,t])=>[e,Number(t)])))),u(``)}catch(t){u(t instanceof Error?t.message:B(e,`settings.generation.invalid`))}},h=o.model??``,g=o.reason===`offline`?B(e,`settings.requests.server_offline`):o.reason===`none`?B(e,`settings.requests.server_none`):o.reason===`disabled`?B(e,`settings.requests.server_disabled`):o.state===`failed`?B(e,`settings.requests.server_failed`,{model:h}):o.state===`ready`?B(e,`settings.requests.server_from`,{model:h}):B(e,`settings.requests.server_loading`,{model:h});return(0,x.jsxs)(`section`,{className:`screen-stack`,"aria-labelledby":p,children:[(0,x.jsxs)(`div`,{className:`settings-heading`,children:[(0,x.jsx)(`h2`,{id:p,children:B(e,`settings.generation.title`)}),(0,x.jsx)(tc,{label:B(e,`settings.help.requests`),content:B(e,`settings.generation.body`)})]}),(0,x.jsx)(`p`,{className:`settings-note`,role:`status`,"data-testid":`settings-requests-server`,children:g}),(0,x.jsx)(`div`,{className:`settings-grid settings-grid-dense`,children:_s.map(n=>{let r=bs(n,void 0,t[n],a?.[n]),{label:i}=Js(e,`default_${n}`);return(0,x.jsx)(Qs,{label:i,value:s[n]??``,step:vs.has(n)?`1`:`any`,min:0,onChange:e=>c(t=>({...t,[n]:e})),hint:`${n} · ${B(e,`settings.requests.effective`,{value:Ss(e,r)})}`,testId:`settings-request-${n}`},n)})}),l?(0,x.jsx)(z,{title:B(e,`settings.generation.error_title`),body:l}):null,(0,x.jsxs)(`div`,{className:`control-row settings-actions`,children:[(0,x.jsx)(P,{tone:`primary`,onClick:m,children:B(e,`settings.generation.save`)}),(0,x.jsx)(P,{onClick:()=>f(!0),children:B(e,`settings.generation.reset_open`)})]}),(0,x.jsxs)(ft,{open:d,title:B(e,`settings.generation.reset.title`),closeLabel:B(e,`common.close`),onClose:()=>f(!1),children:[(0,x.jsx)(`p`,{children:B(e,`settings.generation.reset.body`)}),(0,x.jsx)(P,{onClick:()=>{r(),c({}),u(``),f(!1)},children:B(e,`settings.generation.reset.confirm`)})]})]})}var ac=[`appearance`,`requests`,`model`,`server`],oc=`appearance`;function sc(e){return ac.includes(e??``)?e:oc}function cc(e){return`#settings/${e}`}function lc(e,t,n){return e===null||e<=0||t===null||n===void 0?`unknown`:t+n>e?`exceeds`:`raw-fits`}function uc({modelId:e,nCtx:t,locale:n}){let r=li(),{defaults:i}=Os(),[a,o]=(0,_.useState)(``),[s,c]=(0,_.useState)(null),[l,u]=(0,_.useState)(!1),[d,f]=(0,_.useState)(!1),p=(0,_.useRef)(null);(0,_.useEffect)(()=>()=>p.current?.abort(),[]);let m=async()=>{p.current?.abort();let t=new AbortController;p.current=t,u(!0),f(!1),c(null);try{let n=await r.getTokenCount(e,a,t.signal);t.signal.aborted||c(n)}catch{t.signal.aborted||f(!0)}finally{t.signal.aborted||u(!1)}},h=lc(t,s,i.max_tokens),g=h===`exceeds`?B(n,`settings.context_check.exceeds`):h===`unknown`?B(n,`settings.context_check.unknown`):B(n,`settings.context_check.fits`);return(0,x.jsxs)(`div`,{children:[(0,x.jsx)(lt,{label:B(n,`settings.context_check.label`),value:a,onChange:e=>{p.current?.abort(),u(!1),o(e),c(null)},hint:B(n,`settings.context_check.hint`)}),(0,x.jsx)(P,{disabled:l||a.length===0,onClick:()=>void m(),children:B(n,`settings.context_check.check`)}),(0,x.jsx)(`p`,{role:`status`,children:d?B(n,`settings.context_check.unavailable`):s===null?B(n,`settings.context_check.not_measured`):B(n,`settings.context_check.count`,{count:String(s),verdict:g})})]})}var dc=[`sampling`,`dry`,`diffusion`,`template`,`other`],fc={sampling:`settings.live.group.sampling`,dry:`settings.live.group.dry`,diffusion:`settings.live.group.diffusion`,template:`settings.live.group.template`,other:`settings.live.group.other`};function pc(e){return e.startsWith(`default_dry_`)?`dry`:e.startsWith(`default_`)?`sampling`:e.startsWith(`diffusion_`)?`diffusion`:e===`chat_template_kwargs`||e===`lang_bias_config`||e.includes(`template`)||e.includes(`bias`)?`template`:`other`}function mc(e){return dc.map(t=>({group:t,specs:e.filter(e=>pc(e.name)===t)})).filter(e=>e.specs.length>0)}function hc(e){if(e.default===null)return`null`;let t=kt(e,e.default);return typeof e.default==`number`?Math.fround(Number(t))===Math.fround(e.default)?t:Et(e,e.default):t}var gc=32,_c=new Map,vc=e=>JSON.stringify([e.instance,e.modelId,e.revision]);function yc(e,t){if(_c.delete(e),t!==null)for(_c.set(e,t);_c.size>gc;){let e=_c.keys().next();if(e.done)break;_c.delete(e.value)}}function bc(e,t){let n=e.current;return e.current=null,n!==null&&t.fingerprint!==n}function xc(e,t){return e!==null&&Object.hasOwn(e.current,t)?e.current[t]??null:null}function Sc(e,t){return Object.hasOwn(e,t)?e[t]:void 0}function Cc({modelId:e,scope:t,locale:n}){let r=li(),[i,a]=(0,_.useState)(null),o=t?vc(t):null,[s]=(0,_.useState)(()=>o===null?void 0:_c.get(o)),[c,l]=(0,_.useState)(()=>({...s?.values})),u=(0,_.useRef)(s?.fingerprint??null),d=i?.fingerprint??s?.fingerprint??null;(0,_.useEffect)(()=>{o!==null&&yc(o,Object.keys(c).length>0?{values:c,fingerprint:d}:null)},[c,o,d]);let[f,p]=(0,_.useState)({}),[m,h]=(0,_.useState)(``),[g,v]=(0,_.useState)(!1),[y,b]=(0,_.useState)(!1),S=(0,_.useRef)(null),C=(0,_.useRef)(n);C.current=n;let w=(0,_.useId)(),T=e=>{a(e),t&&Is(t,e)},E=(0,_.useRef)(T);E.current=T,(0,_.useEffect)(()=>{let t=new AbortController;return S.current=t,v(!0),r.getSettings(e,t.signal).then(e=>{t.signal.aborted||(E.current(e),bc(u,e)&&h(B(C.current,`settings.live.conflict`)))}).catch(()=>{t.signal.aborted||h(B(C.current,`settings.live.unavailable`))}).finally(()=>{t.signal.aborted||v(!1)}),()=>t.abort()},[r,e]);let D=async()=>{let t=S.current;if(!(t===null||t.signal.aborted)){v(!0);try{let n=await r.getSettings(e,t.signal);t.signal.aborted||(T(n),h(B(C.current,bc(u,n)?`settings.live.conflict`:`settings.live.refreshed`)))}catch{t.signal.aborted||h(B(C.current,`settings.live.refresh_failed`))}finally{t.signal.aborted||v(!1)}}},O=async()=>{let t=S.current;if(t===null||t.signal.aborted||i===null)return;let a={},o={};for(let e of i.schema)if(Object.hasOwn(c,e.name))try{a[e.name]=Tt(e,c[e.name])}catch(t){o[e.name]=t instanceof Error?t.message:B(n,`settings.live.invalid_value`)}if(p(o),!(Object.keys(o).length>0)){v(!0);try{let n=await r.getSettings(e,t.signal);if(t.signal.aborted)return;if(n.fingerprint!==i.fingerprint){T(n),h(B(C.current,`settings.live.conflict`));return}let o=await r.patchSettings(e,a,t.signal);if(t.signal.aborted)return;p(Object.fromEntries(o.rejected.map(e=>[e.name,e.reason]))),l(e=>Object.fromEntries(Object.entries(e).filter(([e])=>!Object.hasOwn(o.applied,e)))),h(o.rejected.length>0?B(C.current,`settings.live.partial`,{applied:String(Object.keys(o.applied).length),rejected:String(o.rejected.length)}):B(C.current,`settings.live.applied`));let s=await r.getSettings(e,t.signal);t.signal.aborted||T(s)}catch{t.signal.aborted||h(B(C.current,`settings.live.unknown_outcome`))}finally{t.signal.aborted||v(!1)}}},k=(e,t)=>l(n=>t===void 0?Object.fromEntries(Object.entries(n).filter(([t])=>t!==e)):{...n,[e]:t}),A=i?.schema.filter(e=>e.mutable)??[];return(0,x.jsxs)(`section`,{className:`screen-stack settings-live`,"aria-labelledby":w,children:[(0,x.jsxs)(`div`,{className:`settings-heading`,children:[(0,x.jsx)(`h2`,{id:w,children:B(n,`settings.live.title`)}),(0,x.jsx)(tc,{label:B(n,`settings.help.live`),content:B(n,`settings.live.body`)}),(0,x.jsxs)(`div`,{className:`control-row settings-actions`,children:[(0,x.jsx)(P,{onClick:()=>void D(),disabled:g,children:B(n,`settings.live.refresh`)}),(0,x.jsx)(P,{tone:`primary`,onClick:()=>void O(),disabled:g||Object.keys(c).length===0,children:B(n,`settings.live.apply`)}),(0,x.jsx)(P,{onClick:()=>b(!0),disabled:g||i===null,children:B(n,`settings.live.reset_open`)})]})]}),m?(0,x.jsx)(z,{tone:`warning`,title:B(n,`settings.live.result_title`),body:m,testId:`settings-live-result`}):null,(0,x.jsx)(`div`,{className:`settings-live-groups`,children:mc(A).map(({group:e,specs:t})=>(0,x.jsxs)(`fieldset`,{className:`settings-group`,"data-group":e,"data-size":t.length<=2?`small`:`large`,children:[(0,x.jsx)(`legend`,{children:B(n,fc[e])}),(0,x.jsx)(`div`,{className:`settings-grid settings-grid-dense`,children:t.map(e=>(0,x.jsx)(ec,{spec:e,value:xc(i,e.name),draft:Object.hasOwn(c,e.name)?c[e.name]:void 0,onChange:t=>k(e.name,t),error:Sc(f,e.name),disabled:g,locale:n},e.name))})]},e))}),(0,x.jsxs)(ft,{open:y,title:B(n,`settings.live.reset.title`),closeLabel:B(n,`common.close`),onClose:()=>b(!1),children:[(0,x.jsx)(`p`,{children:B(n,`settings.live.reset.body`)}),(0,x.jsx)(P,{onClick:()=>{i!==null&&l(Object.fromEntries(i.schema.filter(e=>e.mutable).map(e=>[e.name,hc(e)]))),p({}),b(!1)},children:B(n,`settings.live.reset.confirm`)})]})]})}function wc({model:e,locale:t,single:n}){let r=Ii(e?.identity.id??null),[i,a]=(0,_.useState)(r.profile),[o,s]=(0,_.useState)(e===null?`reusable`:`model`),[c,l]=(0,_.useState)(``),[u,d]=(0,_.useState)(``),[f,p]=(0,_.useState)(!1),[m,h]=(0,_.useState)(`idle`),g=(0,_.useId)(),v=o===`reusable`?r.reusable:r.modelProfile;(0,_.useEffect)(()=>{a(v)},[v]);let y=e=>{try{e(),d(B(t,`settings.profile.saved`))}catch(e){d(e instanceof Error?e.message:B(t,`settings.profile.failed`))}},b=Li(v),S=async()=>{try{if(!navigator.clipboard)throw Error(`Clipboard unavailable`);await navigator.clipboard.writeText(b),h(`copied`)}catch{h(`failed`)}},C=e=>e.trim()===``?void 0:Number(e);return(0,x.jsxs)(`section`,{className:`screen-stack`,"aria-labelledby":g,children:[(0,x.jsxs)(`div`,{className:`settings-heading`,children:[(0,x.jsx)(`h2`,{id:g,children:B(t,`settings.profile.title`)}),(0,x.jsx)(tc,{label:B(t,`settings.help.profile`),content:B(t,`settings.profile.body`)})]}),n?(0,x.jsx)(z,{tone:`info`,title:B(t,`settings.profile.single.title`),body:B(t,`settings.profile.single.body`)}):null,(0,x.jsxs)(`div`,{className:`settings-grid settings-grid-dense`,children:[(0,x.jsx)(Me,{locale:t,label:B(t,`settings.profile.scope`),value:o,onChange:e=>s(e),options:[{value:`reusable`,label:B(t,`settings.profile.scope.reusable`)},...e===null?[]:[{value:`model`,label:e.identity.display_name}]]}),(0,x.jsx)(Qs,{label:B(t,`settings.profile.ctx_size`),value:i.ctx_size?.toString()??``,step:`1`,min:1,max:262144,onChange:e=>a({...i,ctx_size:C(e)}),hint:B(t,`settings.profile.ctx_hint`)}),(0,x.jsx)(Qs,{label:B(t,`settings.profile.n_parallel`),value:i.n_parallel?.toString()??``,step:`1`,min:1,max:32,onChange:e=>a({...i,n_parallel:C(e)}),hint:B(t,`settings.profile.parallel_hint`)}),(0,x.jsx)(Me,{locale:t,label:B(t,`settings.profile.kv_cache_mode`),value:i.kv_cache_mode??``,onChange:e=>a({...i,kv_cache_mode:e===``?void 0:e}),options:[{value:``,label:B(t,`settings.profile.inherit`)},...Ti.map(e=>({value:e,label:e}))]})]}),u?(0,x.jsx)(z,{tone:`info`,title:B(t,`settings.profile.result_title`),body:u,testId:`settings-profile-result`}):null,(0,x.jsxs)(`div`,{className:`control-row settings-actions`,children:[(0,x.jsx)(P,{tone:`primary`,onClick:()=>y(()=>r.save(Di(i),o)),children:B(t,`settings.profile.save`)}),(0,x.jsx)(P,{onClick:()=>{a(v),d(``)},children:B(t,`settings.profile.discard`)}),(0,x.jsx)(P,{onClick:()=>p(!0),children:B(t,`settings.profile.reset_open`)})]}),(0,x.jsxs)(`div`,{className:`settings-disclosures`,children:[(0,x.jsxs)(`details`,{className:`settings-disclosure`,onToggle:()=>h(`idle`),children:[(0,x.jsx)(`summary`,{children:B(t,`settings.profile.show_cli`)}),(0,x.jsxs)(`p`,{children:[B(t,`settings.profile.saved_pending`),`: `,(0,x.jsx)(`code`,{children:JSON.stringify(v)})]}),(0,x.jsx)(`p`,{children:B(t,`settings.profile.cli`)}),(0,x.jsx)(`code`,{className:`settings-cli`,"data-testid":`settings-profile-cli`,children:b}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(P,{onClick:()=>void S(),children:B(t,`settings.profile.copy`)}),(0,x.jsx)(`span`,{role:`status`,children:m===`copied`?B(t,`settings.profile.copied`):m===`failed`?B(t,`settings.profile.copy_failed`):``})]})]}),(0,x.jsxs)(`details`,{className:`settings-disclosure`,children:[(0,x.jsx)(`summary`,{children:B(t,`settings.profile.transfer`)}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:B(t,`settings.profile.transfer.label`)}),(0,x.jsx)(`textarea`,{value:c,onChange:e=>l(e.currentTarget.value),maxLength:65536,rows:6})]}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(P,{onClick:()=>l(r.exportJson()),children:B(t,`settings.profile.transfer.export`)}),(0,x.jsx)(P,{onClick:()=>y(()=>r.importJson(c)),children:B(t,`settings.profile.transfer.import`)})]})]})]}),(0,x.jsxs)(ft,{open:f,title:B(t,`settings.profile.reset.title`),closeLabel:B(t,`common.close`),onClose:()=>p(!1),children:[(0,x.jsx)(`p`,{children:B(t,`settings.profile.reset.body`)}),(0,x.jsx)(P,{onClick:()=>{y(()=>r.reset(o)),p(!1)},children:B(t,`settings.profile.reset.confirm`)})]})]})}function Tc({locale:e,target:t}){let{snapshot:n,actions:r}=t;return(0,x.jsx)(Me,{locale:e,label:B(e,`settings.server.model`),value:n.selectedModelId??``,options:[{value:``,label:B(e,`settings.server.model.none`)},...n.catalog.map(t=>({value:t.identity.id,label:`${t.identity.display_name} · ${Vn(e,t.lifecycle.state)}`}))],onChange:e=>r.selectModel(e===``?null:e),hint:B(e,`settings.server.model.hint`),testId:`settings-model-selector`})}var Ec=e=>(0,x.jsx)(z,{tone:`warning`,title:B(e,`settings.server.unavailable.title`),body:B(e,`settings.server.unavailable.body`)}),Dc=e=>(0,x.jsx)(z,{tone:`info`,title:B(e,`settings.server.no_model.title`),body:B(e,`settings.server.no_model.body`)}),Oc=e=>(0,x.jsx)(z,{tone:`info`,title:B(e,`settings.server.live_disabled.title`),body:B(e,`settings.server.live_disabled.body`)});function kc({locale:e}){let t=nc();if(!t.connected)return(0,x.jsx)(`div`,{className:`screen-stack settings-sections`,children:Ec(e)});let{selected:n,readyId:r,scope:i}=t;return(0,x.jsxs)(`div`,{className:`screen-stack settings-sections`,children:[(0,x.jsx)(Tc,{locale:e,target:t}),(0,x.jsx)(wc,{model:n,locale:e,single:t.single},n?.identity.id??`reusable`),r===null?Dc(e):t.liveEnabled?(0,x.jsx)(Cc,{modelId:r,scope:i,locale:e},r):Oc(e)]})}function Ac({locale:e,specs:t,current:n}){let r=(0,_.useId)(),i=new Map;for(let e of t){let t=e.reason??e.help;i.set(t,[...i.get(t)??[],e.name])}let a=t=>{let r=Object.hasOwn(n,t.name)?n[t.name]:t.default;return r==null?B(e,`settings.value.unset`):kt(t,r)};return(0,x.jsxs)(`section`,{className:`screen-stack`,"aria-labelledby":r,children:[(0,x.jsx)(`h2`,{id:r,children:B(e,`settings.live.startup`)}),(0,x.jsx)(`dl`,{className:`settings-values settings-keys`,"data-testid":`settings-startup-values`,children:t.map(t=>(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`dt`,{children:Js(e,t.name).label}),(0,x.jsx)(`dd`,{children:a(t)})]},t.name))}),(0,x.jsxs)(`details`,{className:`settings-disclosure`,children:[(0,x.jsx)(`summary`,{children:B(e,`settings.server.startup.reasons`)}),(0,x.jsx)(`dl`,{className:`settings-reasons`,children:[...i].map(([e,t])=>(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`dt`,{children:e}),(0,x.jsx)(`dd`,{children:t.join(`, `)})]},e))})]})]})}function jc({locale:e}){let t=nc(),{actions:n,readyId:r,scope:i,selected:a,liveEnabled:o}=t,[s,c]=(0,_.useState)(null),[l,u]=(0,_.useState)(!1),[d,f]=(0,_.useState)(null),[p,m]=(0,_.useState)(!1),h=(0,_.useId)();if((0,_.useEffect)(()=>{if(c(null),u(!1),r===null)return;let e=new AbortController;return n.getModelProps(r,e.signal).then(t=>{e.signal.aborted||c(t)}).catch(()=>{e.signal.aborted||u(!0)}),()=>e.abort()},[n,r,a?.identity.revision]),(0,_.useEffect)(()=>{if(f(null),m(!1),r===null||!o)return;let e=new AbortController;return n.getSettings(r,e.signal).then(t=>{e.signal.aborted||(f(t),i!==null&&Is(i,t))}).catch(()=>{e.signal.aborted||m(!0)}),()=>e.abort()},[n,r,i?.instance,i?.revision,o]),!t.connected)return(0,x.jsx)(`div`,{className:`screen-stack settings-sections`,children:Ec(e)});let g=B(e,`settings.server.unknown`),v=d?.schema.filter(e=>!e.mutable)??[];return(0,x.jsxs)(`div`,{className:`screen-stack settings-sections`,children:[(0,x.jsx)(Tc,{locale:e,target:t}),r===null?Dc(e):(0,x.jsxs)(x.Fragment,{children:[(0,x.jsxs)(`section`,{className:`screen-stack`,"aria-labelledby":h,children:[(0,x.jsxs)(`div`,{className:`settings-heading`,children:[(0,x.jsx)(`h2`,{id:h,children:B(e,`settings.server.context.title`)}),(0,x.jsx)(tc,{label:B(e,`settings.help.context`),content:B(e,`settings.server.context.n_ctx_hint`)})]}),l?(0,x.jsx)(z,{tone:`info`,title:B(e,`settings.server.props_unavailable.title`),body:B(e,`settings.server.props_unavailable.body`),testId:`settings-props-unavailable`}):null,(0,x.jsxs)(`dl`,{className:`settings-values`,"data-testid":`settings-context-values`,children:[(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`dt`,{children:B(e,`settings.server.context.n_ctx`)}),(0,x.jsx)(`dd`,{"data-testid":`settings-context-n-ctx`,children:s?.nCtx?.toString()??g})]}),(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`dt`,{children:B(e,`settings.server.context.slots`)}),(0,x.jsx)(`dd`,{children:s?.totalSlots?.toString()??g})]}),(0,x.jsxs)(`div`,{children:[(0,x.jsx)(`dt`,{children:B(e,`settings.server.context.kv_mode`)}),(0,x.jsx)(`dd`,{children:s?.kvCacheMode??g})]})]}),(0,x.jsxs)(`details`,{className:`settings-disclosure`,children:[(0,x.jsx)(`summary`,{children:B(e,`settings.server.context.geometry`)}),(0,x.jsx)(`code`,{className:`settings-cli`,children:s===null||s.geometry===null?B(e,`settings.server.context.geometry_unknown`):JSON.stringify(s.geometry)})]}),(0,x.jsx)(uc,{modelId:r,nCtx:s?.nCtx??null,locale:e},r)]}),o?p?(0,x.jsx)(z,{tone:`warning`,title:B(e,`settings.server.startup.failed`),body:B(e,`settings.live.unavailable`)}):d===null?null:(0,x.jsx)(Ac,{locale:e,specs:v,current:d.current}):Oc(e)]})]})}var Mc={appearance:`settings.section.appearance`,requests:`settings.section.requests`,model:`settings.section.model`,server:`settings.section.server`},Nc={appearance:`settings.browser_only`,requests:`settings.section.requests.description`,model:`settings.section.model.description`,server:`settings.section.server.description`},Pc={appearance:V(`settings.appearance`),requests:V(`settings.section.requests.description`),model:V(`settings.section.model.description`),server:V(`settings.section.server.description`)};function Fc(e,t,n,r){return e===`appearance`?(0,x.jsx)(qs,{appearance:n,setAppearance:r}):e===`requests`?(0,x.jsx)(ic,{locale:t}):e===`model`?(0,x.jsx)(kc,{locale:t}):(0,x.jsx)(jc,{locale:t})}function Ic(e){let t=e.appearance.locale;return(0,x.jsxs)(`div`,{className:`screen-stack settings-screen`,children:[(0,x.jsx)(Ye,{title:B(t,`settings.title`),titleTestId:V(`settings.title`),description:B(t,Nc[e.section]),descriptionTestId:Pc[e.section],actions:(0,x.jsx)(tc,{label:B(t,`settings.help.storage`),content:B(t,`settings.server.privacy.body`)})}),(0,x.jsx)(fe,{label:B(t,`settings.sections`),active:e.section,onChange:t=>e.onSectionChange(t),tabs:ac.map(n=>({id:n,label:B(t,Mc[n]),panel:n===e.section?Fc(n,t,e.appearance,e.setAppearance):null}))})]})}var Lc=[`models`,`chat`,`activity`,`settings`,`gallery`];function Rc(e){return e.auth.status===`authenticated`&&e.connection!==`schema-mismatch`}function zc(e=window.location.hash){let t=e.replace(/^#/,``).replace(/^\//,``),[n,...r]=t.split(`/`);return n===`settings`?{route:`settings`,section:sc(r.join(`/`)||void 0)}:{route:Lc.includes(t)?t:`models`,section:oc}}function Bc(){let e=ci(),t=li(),[n,r]=(0,_.useState)(()=>zc()),i=n.route,[a,o]=(0,_.useState)(br),[s,c]=(0,_.useState)(null),[l,u]=(0,_.useState)(null),d=(0,_.useRef)(0);(0,_.useEffect)(()=>()=>{d.current+=1},[]),(0,_.useEffect)(()=>{let e=()=>r(zc());return window.addEventListener(`hashchange`,e),()=>window.removeEventListener(`hashchange`,e)},[]),(0,_.useEffect)(()=>{if(Cr(document.documentElement,a),xr(a),a.colorScheme===`system`)return ur(e=>Cr(document.documentElement,a,e))},[a]),(0,_.useEffect)(()=>{let e=()=>{document.documentElement.dataset.documentHidden=String(document.hidden)};return e(),document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e)},[]),(0,_.useEffect)(()=>{e.auth.status===`authenticated`&&u(null)},[e.auth.status]),(0,_.useEffect)(()=>{e.auth.status===`signed-out`&&Co([])},[e.auth.status]);let f=e=>{r({route:e,section:oc}),window.history.replaceState(null,``,`#${e}`)},p=e=>{r({route:`settings`,section:e}),window.history.replaceState(null,``,cc(e))},m=e=>{let n=d.current+1;d.current=n,u(null),t.login(e).catch(e=>{d.current===n&&(e instanceof DOMException&&e.name===`AbortError`||u(Tn(e)))})},h=()=>{d.current+=1,u(null),t.logout()},g=()=>{u(null),t.refresh()},v=()=>{h(),window.location.reload()},y=n=>{let r=An(e,n);r!==e.selectedModelId&&t.selectModel(r),f(`models`),c(null)},b=()=>{Rc(e)&&wo(),f(`chat`),c(null)},S=kn(e),C=e.auth.status===`authenticated`&&e.catalogSequence!==null?S.map(e=>({id:e.identity.id,name:e.identity.display_name,state:e.lifecycle.state,stateLabel:Vn(a.locale,e.lifecycle.state)})):null,w=Vc(n,a,o,p,{snapshot:e,authFailure:l,login:m,logout:h,retry:g,recoverSchema:v});return(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(Jn,{locale:a.locale,route:i,onRouteChange:f,onCommand:()=>c(`command`),onHelp:()=>c(`help`),onNewChat:b,onOpenModel:y,loadedModels:C,connection:{label:jn(a.locale,e),state:e.connection,details:Mn(a.locale,e)},sessionAction:e.auth.tokenPresent?(0,x.jsx)(F,{label:B(a.locale,`toolbar.logout`),icon:`key`,onClick:h,"data-testid":V(`toolbar.logout`)}):null,inspector:null,children:w}),(0,x.jsx)(Gn,{open:s===`command`,locale:a.locale,catalog:e.catalog,loaded:S,onClose:()=>c(null),onNavigate:e=>{f(e),c(null)},onNewChat:b,onOpenModel:y}),(0,x.jsxs)(ft,{open:s===`help`,title:B(a.locale,`help.title`),onClose:()=>c(null),testId:`help-dialog`,closeLabel:B(a.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":V(`help.body`),children:B(a.locale,`help.body`)}),(0,x.jsx)(`ul`,{className:`shortcut-list`,"data-testid":`help-shortcuts`,children:qn.map(e=>(0,x.jsxs)(`li`,{"data-testid":`help-shortcut-${e.id}`,children:[(0,x.jsx)(`span`,{className:`shortcut-keys`,children:e.keys.map((t,n)=>(0,x.jsxs)(_.Fragment,{children:[n>0?` ${e.separator} `:null,(0,x.jsx)(`kbd`,{children:t})]},t))}),(0,x.jsx)(`span`,{children:B(a.locale,e.key)})]},e.id))})]})]})}function Vc(e,t,n,r,i){let{route:a}=e;return a===`models`?(0,x.jsx)(Hc,{locale:t.locale,context:i}):a===`chat`?(0,x.jsx)(Uc,{locale:t.locale,context:i}):a===`activity`?(0,x.jsx)(Wc,{locale:t.locale,context:i}):a===`settings`?(0,x.jsx)(Ic,{appearance:t,setAppearance:n,section:e.section,onSectionChange:r}):(0,x.jsx)(pa,{locale:t.locale})}function Hc(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,x.jsx)(ca,{locale:e.locale}):(0,x.jsx)(Pn,{locale:e.locale,title:B(e.locale,`models.title`),titleTestId:V(`models.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function Uc(e){return e.context.snapshot.auth.status===`authenticated`&&e.context.snapshot.connection!==`schema-mismatch`?(0,x.jsx)(Ks,{locale:e.locale}):(0,x.jsx)(Pn,{locale:e.locale,title:B(e.locale,`chat.title`),titleTestId:V(`chat.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function Wc(e){return e.context.snapshot.auth.status===`authenticated`&&![`schema-mismatch`,`forbidden`,`unauthorized`].includes(e.context.snapshot.connection)?(0,x.jsx)(wi,{locale:e.locale}):(0,x.jsx)(Pn,{locale:e.locale,title:B(e.locale,`activity.title`),titleTestId:V(`activity.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}var Gc=document.getElementById(`root`);if(Gc===null)throw Error(`Missing #root element for mlxcel WebUI.`);(0,v.createRoot)(Gc).render((0,x.jsx)(_.StrictMode,{children:(0,x.jsx)(si,{children:(0,x.jsx)(Bc,{})})}));export{c as i,b as n,u as r,B as t};