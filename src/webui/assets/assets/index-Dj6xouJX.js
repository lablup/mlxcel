var e=Object.create,t=Object.defineProperty,n=Object.getOwnPropertyDescriptor,r=Object.getOwnPropertyNames,i=Object.getPrototypeOf,a=Object.prototype.hasOwnProperty,o=(e,t)=>()=>(t||(e((t={exports:{}}).exports,t),e=null),t.exports),s=(e,i,o,s)=>{if(i&&typeof i==`object`||typeof i==`function`)for(var c=r(i),l=0,u=c.length,d;l<u;l++)d=c[l],!a.call(e,d)&&d!==o&&t(e,d,{get:(e=>i[e]).bind(null,d),enumerable:!(s=n(i,d))||s.enumerable});return e},c=(n,r,o)=>(o=n==null?{}:e(i(n)),s(r||!n||!n.__esModule||!a.call(n,`default`)?t(o,`default`,{value:n,enumerable:!0}):o,n)),l=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.portal`),r=Symbol.for(`react.fragment`),i=Symbol.for(`react.strict_mode`),a=Symbol.for(`react.profiler`),o=Symbol.for(`react.consumer`),s=Symbol.for(`react.context`),c=Symbol.for(`react.forward_ref`),l=Symbol.for(`react.suspense`),u=Symbol.for(`react.memo`),d=Symbol.for(`react.lazy`),f=Symbol.for(`react.activity`),p=Symbol.for(`react.view_transition`),m=Symbol.iterator;function h(e){return typeof e!=`object`||!e?null:(e=m&&e[m]||e[`@@iterator`],typeof e==`function`?e:null)}var g={isMounted:function(){return!1},enqueueForceUpdate:function(){},enqueueReplaceState:function(){},enqueueSetState:function(){}},_=Object.assign,v={};function y(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}y.prototype.isReactComponent={},y.prototype.setState=function(e,t){if(typeof e!=`object`&&typeof e!=`function`&&e!=null)throw Error(`takes an object of state variables to update or a function which returns an object of state variables.`);this.updater.enqueueSetState(this,e,t,`setState`)},y.prototype.forceUpdate=function(e){this.updater.enqueueForceUpdate(this,e,`forceUpdate`)};function b(){}b.prototype=y.prototype;function x(e,t,n){this.props=e,this.context=t,this.refs=v,this.updater=n||g}var ee=x.prototype=new b;ee.constructor=x,_(ee,y.prototype),ee.isPureReactComponent=!0;var te=Array.isArray;function S(){}var C={H:null,A:null,T:null,S:null},ne=Object.prototype.hasOwnProperty;function w(e,n,r){var i=r.ref;return{$$typeof:t,type:e,key:n,ref:i===void 0?null:i,props:r}}function re(e,t){return w(e.type,t,e.props)}function ie(e){return typeof e==`object`&&!!e&&e.$$typeof===t}function ae(e){var t={"=":`=0`,":":`=2`};return`$`+e.replace(/[=:]/g,function(e){return t[e]})}var oe=/\/+/g;function se(e,t){return typeof e==`object`&&e&&e.key!=null?ae(``+e.key):t.toString(36)}function ce(e){switch(e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason;default:switch(typeof e.status==`string`?e.then(S,S):(e.status=`pending`,e.then(function(t){e.status===`pending`&&(e.status=`fulfilled`,e.value=t)},function(t){e.status===`pending`&&(e.status=`rejected`,e.reason=t)})),e.status){case`fulfilled`:return e.value;case`rejected`:throw e.reason}}throw e}function le(e,r,i,a,o){var s=typeof e;(s===`undefined`||s===`boolean`)&&(e=null);var c=!1;if(e===null)c=!0;else switch(s){case`bigint`:case`string`:case`number`:c=!0;break;case`object`:switch(e.$$typeof){case t:case n:c=!0;break;case d:return c=e._init,le(c(e._payload),r,i,a,o)}}if(c)return o=o(e),c=a===``?`.`+se(e,0):a,te(o)?(i=``,c!=null&&(i=c.replace(oe,`$&/`)+`/`),le(o,r,i,``,function(e){return e})):o!=null&&(ie(o)&&(o=re(o,i+(o.key==null||e&&e.key===o.key?``:(``+o.key).replace(oe,`$&/`)+`/`)+c)),r.push(o)),1;c=0;var l=a===``?`.`:a+`:`;if(te(e))for(var u=0;u<e.length;u++)a=e[u],s=l+se(a,u),c+=le(a,r,i,s,o);else if(u=h(e),typeof u==`function`)for(e=u.call(e),u=0;!(a=e.next()).done;)a=a.value,s=l+se(a,u++),c+=le(a,r,i,s,o);else if(s===`object`){if(typeof e.then==`function`)return le(ce(e),r,i,a,o);throw r=String(e),Error(`Objects are not valid as a React child (found: `+(r===`[object Object]`?`object with keys {`+Object.keys(e).join(`, `)+`}`:r)+`). If you meant to render a collection of children, use an array instead.`)}return c}function ue(e,t,n){if(e==null)return e;var r=[],i=0;return le(e,r,``,``,function(e){return t.call(n,e,i++)}),r}function de(e){if(e._status===-1){var t=e._result,n=t();n.then(function(t){(e._status===0||e._status===-1)&&(e._status=1,e._result=t,n.status===void 0&&(n.status=`fulfilled`,n.value=t))},function(t){(e._status===0||e._status===-1)&&(e._status=2,e._result=t,n.status===void 0&&(n.status=`rejected`,n.reason=t))}),e._status===-1&&(e._status=0,e._result=n)}if(e._status===1)return e._result.default;throw e._result}var fe=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)};function pe(e){var t=C.T,n={};n.types=t===null?null:t.types,C.T=n;try{var r=e(),i=C.S;i!==null&&i(n,r),typeof r==`object`&&r&&typeof r.then==`function`&&r.then(S,fe)}catch(e){fe(e)}finally{t!==null&&n.types!==null&&(t.types=n.types),C.T=t}}function me(e){var t=C.T;if(t!==null){var n=t.types;n===null?t.types=[e]:n.indexOf(e)===-1&&n.push(e)}else pe(me.bind(null,e))}var he={map:ue,forEach:function(e,t,n){ue(e,function(){t.apply(this,arguments)},n)},count:function(e){var t=0;return ue(e,function(){t++}),t},toArray:function(e){return ue(e,function(e){return e})||[]},only:function(e){if(!ie(e))throw Error(`React.Children.only expected to receive a single React element child.`);return e}};e.Activity=f,e.Children=he,e.Component=y,e.Fragment=r,e.Profiler=a,e.PureComponent=x,e.StrictMode=i,e.Suspense=l,e.ViewTransition=p,e.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=C,e.__COMPILER_RUNTIME={__proto__:null,c:function(e){return C.H.useMemoCache(e)}},e.addTransitionType=me,e.cache=function(e){return function(){return e.apply(null,arguments)}},e.cacheSignal=function(){return null},e.cloneElement=function(e,t,n){if(e==null)throw Error(`The argument must be a React element, but you passed `+e+`.`);var r=_({},e.props),i=e.key;if(t!=null)for(a in t.key!==void 0&&(i=``+t.key),t)!ne.call(t,a)||a===`key`||a===`__self`||a===`__source`||a===`ref`&&t.ref===void 0||(r[a]=t[a]);var a=arguments.length-2;if(a===1)r.children=n;else if(1<a){for(var o=Array(a),s=0;s<a;s++)o[s]=arguments[s+2];r.children=o}return w(e.type,i,r)},e.createContext=function(e){return e={$$typeof:s,_currentValue:e,_currentValue2:e,_threadCount:0,Provider:null,Consumer:null},e.Provider=e,e.Consumer={$$typeof:o,_context:e},e},e.createElement=function(e,t,n){var r,i={},a=null;if(t!=null)for(r in t.key!==void 0&&(a=``+t.key),t)ne.call(t,r)&&r!==`key`&&r!==`__self`&&r!==`__source`&&(i[r]=t[r]);var o=arguments.length-2;if(o===1)i.children=n;else if(1<o){for(var s=Array(o),c=0;c<o;c++)s[c]=arguments[c+2];i.children=s}if(e&&e.defaultProps)for(r in o=e.defaultProps,o)i[r]===void 0&&(i[r]=o[r]);return w(e,a,i)},e.createRef=function(){return{current:null}},e.forwardRef=function(e){return{$$typeof:c,render:e}},e.isValidElement=ie,e.lazy=function(e){return{$$typeof:d,_payload:{_status:-1,_result:e},_init:de}},e.memo=function(e,t){return{$$typeof:u,type:e,compare:t===void 0?null:t}},e.startTransition=pe,e.unstable_useCacheRefresh=function(){return C.H.useCacheRefresh()},e.use=function(e){return C.H.use(e)},e.useActionState=function(e,t,n){return C.H.useActionState(e,t,n)},e.useCallback=function(e,t){return C.H.useCallback(e,t)},e.useContext=function(e){return C.H.useContext(e)},e.useDebugValue=function(){},e.useDeferredValue=function(e,t){return C.H.useDeferredValue(e,t)},e.useEffect=function(e,t){return C.H.useEffect(e,t)},e.useEffectEvent=function(e){return C.H.useEffectEvent(e)},e.useId=function(){return C.H.useId()},e.useImperativeHandle=function(e,t,n){return C.H.useImperativeHandle(e,t,n)},e.useInsertionEffect=function(e,t){return C.H.useInsertionEffect(e,t)},e.useLayoutEffect=function(e,t){return C.H.useLayoutEffect(e,t)},e.useMemo=function(e,t){return C.H.useMemo(e,t)},e.useOptimistic=function(e,t){return C.H.useOptimistic(e,t)},e.useReducer=function(e,t,n){return C.H.useReducer(e,t,n)},e.useRef=function(e){return C.H.useRef(e)},e.useState=function(e){return C.H.useState(e)},e.useSyncExternalStore=function(e,t,n){return C.H.useSyncExternalStore(e,t,n)},e.useTransition=function(){return C.H.useTransition()},e.version=`19.3.0`})),u=o(((e,t)=>{t.exports=l()})),d=o((e=>{function t(e,t){var n=e.length;e.push(t);a:for(;0<n;){var r=n-1>>>1,a=e[r];if(0<i(a,t))e[r]=t,e[n]=a,n=r;else break a}}function n(e){return e.length===0?null:e[0]}function r(e){if(e.length===0)return null;var t=e[0],n=e.pop();if(n!==t){e[0]=n;a:for(var r=0,a=e.length,o=a>>>1;r<o;){var s=2*(r+1)-1,c=e[s],l=s+1,u=e[l];if(0>i(c,n))l<a&&0>i(u,c)?(e[r]=u,e[l]=n,r=l):(e[r]=c,e[s]=n,r=s);else if(l<a&&0>i(u,n))e[r]=u,e[l]=n,r=l;else break a}}return t}function i(e,t){var n=e.sortIndex-t.sortIndex;return n===0?e.id-t.id:n}if(e.unstable_now=void 0,typeof performance==`object`&&typeof performance.now==`function`){var a=performance;e.unstable_now=function(){return a.now()}}else{var o=Date,s=o.now();e.unstable_now=function(){return o.now()-s}}var c=[],l=[],u=1,d=null,f=3,p=!1,m=!1,h=!1,g=!1,_=typeof setTimeout==`function`?setTimeout:null,v=typeof clearTimeout==`function`?clearTimeout:null,y=typeof setImmediate<`u`?setImmediate:null;function b(e){for(var i=n(l);i!==null;){if(i.callback===null)r(l);else if(i.startTime<=e)r(l),i.sortIndex=i.expirationTime,t(c,i);else break;i=n(l)}}function x(e){if(h=!1,b(e),!m){if(n(c)!==null)m=!0,ee||(ee=!0,re());else{var t=n(l);t!==null&&oe(x,t.startTime-e)}}}var ee=!1,te=-1,S=5,C=-1;function ne(){return g?!0:!(e.unstable_now()-C<S)}function w(){if(g=!1,ee){var t=e.unstable_now();C=t;var i=!0;try{a:{m=!1,h&&(h=!1,v(te),te=-1),p=!0;var a=f;try{b:{for(b(t),d=n(c);d!==null&&!(d.expirationTime>t&&ne());){var o=d.callback;if(typeof o==`function`){d.callback=null,f=d.priorityLevel;var s=o(d.expirationTime<=t);if(t=e.unstable_now(),typeof s==`function`){d.callback=s,b(t),i=!0;break b}d===n(c)&&r(c),b(t)}else r(c);d=n(c)}if(d!==null)i=!0;else{var u=n(l);u!==null&&oe(x,u.startTime-t),i=!1}}break a}finally{d=null,f=a,p=!1}i=void 0}}finally{i?re():ee=!1}}}var re;if(typeof y==`function`)re=function(){y(w)};else if(typeof MessageChannel<`u`){var ie=new MessageChannel,ae=ie.port2;ie.port1.onmessage=w,re=function(){ae.postMessage(null)}}else re=function(){_(w,0)};function oe(t,n){te=_(function(){t(e.unstable_now())},n)}e.unstable_IdlePriority=5,e.unstable_ImmediatePriority=1,e.unstable_LowPriority=4,e.unstable_NormalPriority=3,e.unstable_Profiling=null,e.unstable_UserBlockingPriority=2,e.unstable_cancelCallback=function(e){e.callback=null},e.unstable_forceFrameRate=function(e){0>e||125<e?console.error(`forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported`):S=0<e?Math.floor(1e3/e):5},e.unstable_getCurrentPriorityLevel=function(){return f},e.unstable_next=function(e){switch(f){case 1:case 2:case 3:var t=3;break;default:t=f}var n=f;f=t;try{return e()}finally{f=n}},e.unstable_requestPaint=function(){g=!0},e.unstable_runWithPriority=function(e,t){switch(e){case 1:case 2:case 3:case 4:case 5:break;default:e=3}var n=f;f=e;try{return t()}finally{f=n}},e.unstable_scheduleCallback=function(r,i,a){var o=e.unstable_now();switch(typeof a==`object`&&a?(a=a.delay,a=typeof a==`number`&&0<a?o+a:o):a=o,r){case 1:var s=-1;break;case 2:s=250;break;case 5:s=1073741823;break;case 4:s=1e4;break;default:s=5e3}return s=a+s,r={id:u++,callback:i,priorityLevel:r,startTime:a,expirationTime:s,sortIndex:-1},a>o?(r.sortIndex=a,t(l,r),n(c)===null&&r===n(l)&&(h?(v(te),te=-1):h=!0,oe(x,a-o))):(r.sortIndex=s,t(c,r),m||p||(m=!0,ee||(ee=!0,re()))),r},e.unstable_shouldYield=ne,e.unstable_wrapCallback=function(e){var t=f;return function(){var n=f;f=t;try{return e.apply(this,arguments)}finally{f=n}}}})),f=o(((e,t)=>{t.exports=d()})),p=o((e=>{var t=u();function n(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function r(){}var i={d:{f:r,r:function(){throw Error(n(522))},D:r,C:r,L:r,m:r,X:r,S:r,M:r},p:0,findDOMNode:null},a=Symbol.for(`react.portal`),o=Symbol.for(`react.recoverable`),s=Symbol.for(`react.optimistic_key`);function c(e,t,n){var r=3<arguments.length&&arguments[3]!==void 0?arguments[3]:null;return{$$typeof:a,key:r==null?null:r===s?s:``+r,children:e,containerInfo:t,implementation:n}}var l=t.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;function d(e,t){if(e===`font`)return``;if(typeof t==`string`)return t===`use-credentials`?t:``}e.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE=i,e.browser=function(e){return{$$typeof:o,_reason:e}},e.createPortal=function(e,t){var r=2<arguments.length&&arguments[2]!==void 0?arguments[2]:null;if(!t||t.nodeType!==1&&t.nodeType!==9&&t.nodeType!==11)throw Error(n(299));return c(e,t,null,r)},e.flushSync=function(e){var t=l.T,n=i.p;try{if(l.T=null,i.p=2,e)return e()}finally{l.T=t,i.p=n,i.d.f()}},e.preconnect=function(e,t){typeof e==`string`&&(t?(t=t.crossOrigin,t=typeof t==`string`?t===`use-credentials`?t:``:void 0):t=null,i.d.C(e,t))},e.prefetchDNS=function(e){typeof e==`string`&&i.d.D(e)},e.preinit=function(e,t){if(typeof e==`string`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin),a=typeof t.integrity==`string`?t.integrity:void 0,o=typeof t.fetchPriority==`string`?t.fetchPriority:void 0;n===`style`?i.d.S(e,typeof t.precedence==`string`?t.precedence:void 0,{crossOrigin:r,integrity:a,fetchPriority:o}):n===`script`&&i.d.X(e,{crossOrigin:r,integrity:a,fetchPriority:o,nonce:typeof t.nonce==`string`?t.nonce:void 0})}},e.preinitModule=function(e,t){if(typeof e==`string`){if(typeof t==`object`&&t){if(t.as==null||t.as===`script`){var n=d(t.as,t.crossOrigin);i.d.M(e,{crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}}else t??i.d.M(e)}},e.preload=function(e,t){if(typeof e==`string`&&typeof t==`object`&&t&&typeof t.as==`string`){var n=t.as,r=d(n,t.crossOrigin);i.d.L(e,n,{crossOrigin:r,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,type:typeof t.type==`string`?t.type:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0,referrerPolicy:typeof t.referrerPolicy==`string`?t.referrerPolicy:void 0,imageSrcSet:typeof t.imageSrcSet==`string`?t.imageSrcSet:void 0,imageSizes:typeof t.imageSizes==`string`?t.imageSizes:void 0,media:typeof t.media==`string`?t.media:void 0})}},e.preloadModule=function(e,t){if(typeof e==`string`){if(t){var n=d(t.as,t.crossOrigin);i.d.m(e,{as:typeof t.as==`string`&&t.as!==`script`?t.as:void 0,crossOrigin:n,integrity:typeof t.integrity==`string`?t.integrity:void 0,nonce:typeof t.nonce==`string`?t.nonce:void 0,fetchPriority:typeof t.fetchPriority==`string`?t.fetchPriority:void 0})}else i.d.m(e)}},e.requestFormReset=function(e){i.d.r(e)},e.unstable_batchedUpdates=function(e,t){return e(t)},e.useFormState=function(e,t,n){return l.H.useFormState(e,t,n)},e.useFormStatus=function(){return l.H.useHostTransitionStatus()},e.version=`19.3.0`})),m=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=p()})),h=o((e=>{var t=f(),n=u(),r=m();function i(e){var t=`https://react.dev/errors/`+e;if(1<arguments.length){t+=`?args[]=`+encodeURIComponent(arguments[1]);for(var n=2;n<arguments.length;n++)t+=`&args[]=`+encodeURIComponent(arguments[n])}return`Minified React error #`+e+`; visit `+t+` for the full message or use the non-minified dev environment for full errors and additional helpful warnings.`}function a(e){return!(!e||e.nodeType!==1&&e.nodeType!==9&&e.nodeType!==11)}function o(e){for(var t=e,n=t;n&&!n.alternate;)t=n,t.flags&4098&&(e=t.return),n=t.return;for(;t.return;)t=t.return;return t.tag===3?e:null}function s(e){if(e.tag===13){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function c(e){if(e.tag===31){var t=e.memoizedState;if(t===null&&(e=e.alternate,e!==null&&(t=e.memoizedState)),t!==null)return t.dehydrated}return null}function l(e){if(o(e)!==e)throw Error(i(188))}function d(e){var t=e.alternate;if(!t){if(t=o(e),t===null)throw Error(i(188));return t===e?e:null}for(var n=e,r=t;;){var a=n.return;if(a===null)break;var s=a.alternate;if(s===null){if(r=a.return,r!==null){n=r;continue}break}if(a.child===s.child){for(s=a.child;s;){if(s===n)return l(a),e;if(s===r)return l(a),t;s=s.sibling}throw Error(i(188))}if(n.return!==r.return)n=a,r=s;else{for(var c=!1,u=a.child;u;){if(u===n){c=!0,n=a,r=s;break}if(u===r){c=!0,r=a,n=s;break}u=u.sibling}if(!c){for(u=s.child;u;){if(u===n){c=!0,n=s,r=a;break}if(u===r){c=!0,r=s,n=a;break}u=u.sibling}if(!c)throw Error(i(189))}}if(n.alternate!==r)throw Error(i(190))}if(n.tag!==3)throw Error(i(188));return n.stateNode.current===n?e:t}function p(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e;for(e=e.child;e!==null;){if(t=p(e),t!==null)return t;e=e.sibling}return null}function h(e,t,n,r,i,a){for(;e!==null;){if((e.tag===5||e.tag===27||e.tag===6)&&n(e,r,i,a)||(e.tag!==22||e.memoizedState===null)&&(t||e.tag!==5&&e.tag!==27)&&h(e.child,t,n,r,i,a))return!0;e=e.sibling}return!1}function g(e){for(e=e.return;e!==null;){if(e.tag===3||e.tag===5||e.tag===27)return e;e=e.return}return null}function _(e){var t=!1;for(e=e.return;e!==null&&(e.tag===4&&(t=!0),e.tag!==3&&e.tag!==5&&e.tag!==27);)e=e.return;return t}function v(e){var t=[null,null],n=g(e);return n===null||y(t,e,n.child,{foundSelf:!1}),t}function y(e,t,n,r){for(;n!==null;){if(n===t)r.foundSelf=!0;else if(n.tag===5||n.tag===27||n.tag===6){if(r.foundSelf)return e[1]=n,!0;e[0]=n}else if((n.tag!==22||n.memoizedState===null)&&y(e,t,n.child,r))return!0;n=n.sibling}return!1}function b(e){switch(e.tag){case 5:case 27:case 6:return e.stateNode;case 3:return e.stateNode.containerInfo;default:throw Error(i(559))}}var x=null,ee=null;function te(e,t,n){return e===n||e===t&&(x=e,!0)}function S(e,t,n){return e===n?(ee=e,!1):e===t&&(ee!==null&&(x=e),!0)}function C(e){if(e===null)return null;do e=e===null?null:e.return;while(e&&e.tag!==5&&e.tag!==27&&e.tag!==3);return e||null}function ne(e,t,n){for(var r=0,i=e;i;i=n(i))r++;i=0;for(var a=t;a;a=n(a))i++;for(;0<r-i;)e=n(e),r--;for(;0<i-r;)t=n(t),i--;for(;r--;){if(e===t||t!==null&&e===t.alternate)return e;e=n(e),t=n(t)}return null}var w=Object.assign,re=Symbol.for(`react.element`),ie=Symbol.for(`react.transitional.element`),ae=Symbol.for(`react.portal`),oe=Symbol.for(`react.fragment`),se=Symbol.for(`react.strict_mode`),ce=Symbol.for(`react.profiler`),le=Symbol.for(`react.consumer`),ue=Symbol.for(`react.context`),de=Symbol.for(`react.forward_ref`),fe=Symbol.for(`react.suspense`),pe=Symbol.for(`react.suspense_list`),me=Symbol.for(`react.memo`),he=Symbol.for(`react.lazy`),ge=Symbol.for(`react.activity`),_e=Symbol.for(`react.legacy_hidden`),ve=Symbol.for(`react.memo_cache_sentinel`),T=Symbol.for(`react.view_transition`),E=Symbol.for(`react.recoverable`),ye=Symbol.iterator;function be(e){return typeof e!=`object`||!e?null:(e=ye&&e[ye]||e[`@@iterator`],typeof e==`function`?e:null)}var xe=Symbol.for(`react.client.reference`);function Se(e){if(e==null)return null;if(typeof e==`function`)return e.$$typeof===xe?null:e.displayName||e.name||null;if(typeof e==`string`)return e;switch(e){case oe:return`Fragment`;case ce:return`Profiler`;case se:return`StrictMode`;case fe:return`Suspense`;case pe:return`SuspenseList`;case ge:return`Activity`;case T:return`ViewTransition`}if(typeof e==`object`)switch(e.$$typeof){case ae:return`Portal`;case ue:return e.displayName||`Context`;case le:return(e._context.displayName||`Context`)+`.Consumer`;case de:var t=e.render;return e=e.displayName,e||=(e=t.displayName||t.name||``,e===``?`ForwardRef`:`ForwardRef(`+e+`)`),e;case me:return t=e.displayName||null,t===null?Se(e.type)||`Memo`:t;case he:t=e._payload,e=e._init;try{return Se(e(t))}catch{}}return null}var Ce=Array.isArray,D=n.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,O=r.__DOM_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,we={pending:!1,data:null,method:null,action:null},Te=[],Ee=-1;function De(e){return{current:e}}function Oe(e){0>Ee||(e.current=Te[Ee],Te[Ee]=null,Ee--)}function k(e,t){Ee++,Te[Ee]=e.current,e.current=t}var ke=De(null),Ae=De(null),je=De(null),Me=De(null);function Ne(e,t){switch(k(je,t),k(Ae,e),k(ke,null),t.nodeType){case 9:case 11:e=(e=t.documentElement)&&(e=e.namespaceURI)?up(e):0;break;default:if(e=t.tagName,t=t.namespaceURI)t=up(t),e=dp(t,e);else switch(e){case`svg`:e=1;break;case`math`:e=2;break;default:e=0}}Oe(ke),k(ke,e)}function Pe(){Oe(ke),Oe(Ae),Oe(je)}function Fe(e){var t=e.memoizedState;t!==null&&(sh._currentValue=t.memoizedState,k(Me,e)),t=ke.current;var n=dp(t,e.type);t!==n&&(k(Ae,e),k(ke,n))}function Ie(e){Ae.current===e&&(Oe(ke),Oe(Ae)),Me.current===e&&(Oe(Me),sh._currentValue=we)}var Le,Re;function ze(e){if(Le===void 0)try{throw Error()}catch(e){var t=e.stack.trim().match(/\n( *(at )?)/);Le=t&&t[1]||``,Re=-1<e.stack.indexOf(`
    at`)?` (<anonymous>)`:-1<e.stack.indexOf(`@`)?`@unknown:0:0`:``}return`
`+Le+e+Re}var Be=!1;function Ve(e,t){if(!e||Be)return``;Be=!0;var n=Error.prepareStackTrace;Error.prepareStackTrace=void 0;try{var r={DetermineComponentFrameRoot:function(){try{if(t){var n=function(){throw Error()};if(Object.defineProperty(n.prototype,"props",{set:function(){throw Error()}}),typeof Reflect==`object`&&Reflect.construct){try{Reflect.construct(n,[])}catch(e){var r=e}Reflect.construct(e,[],n)}else{try{n.call()}catch(e){r=e}n=!1;try{var i=Object.getOwnPropertyDescriptor(e.prototype,`props`);Object.defineProperty(e.prototype,"props",{configurable:!0,set:function(){throw Error()}}),n=!0,new e}finally{n&&(i===void 0?delete e.prototype.props:Object.defineProperty(e.prototype,"props",i))}}}else{try{throw Error()}catch(e){r=e}(n=e())&&typeof n.catch==`function`&&n.catch(function(){})}}catch(e){if(e&&r&&typeof e.stack==`string`)return[e.stack,r.stack]}return[null,null]}};r.DetermineComponentFrameRoot.displayName=`DetermineComponentFrameRoot`;var i=Object.getOwnPropertyDescriptor(r.DetermineComponentFrameRoot,`name`);i&&i.configurable&&Object.defineProperty(r.DetermineComponentFrameRoot,"name",{value:`DetermineComponentFrameRoot`});var a=r.DetermineComponentFrameRoot(),o=a[0],s=a[1];if(o&&s){var c=o.split(`
`),l=s.split(`
`);for(i=r=0;r<c.length&&!c[r].includes(`DetermineComponentFrameRoot`);)r++;for(;i<l.length&&!l[i].includes(`DetermineComponentFrameRoot`);)i++;if(r===c.length||i===l.length)for(r=c.length-1,i=l.length-1;1<=r&&0<=i&&c[r]!==l[i];)i--;for(;1<=r&&0<=i;r--,i--)if(c[r]!==l[i]){if(r!==1||i!==1)do if(r--,i--,0>i||c[r]!==l[i]){var u=`
`+c[r].replace(` at new `,` at `);return e.displayName&&u.includes(`<anonymous>`)&&(u=u.replace(`<anonymous>`,e.displayName)),u}while(1<=r&&0<=i);break}}}finally{Be=!1,Error.prepareStackTrace=n}return(n=e?e.displayName||e.name:``)?ze(n):``}function He(e,t){switch(e.tag){case 26:case 27:case 5:return ze(e.type);case 16:return ze(`Lazy`);case 13:return e.child!==t&&t!==null?ze(`Suspense Fallback`):ze(`Suspense`);case 19:return ze(`SuspenseList`);case 0:case 15:return Ve(e.type,!1);case 11:return Ve(e.type.render,!1);case 1:return Ve(e.type,!0);case 31:return ze(`Activity`);case 30:return ze(`ViewTransition`);default:return``}}function Ue(e){try{var t=``,n=null;do t+=He(e,n),n=e,e=e.return;while(e);return t}catch(e){return`
Error generating stack: `+e.message+`
`+e.stack}}var We=Object.prototype.hasOwnProperty,Ge=t.unstable_scheduleCallback,A=t.unstable_cancelCallback,Ke=t.unstable_shouldYield,qe=t.unstable_requestPaint,Je=t.unstable_now,Ye=t.unstable_getCurrentPriorityLevel,Xe=t.unstable_ImmediatePriority,Ze=t.unstable_UserBlockingPriority,Qe=t.unstable_NormalPriority,$e=t.unstable_LowPriority,et=t.unstable_IdlePriority,tt=t.log,nt=t.unstable_setDisableYieldValue,rt=null,it=null;function at(e){if(typeof tt==`function`&&nt(e),it&&typeof it.setStrictMode==`function`)try{it.setStrictMode(rt,e)}catch{}}var ot=Math.clz32?Math.clz32:lt,st=Math.log,ct=Math.LN2;function lt(e){return e>>>=0,e===0?32:31-(st(e)/ct|0)|0}var ut=256,dt=262144,ft=4194304;function pt(e){var t=e&42;if(t!==0)return t;switch(e&-e){case 1:return 1;case 2:return 2;case 4:return 4;case 8:return 8;case 16:return 16;case 32:return 32;case 64:return 64;case 128:return 128;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:return e&-e;case 262144:case 524288:case 1048576:case 2097152:return e&3932160;case 4194304:case 8388608:case 16777216:case 33554432:return e&62914560;case 67108864:return 67108864;case 134217728:return 134217728;case 268435456:return 268435456;case 536870912:return 536870912;case 1073741824:return 0;default:return e}}function mt(e,t,n){var r=e.pendingLanes;if(r===0)return 0;var i=0,a=e.suspendedLanes,o=e.pingedLanes;e=e.warmLanes;var s=r&134217727;return s===0?(s=r&~a,s===0?o===0?n||(n=r&~e,n!==0&&(i=pt(n))):i=pt(o):i=pt(s)):(r=s&~a,r===0?(o&=s,o===0?n||(n=s&~e,n!==0&&(i=pt(n))):i=pt(o)):i=pt(r)),i===0?0:t!==0&&t!==i&&(t&a)===0&&(a=i&-i,n=t&-t,a>=n||a===32&&n&4194048)?t:i}function ht(e,t){return(e.pendingLanes&~(e.suspendedLanes&~e.pingedLanes)&t)===0}function gt(e,t){t&8&&(t|=t&32);var n=e.entangledLanes;if(n!==0)for(e=e.entanglements,n&=t;0<n;){var r=31-ot(n),i=1<<r;t|=e[r],n&=~i}return t}function _t(e,t){switch(e){case 1:case 2:case 4:case 8:case 64:return t+250;case 16:case 32:case 128:case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:return t+5e3;case 4194304:case 8388608:case 16777216:case 33554432:return-1;case 67108864:case 134217728:case 268435456:case 536870912:case 1073741824:return-1;default:return-1}}function vt(){var e=ft;return ft<<=1,!(ft&62914560)&&(ft=4194304),e}function yt(e){for(var t=[],n=0;31>n;n++)t.push(e);return t}function bt(e,t){e.pendingLanes|=t,t!==268435456&&(e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0)}function xt(e,t,n,r,i,a){var o=e.pendingLanes;e.pendingLanes=n,e.suspendedLanes=0,e.pingedLanes=0,e.warmLanes=0,e.expiredLanes&=n,e.entangledLanes&=n,e.errorRecoveryDisabledLanes&=n,e.shellSuspendCounter=0;var s=e.entanglements,c=e.expirationTimes,l=e.hiddenUpdates;for(n=o&~n;0<n;){var u=31-ot(n),d=1<<u;s[u]=0,c[u]=-1;var f=l[u];if(f!==null)for(l[u]=null,u=0;u<f.length;u++){var p=f[u];p!==null&&(p.lane&=-536870913)}n&=~d}r!==0&&St(e,r,0),a!==0&&i===0&&e.tag!==0&&(e.suspendedLanes|=a&~(o&~t))}function St(e,t,n){e.pendingLanes|=t,e.suspendedLanes&=~t;var r=31-ot(t);e.entangledLanes|=t,e.entanglements[r]=e.entanglements[r]|1073741824|n&261930}function Ct(e,t){var n=e.entangledLanes|=t;for(e=e.entanglements;n;){var r=31-ot(n),i=1<<r;i&t|e[r]&t&&(e[r]|=t),n&=~i}}function wt(e,t){var n=t&-t;return n=n&42?1:Tt(n),(n&(e.suspendedLanes|t))===0?n:0}function Tt(e){switch(e){case 2:e=1;break;case 8:e=4;break;case 32:e=16;break;case 256:case 512:case 1024:case 2048:case 4096:case 8192:case 16384:case 32768:case 65536:case 131072:case 262144:case 524288:case 1048576:case 2097152:case 4194304:case 8388608:case 16777216:case 33554432:e=128;break;case 268435456:e=134217728;break;default:e=0}return e}function Et(e){return e&=-e,2<e?8<e?e&134217727?32:268435456:8:2}function Dt(){var e=O.p;return e===0?(e=window.event,e===void 0?32:Ch(e.type)):e}function Ot(e,t){var n=O.p;try{return O.p=e,t()}finally{O.p=n}}var kt=Math.random().toString(36).slice(2),At=`__reactFiber$`+kt,jt=`__reactProps$`+kt,Mt=`__reactContainer$`+kt,Nt=`__reactEvents$`+kt,Pt=`__reactListeners$`+kt,Ft=`__reactHandles$`+kt,It=`__reactResources$`+kt,Lt=`__reactMarker$`+kt,Rt=`__reactLoad$`+kt;function zt(e){delete e[At],delete e[jt],delete e[Pt],delete e[Ft]}function Bt(e){var t;if(t=e[At])return t;for(var n=e.parentNode;n;){if(t=n[Mt]||n[At]){if(n=t.alternate,t.child!==null||n!==null&&n.child!==null)for(e=fm(e);e!==null;){if(n=e[At])return n;e=fm(e)}return t}e=n,n=e.parentNode}return null}function Vt(e){if(e=e[At]||e[Mt]){var t=e.tag;if(t===5||t===6||t===13||t===31||t===26||t===27||t===3)return e}return null}function Ht(e){var t=e.tag;if(t===5||t===26||t===27||t===6)return e.stateNode;throw Error(i(33))}function Ut(e){var t=e[It];return t||=e[It]={hoistableStyles:new Map,hoistableScripts:new Map},t}function j(e){e[Lt]=!0}function Wt(e){e[Rt]=void 0}var Gt=new Set,Kt={};function qt(e,t){Jt(e,t),Jt(e+`Capture`,t)}function Jt(e,t){for(Kt[e]=t,e=0;e<t.length;e++)Gt.add(t[e])}var Yt=RegExp(`^[:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD][:A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD\\-.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*$`),Xt={},Zt={};function Qt(e){return We.call(Zt,e)?!0:We.call(Xt,e)?!1:Yt.test(e)?Zt[e]=!0:(Xt[e]=!0,!1)}var M=!1;function $t(){var e=M;return M=!1,e}function en(e,t,n){if(Qt(t)){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:e.removeAttribute(t);return;case`boolean`:var r=t.toLowerCase().slice(0,5);if(r!==`data-`&&r!==`aria-`){e.removeAttribute(t);return}}e.setAttribute(t,n)}}}function tn(e,t,n){if(n===null)e.removeAttribute(t);else{switch(typeof n){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(t);return}e.setAttribute(t,n)}}function nn(e,t,n,r){if(r===null)e.removeAttribute(n);else{switch(typeof r){case`undefined`:case`function`:case`symbol`:case`boolean`:e.removeAttribute(n);return}e.setAttributeNS(t,n,r)}}function N(e){switch(typeof e){case`bigint`:case`boolean`:case`number`:case`string`:case`undefined`:return e;case`object`:return e;default:return``}}function rn(e){var t=e.type;return(e=e.nodeName)&&e.toLowerCase()===`input`&&(t===`checkbox`||t===`radio`)}function an(e,t,n){var r=Object.getOwnPropertyDescriptor(e.constructor.prototype,t);if(!e.hasOwnProperty(t)&&r!==void 0&&typeof r.get==`function`&&typeof r.set==`function`){var i=r.get,a=r.set;return Object.defineProperty(e,t,{configurable:!0,get:function(){return i.call(this)},set:function(e){n=``+e,a.call(this,e)}}),Object.defineProperty(e,t,{enumerable:r.enumerable}),{getValue:function(){return n},setValue:function(e){n=``+e},stopTracking:function(){e._valueTracker=null,delete e[t]}}}}function on(e){if(!e._valueTracker){var t=rn(e)?`checked`:`value`;e._valueTracker=an(e,t,``+e[t])}}function sn(e){if(!e)return!1;var t=e._valueTracker;if(!t)return!0;var n=t.getValue(),r=``;return e&&(r=rn(e)?e.checked?`true`:`false`:e.value),e=r,e!==n&&(t.setValue(e),!0)}var cn=/[\n"\\]/g;function ln(e){return e.replace(cn,function(e){return`\\`+e.charCodeAt(0).toString(16)+` `})}function un(e,t,n,r,i,a,o,s){e.name=``,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`?e.type=o:e.removeAttribute(`type`),t==null?o!==`submit`&&o!==`reset`||e.removeAttribute(`value`):o===`number`?(t===0&&e.value===``||e.value!=t)&&(e.value=``+N(t)):e.value!==``+N(t)&&(e.value=``+N(t)),t==null?n==null?r!=null&&e.removeAttribute(`value`):fn(e,N(n)):o===`number`&&e.value==t?fn(e,N(e.value)):fn(e,N(t)),i==null&&a!=null&&(e.defaultChecked=!!a),i!=null&&(e.checked=i&&typeof i!=`function`&&typeof i!=`symbol`),s!=null&&typeof s!=`function`&&typeof s!=`symbol`&&typeof s!=`boolean`?e.name=``+N(s):e.removeAttribute(`name`)}function dn(e,t,n,r,i,a,o,s){if(a!=null&&typeof a!=`function`&&typeof a!=`symbol`&&typeof a!=`boolean`&&(e.type=a),t!=null||n!=null){if(!(a!==`submit`&&a!==`reset`||t!=null)){on(e);return}n=n==null?``:``+N(n),t=t==null?n:``+N(t),s||t===e.value||(e.value=t),e.defaultValue=t}r??=i,r=typeof r!=`function`&&typeof r!=`symbol`&&!!r,e.checked=s?e.checked:!!r,e.defaultChecked=!!r,o!=null&&typeof o!=`function`&&typeof o!=`symbol`&&typeof o!=`boolean`&&(e.name=o),on(e)}function fn(e,t){e.defaultValue!==``+t&&(e.defaultValue=``+t)}function pn(e,t,n,r){if(e=e.options,t){t={};for(var i=0;i<n.length;i++)t[`$`+n[i]]=!0;for(n=0;n<e.length;n++)i=t.hasOwnProperty(`$`+e[n].value),e[n].selected!==i&&(e[n].selected=i),i&&r&&(e[n].defaultSelected=!0)}else{for(n=``+N(n),t=null,i=0;i<e.length;i++){if(e[i].value===n){e[i].selected=!0,r&&(e[i].defaultSelected=!0);return}t!==null||e[i].disabled||(t=e[i])}t!==null&&(t.selected=!0)}}function mn(e,t,n){if(t!=null&&(t=``+N(t),t!==e.value&&(e.value=t),n==null)){e.defaultValue!==t&&(e.defaultValue=t);return}e.defaultValue=n==null?``:``+N(n)}function hn(e,t,n,r){if(t==null){if(r!=null){if(n!=null)throw Error(i(92));if(Ce(r)){if(1<r.length)throw Error(i(93));r=r[0]}n=r}n??=``,t=n}n=N(t),e.defaultValue=n,r=e.textContent,r===n&&r!==``&&r!==null&&(e.value=r),on(e)}function gn(e,t){if(t){var n=e.firstChild;if(n&&n===e.lastChild&&n.nodeType===3){n.nodeValue=t;return}}e.textContent=t}var _n=new Set(`animationIterationCount aspectRatio borderImageOutset borderImageSlice borderImageWidth boxFlex boxFlexGroup boxOrdinalGroup columnCount columns flex flexGrow flexPositive flexShrink flexNegative flexOrder gridArea gridRow gridRowEnd gridRowSpan gridRowStart gridColumn gridColumnEnd gridColumnSpan gridColumnStart fontWeight lineClamp lineHeight opacity order orphans scale tabSize widows zIndex zoom fillOpacity floodOpacity stopOpacity strokeDasharray strokeDashoffset strokeMiterlimit strokeOpacity strokeWidth MozAnimationIterationCount MozBoxFlex MozBoxFlexGroup MozLineClamp msAnimationIterationCount msFlex msZoom msFlexGrow msFlexNegative msFlexOrder msFlexPositive msFlexShrink msGridColumn msGridColumnSpan msGridRow msGridRowSpan WebkitAnimationIterationCount WebkitBoxFlex WebKitBoxFlexGroup WebkitBoxOrdinalGroup WebkitColumnCount WebkitColumns WebkitFlex WebkitFlexGrow WebkitFlexPositive WebkitFlexShrink WebkitLineClamp`.split(` `));function vn(e,t,n){var r=t.indexOf(`--`)===0;n==null||typeof n==`boolean`||n===``?r?e.setProperty(t,``):t===`float`?e.cssFloat=``:e[t]=``:r?e.setProperty(t,n):typeof n!=`number`||n===0||_n.has(t)?t===`float`?e.cssFloat=n:e[t]=(``+n).trim():e[t]=n+`px`}function yn(e,t,n){if(t!=null&&typeof t!=`object`)throw Error(i(62));if(e=e.style,n!=null){for(var r in n)!n.hasOwnProperty(r)||t!=null&&t.hasOwnProperty(r)||(r.indexOf(`--`)===0?e.setProperty(r,``):r===`float`?e.cssFloat=``:e[r]=``,M=!0);for(var a in t)r=t[a],t.hasOwnProperty(a)&&n[a]!==r&&(vn(e,a,r),M=!0)}else for(var o in t)t.hasOwnProperty(o)&&vn(e,o,t[o])}function bn(e){if(e.indexOf(`-`)===-1)return!1;switch(e){case`annotation-xml`:case`color-profile`:case`font-face`:case`font-face-src`:case`font-face-uri`:case`font-face-format`:case`font-face-name`:case`missing-glyph`:return!1;default:return!0}}var xn=new Map([[`acceptCharset`,`accept-charset`],[`htmlFor`,`for`],[`httpEquiv`,`http-equiv`],[`crossOrigin`,`crossorigin`],[`accentHeight`,`accent-height`],[`alignmentBaseline`,`alignment-baseline`],[`arabicForm`,`arabic-form`],[`baselineShift`,`baseline-shift`],[`capHeight`,`cap-height`],[`clipPath`,`clip-path`],[`clipRule`,`clip-rule`],[`colorInterpolation`,`color-interpolation`],[`colorInterpolationFilters`,`color-interpolation-filters`],[`colorProfile`,`color-profile`],[`colorRendering`,`color-rendering`],[`dominantBaseline`,`dominant-baseline`],[`enableBackground`,`enable-background`],[`fillOpacity`,`fill-opacity`],[`fillRule`,`fill-rule`],[`floodColor`,`flood-color`],[`floodOpacity`,`flood-opacity`],[`fontFamily`,`font-family`],[`fontSize`,`font-size`],[`fontSizeAdjust`,`font-size-adjust`],[`fontStretch`,`font-stretch`],[`fontStyle`,`font-style`],[`fontVariant`,`font-variant`],[`fontWeight`,`font-weight`],[`glyphName`,`glyph-name`],[`glyphOrientationHorizontal`,`glyph-orientation-horizontal`],[`glyphOrientationVertical`,`glyph-orientation-vertical`],[`horizAdvX`,`horiz-adv-x`],[`horizOriginX`,`horiz-origin-x`],[`imageRendering`,`image-rendering`],[`letterSpacing`,`letter-spacing`],[`lightingColor`,`lighting-color`],[`markerEnd`,`marker-end`],[`markerMid`,`marker-mid`],[`markerStart`,`marker-start`],[`maskType`,`mask-type`],[`overlinePosition`,`overline-position`],[`overlineThickness`,`overline-thickness`],[`paintOrder`,`paint-order`],[`panose-1`,`panose-1`],[`pointerEvents`,`pointer-events`],[`renderingIntent`,`rendering-intent`],[`shapeRendering`,`shape-rendering`],[`stopColor`,`stop-color`],[`stopOpacity`,`stop-opacity`],[`strikethroughPosition`,`strikethrough-position`],[`strikethroughThickness`,`strikethrough-thickness`],[`strokeDasharray`,`stroke-dasharray`],[`strokeDashoffset`,`stroke-dashoffset`],[`strokeLinecap`,`stroke-linecap`],[`strokeLinejoin`,`stroke-linejoin`],[`strokeMiterlimit`,`stroke-miterlimit`],[`strokeOpacity`,`stroke-opacity`],[`strokeWidth`,`stroke-width`],[`textAnchor`,`text-anchor`],[`textDecoration`,`text-decoration`],[`textRendering`,`text-rendering`],[`transformOrigin`,`transform-origin`],[`underlinePosition`,`underline-position`],[`underlineThickness`,`underline-thickness`],[`unicodeBidi`,`unicode-bidi`],[`unicodeRange`,`unicode-range`],[`unitsPerEm`,`units-per-em`],[`vAlphabetic`,`v-alphabetic`],[`vHanging`,`v-hanging`],[`vIdeographic`,`v-ideographic`],[`vMathematical`,`v-mathematical`],[`vectorEffect`,`vector-effect`],[`vertAdvY`,`vert-adv-y`],[`vertOriginX`,`vert-origin-x`],[`vertOriginY`,`vert-origin-y`],[`wordSpacing`,`word-spacing`],[`writingMode`,`writing-mode`],[`xmlnsXlink`,`xmlns:xlink`],[`xHeight`,`x-height`]]),Sn=/^[\u0000-\u001F ]*j[\r\n\t]*a[\r\n\t]*v[\r\n\t]*a[\r\n\t]*s[\r\n\t]*c[\r\n\t]*r[\r\n\t]*i[\r\n\t]*p[\r\n\t]*t[\r\n\t]*:/i;function Cn(e){return Sn.test(``+e)?`javascript:throw new Error('React has blocked a javascript: URL as a security precaution.')`:e}function wn(){}var Tn=null;function En(e){return e=e.target||e.srcElement||window,e.correspondingUseElement&&(e=e.correspondingUseElement),e.nodeType===3?e.parentNode:e}var Dn=null,On=null;function kn(e){var t=Vt(e);if(t&&(e=t.stateNode)){var n=e[jt]||null;a:switch(e=t.stateNode,t.type){case`input`:if(un(e,n.value,n.defaultValue,n.defaultValue,n.checked,n.defaultChecked,n.type,n.name),t=n.name,n.type===`radio`&&t!=null){for(n=e;n.parentNode;)n=n.parentNode;for(n=n.querySelectorAll(`input[name="`+ln(``+t)+`"][type="radio"]`),t=0;t<n.length;t++){var r=n[t];if(r!==e&&r.form===e.form){var a=r[jt]||null;if(!a)throw Error(i(90));un(r,a.value,a.defaultValue,a.defaultValue,a.checked,a.defaultChecked,a.type,a.name)}}for(t=0;t<n.length;t++)r=n[t],r.form===e.form&&sn(r)}break a;case`textarea`:mn(e,n.value,n.defaultValue);break a;case`select`:t=n.value,t!=null&&pn(e,!!n.multiple,t,!1)}}}var An=!1;function jn(e,t,n){if(An)return e(t,n);An=!0;try{return e(t)}finally{if(An=!1,(Dn!==null||On!==null)&&(zd(),Dn&&(t=Dn,e=On,On=Dn=null,kn(t),e)))for(t=0;t<e.length;t++)kn(e[t])}}function Mn(e,t){var n=e.stateNode;if(n===null)return null;var r=n[jt]||null;if(r===null)return null;n=r[t];a:switch(t){case`onClick`:case`onClickCapture`:case`onDoubleClick`:case`onDoubleClickCapture`:case`onMouseDown`:case`onMouseDownCapture`:case`onMouseMove`:case`onMouseMoveCapture`:case`onMouseUp`:case`onMouseUpCapture`:case`onMouseEnter`:(r=!r.disabled)||(e=e.type,r=e!==`button`&&e!==`input`&&e!==`select`&&e!==`textarea`),e=!r;break a;default:e=!1}if(e)return null;if(n&&typeof n!=`function`)throw Error(i(231,t,typeof n));return n}var Nn=!(typeof window>`u`||window.document===void 0||window.document.createElement===void 0),Pn=!1;if(Nn)try{var Fn={};Object.defineProperty(Fn,"passive",{get:function(){Pn=!0}}),window.addEventListener(`test`,Fn,Fn),window.removeEventListener(`test`,Fn,Fn)}catch{Pn=!1}var In=null,Ln=null,Rn=null;function zn(){if(Rn)return Rn;var e,t=Ln,n=t.length,r,i=`value`in In?In.value:In.textContent,a=i.length;for(e=0;e<n&&t[e]===i[e];e++);var o=n-e;for(r=1;r<=o&&t[n-r]===i[a-r];r++);return Rn=i.slice(e,1<r?1-r:void 0)}function Bn(e){var t=e.keyCode;return`charCode`in e?(e=e.charCode,e===0&&t===13&&(e=13)):e=t,e===10&&(e=13),32<=e||e===13?e:0}function Vn(){return!0}function Hn(){return!1}function Un(e){function t(t,n,r,i,a){for(var o in this._reactName=t,this._targetInst=r,this.type=n,this.nativeEvent=i,this.target=a,this.currentTarget=null,e)e.hasOwnProperty(o)&&(t=e[o],this[o]=t?t(i):i[o]);return this.isDefaultPrevented=(i.defaultPrevented==null?!1===i.returnValue:i.defaultPrevented)?Vn:Hn,this.isPropagationStopped=Hn,this}return w(t.prototype,{preventDefault:function(){this.defaultPrevented=!0;var e=this.nativeEvent;e&&(e.preventDefault?e.preventDefault():typeof e.returnValue!=`unknown`&&(e.returnValue=!1),this.isDefaultPrevented=Vn)},stopPropagation:function(){var e=this.nativeEvent;e&&(e.stopPropagation?e.stopPropagation():typeof e.cancelBubble!=`unknown`&&(e.cancelBubble=!0),this.isPropagationStopped=Vn)},persist:function(){},isPersistent:Vn}),t}var Wn={eventPhase:0,bubbles:0,cancelable:0,timeStamp:function(e){return e.timeStamp||Date.now()},defaultPrevented:0,isTrusted:0},Gn=Un(Wn),Kn=w({},Wn,{view:0,detail:0}),qn=Un(Kn),Jn,Yn,Xn,Zn=w({},Kn,{screenX:0,screenY:0,clientX:0,clientY:0,pageX:0,pageY:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,getModifierState:cr,button:0,buttons:0,relatedTarget:function(e){return e.relatedTarget===void 0?e.fromElement===e.srcElement?e.toElement:e.fromElement:e.relatedTarget},movementX:function(e){return`movementX`in e?e.movementX:(e!==Xn&&(Xn&&e.type===`mousemove`?(Jn=e.screenX-Xn.screenX,Yn=e.screenY-Xn.screenY):Yn=Jn=0,Xn=e),Jn)},movementY:function(e){return`movementY`in e?e.movementY:Yn}}),Qn=Un(Zn),$n=Un(w({},Zn,{dataTransfer:0})),er=Un(w({},Kn,{relatedTarget:0})),tr=Un(w({},Wn,{animationName:0,elapsedTime:0,pseudoElement:0})),nr=Un(w({},Wn,{clipboardData:function(e){return`clipboardData`in e?e.clipboardData:window.clipboardData}})),rr=Un(w({},Wn,{data:0})),ir={Esc:`Escape`,Spacebar:` `,Left:`ArrowLeft`,Up:`ArrowUp`,Right:`ArrowRight`,Down:`ArrowDown`,Del:`Delete`,Win:`OS`,Menu:`ContextMenu`,Apps:`ContextMenu`,Scroll:`ScrollLock`,MozPrintableKey:`Unidentified`},ar={8:`Backspace`,9:`Tab`,12:`Clear`,13:`Enter`,16:`Shift`,17:`Control`,18:`Alt`,19:`Pause`,20:`CapsLock`,27:`Escape`,32:` `,33:`PageUp`,34:`PageDown`,35:`End`,36:`Home`,37:`ArrowLeft`,38:`ArrowUp`,39:`ArrowRight`,40:`ArrowDown`,45:`Insert`,46:`Delete`,112:`F1`,113:`F2`,114:`F3`,115:`F4`,116:`F5`,117:`F6`,118:`F7`,119:`F8`,120:`F9`,121:`F10`,122:`F11`,123:`F12`,144:`NumLock`,145:`ScrollLock`,224:`Meta`},or={Alt:`altKey`,Control:`ctrlKey`,Meta:`metaKey`,Shift:`shiftKey`};function sr(e){var t=this.nativeEvent;return t.getModifierState?t.getModifierState(e):(e=or[e])?!!t[e]:!1}function cr(){return sr}var lr=Un(w({},Kn,{key:function(e){if(e.key){var t=ir[e.key]||e.key;if(t!==`Unidentified`)return t}return e.type===`keypress`?(e=Bn(e),e===13?`Enter`:String.fromCharCode(e)):e.type===`keydown`||e.type===`keyup`?ar[e.keyCode]||`Unidentified`:``},code:0,location:0,ctrlKey:0,shiftKey:0,altKey:0,metaKey:0,repeat:0,locale:0,getModifierState:cr,charCode:function(e){return e.type===`keypress`?Bn(e):0},keyCode:function(e){return e.type===`keydown`||e.type===`keyup`?e.keyCode:0},which:function(e){return e.type===`keypress`?Bn(e):e.type===`keydown`||e.type===`keyup`?e.keyCode:0}})),ur=Un(w({},Zn,{pointerId:0,width:0,height:0,pressure:0,tangentialPressure:0,tiltX:0,tiltY:0,twist:0,pointerType:0,isPrimary:0})),dr=Un(w({},Wn,{submitter:0})),fr=Un(w({},Kn,{touches:0,targetTouches:0,changedTouches:0,altKey:0,metaKey:0,ctrlKey:0,shiftKey:0,getModifierState:cr})),pr=Un(w({},Wn,{propertyName:0,elapsedTime:0,pseudoElement:0})),mr=Un(w({},Zn,{deltaX:function(e){return`deltaX`in e?e.deltaX:`wheelDeltaX`in e?-e.wheelDeltaX:0},deltaY:function(e){return`deltaY`in e?e.deltaY:`wheelDeltaY`in e?-e.wheelDeltaY:`wheelDelta`in e?-e.wheelDelta:0},deltaZ:0,deltaMode:0})),hr=Un(w({},Wn,{newState:0,oldState:0,source:0})),gr=[9,13,27,32],_r=Nn&&`CompositionEvent`in window,vr=null;Nn&&`documentMode`in document&&(vr=document.documentMode);var yr=Nn&&`TextEvent`in window&&!vr,br=Nn&&(!_r||vr&&8<vr&&11>=vr),xr=` `,Sr=!1;function Cr(e,t){switch(e){case`keyup`:return gr.indexOf(t.keyCode)!==-1;case`keydown`:return t.keyCode!==229;case`keypress`:case`mousedown`:case`focusout`:return!0;default:return!1}}function wr(e){return e=e.detail,typeof e==`object`&&`data`in e?e.data:null}var Tr=!1;function Er(e,t){switch(e){case`compositionend`:return wr(t);case`keypress`:return t.which===32?(Sr=!0,xr):null;case`textInput`:return e=t.data,e===xr&&Sr?null:e;default:return null}}function Dr(e,t){if(Tr)return e===`compositionend`||!_r&&Cr(e,t)?(e=zn(),Rn=Ln=In=null,Tr=!1,e):null;switch(e){case`paste`:return null;case`keypress`:if(!(t.ctrlKey||t.altKey||t.metaKey)||t.ctrlKey&&t.altKey){if(t.char&&1<t.char.length)return t.char;if(t.which)return String.fromCharCode(t.which)}return null;case`compositionend`:return br&&t.locale!==`ko`?null:t.data;default:return null}}var Or={color:!0,date:!0,datetime:!0,"datetime-local":!0,email:!0,month:!0,number:!0,password:!0,range:!0,search:!0,tel:!0,text:!0,time:!0,url:!0,week:!0};function kr(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t===`input`?!!Or[e.type]:t===`textarea`}function Ar(e,t,n,r){Dn?On?On.push(r):On=[r]:Dn=r,t=Jf(t,`onChange`),0<t.length&&(n=new Gn(`onChange`,`change`,null,n,r),e.push({event:n,listeners:t}))}var jr=null,Mr=null;function Nr(e){Vf(e,0)}function Pr(e){if(sn(Ht(e)))return e}function Fr(e,t){if(e===`change`)return t}var Ir=!1;if(Nn){var Lr;if(Nn){var Rr=`oninput`in document;if(!Rr){var zr=document.createElement(`div`);zr.setAttribute(`oninput`,`return;`),Rr=typeof zr.oninput==`function`}Lr=Rr}else Lr=!1;Ir=Lr&&(!document.documentMode||9<document.documentMode)}function Br(){jr&&(jr.detachEvent(`onpropertychange`,Vr),Mr=jr=null)}function Vr(e){if(e.propertyName===`value`&&Pr(Mr)){var t=[];Ar(t,Mr,e,En(e)),jn(Nr,t)}}function Hr(e,t,n){e===`focusin`?(Br(),jr=t,Mr=n,jr.attachEvent(`onpropertychange`,Vr)):e===`focusout`&&Br()}function Ur(e){if(e===`selectionchange`||e===`keyup`||e===`keydown`)return Pr(Mr)}function Wr(e,t){if(e===`click`)return Pr(t)}function Gr(e,t){if(e===`input`||e===`change`)return Pr(t)}function Kr(e,t){return e===t&&(e!==0||1/e==1/t)||e!==e&&t!==t}var qr=typeof Object.is==`function`?Object.is:Kr;function Jr(e,t){if(qr(e,t))return!0;if(typeof e!=`object`||!e||typeof t!=`object`||!t)return!1;var n=Object.keys(e),r=Object.keys(t);if(n.length!==r.length)return!1;for(r=0;r<n.length;r++){var i=n[r];if(!We.call(t,i)||!qr(e[i],t[i]))return!1}return!0}function Yr(e){if(e||=typeof document<`u`?document:void 0,e===void 0)return null;try{return e.activeElement||e.body}catch{return e.body}}function Xr(e){for(;e&&e.firstChild;)e=e.firstChild;return e}function Zr(e,t){var n=Xr(e);e=0;for(var r;n;){if(n.nodeType===3){if(r=e+n.textContent.length,e<=t&&r>=t)return{node:n,offset:t-e};e=r}a:{for(;n;){if(n.nextSibling){n=n.nextSibling;break a}n=n.parentNode}n=void 0}n=Xr(n)}}function Qr(e,t){return e&&t?e===t?!0:e&&e.nodeType===3?!1:t&&t.nodeType===3?Qr(e,t.parentNode):`contains`in e?e.contains(t):e.compareDocumentPosition?!!(e.compareDocumentPosition(t)&16):!1:!1}function $r(e){e=e!=null&&e.ownerDocument!=null&&e.ownerDocument.defaultView!=null?e.ownerDocument.defaultView:window;for(var t=Yr(e.document);t instanceof e.HTMLIFrameElement;){try{var n=typeof t.contentWindow.location.href==`string`}catch{n=!1}if(n)e=t.contentWindow;else break;t=Yr(e.document)}return t}function ei(e){var t=e&&e.nodeName&&e.nodeName.toLowerCase();return t&&(t===`input`&&(e.type===`text`||e.type===`search`||e.type===`tel`||e.type===`url`||e.type===`password`)||t===`textarea`||e.contentEditable===`true`)}var ti=Nn&&`documentMode`in document&&11>=document.documentMode,ni=null,ri=null,ii=null,ai=!1;function oi(e,t,n){var r=n.window===n?n.document:n.nodeType===9?n:n.ownerDocument;ai||ni==null||ni!==Yr(r)||(r=ni,`selectionStart`in r&&ei(r)?r={start:r.selectionStart,end:r.selectionEnd}:(r=(r.ownerDocument&&r.ownerDocument.defaultView||window).getSelection(),r={anchorNode:r.anchorNode,anchorOffset:r.anchorOffset,focusNode:r.focusNode,focusOffset:r.focusOffset}),ii&&Jr(ii,r)||(ii=r,r=Jf(ri,`onSelect`),0<r.length&&(t=new Gn(`onSelect`,`select`,null,t,n),e.push({event:t,listeners:r}),t.target=ni)))}function si(e,t){var n={};return n[e.toLowerCase()]=t.toLowerCase(),n[`Webkit`+e]=`webkit`+t,n[`Moz`+e]=`moz`+t,n}var ci={animationend:si(`Animation`,`AnimationEnd`),animationiteration:si(`Animation`,`AnimationIteration`),animationstart:si(`Animation`,`AnimationStart`),transitionrun:si(`Transition`,`TransitionRun`),transitionstart:si(`Transition`,`TransitionStart`),transitioncancel:si(`Transition`,`TransitionCancel`),transitionend:si(`Transition`,`TransitionEnd`)},li={},ui={};Nn&&(ui=document.createElement(`div`).style,`AnimationEvent`in window||(delete ci.animationend.animation,delete ci.animationiteration.animation,delete ci.animationstart.animation),`TransitionEvent`in window||delete ci.transitionend.transition);function di(e){if(li[e])return li[e];if(!ci[e])return e;var t=ci[e],n;for(n in t)if(t.hasOwnProperty(n)&&n in ui)return li[e]=t[n];return e}var fi=di(`animationend`),pi=di(`animationiteration`),mi=di(`animationstart`),hi=di(`transitionrun`),gi=di(`transitionstart`),_i=di(`transitioncancel`),vi=di(`transitionend`),yi=new Map,bi=`abort auxClick beforeToggle cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error fullscreenChange fullscreenError gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel`.split(` `);bi.push(`scrollEnd`);function xi(e,t){yi.set(e,t),qt(t,[e])}var Si=0;function Ci(e,t){if(e.name!=null&&e.name!==`auto`)return e.name;if(t.autoName!==null)return t.autoName;e=bd.identifierPrefix;var n=Si++;return e=`_`+e+`t_`+n.toString(32)+`_`,t.autoName=e}function wi(e){if(e==null||typeof e==`string`)return e;var t=null,n=Od;if(n!==null)for(var r=0;r<n.length;r++){var i=e[n[r]];if(i!=null){if(i===`none`)return`none`;t=t==null?i:t+(` `+i)}}return t??e.default}function Ti(e,t){return e=wi(e),t=wi(t),t==null?e===`auto`?null:e:t===`auto`?null:t}var Ei=typeof reportError==`function`?reportError:function(e){if(typeof window==`object`&&typeof window.ErrorEvent==`function`){var t=new window.ErrorEvent(`error`,{bubbles:!0,cancelable:!0,message:typeof e==`object`&&e&&typeof e.message==`string`?String(e.message):String(e),error:e});if(!window.dispatchEvent(t))return}else if(typeof process==`object`&&typeof process.emit==`function`){process.emit(`uncaughtException`,e);return}console.error(e)},Di=[],Oi=0,ki=0;function Ai(){for(var e=Oi,t=ki=Oi=0;t<e;){var n=Di[t];Di[t++]=null;var r=Di[t];Di[t++]=null;var i=Di[t];Di[t++]=null;var a=Di[t];if(Di[t++]=null,r!==null&&i!==null){var o=r.pending;o===null?i.next=i:(i.next=o.next,o.next=i),r.pending=i}a!==0&&Pi(n,i,a)}}function ji(e,t,n,r){Di[Oi++]=e,Di[Oi++]=t,Di[Oi++]=n,Di[Oi++]=r,ki|=r,e.lanes|=r,e=e.alternate,e!==null&&(e.lanes|=r)}function Mi(e,t,n,r){return ji(e,t,n,r),Fi(e)}function Ni(e,t){return ji(e,null,null,t),Fi(e)}function Pi(e,t,n){e.lanes|=n;var r=e.alternate;r!==null&&(r.lanes|=n);for(var i=!1,a=e.return;a!==null;)a.childLanes|=n,r=a.alternate,r!==null&&(r.childLanes|=n),a.tag===22&&(e=a.stateNode,e===null||e._visibility&1||(i=!0)),e=a,a=a.return;return e.tag===3?(a=e.stateNode,i&&t!==null&&(i=31-ot(n),e=a.hiddenUpdates,r=e[i],r===null?e[i]=[t]:r.push(t),t.lane=n|536870912),a):null}function Fi(e){if(50<kd)throw kd=0,Ad=null,Error(i(185));for(var t=e.return;t!==null;)e=t,t=e.return;return e.tag===3?e.stateNode:null}var Ii={};function Li(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.refCleanup=this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.subtreeFlags=this.flags=0,this.deletions=null,this.childLanes=this.lanes=0,this.alternate=null}function Ri(e,t,n,r){return new Li(e,t,n,r)}function zi(e){return e=e.prototype,!(!e||!e.isReactComponent)}function Bi(e,t){var n=e.alternate;return n===null?(n=Ri(e.tag,t,e.key,e.mode),n.elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.subtreeFlags=0,n.deletions=null),n.flags=e.flags&1206910976,n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n.refCleanup=e.refCleanup,n}function Vi(e,t){e.flags&=1206910978;var n=e.alternate;return n===null?(e.childLanes=0,e.lanes=t,e.child=null,e.subtreeFlags=0,e.memoizedProps=null,e.memoizedState=null,e.updateQueue=null,e.dependencies=null,e.stateNode=null):(e.childLanes=n.childLanes,e.lanes=n.lanes,e.child=n.child,e.subtreeFlags=0,e.deletions=null,e.memoizedProps=n.memoizedProps,e.memoizedState=n.memoizedState,e.updateQueue=n.updateQueue,e.type=n.type,t=n.dependencies,e.dependencies=t===null?null:{lanes:t.lanes,firstContext:t.firstContext}),e}function Hi(e,t,n,r,a,o){var s=0;if(r=e,typeof r==`function`)zi(r)&&(s=1);else if(typeof r==`string`)s=qm(e,n,ke.current)?26:e===`html`||e===`head`||e===`body`?27:5;else a:switch(r){case ge:return e=Ri(31,n,t,a),e.elementType=ge,e.lanes=o,e;case oe:return Ui(n.children,a,o,t);case se:s=8,a|=24;break;case ce:return e=Ri(12,n,t,a|2),e.elementType=ce,e.lanes=o,e;case fe:return e=Ri(13,n,t,a),e.elementType=fe,e.lanes=o,e;case pe:return e=Ri(19,n,t,a),e.elementType=pe,e.lanes=o,e;case _e:case T:return e=a|32,e=Ri(30,n,t,e),e.elementType=T,e.lanes=o,e.stateNode={autoName:null,paired:null,clones:null,ref:null},e;default:if(typeof r==`object`&&r)switch(r.$$typeof){case ue:s=10;break a;case le:s=9;break a;case de:s=11;break a;case me:s=14;break a;case he:s=16,r=null;break a}s=29,n=Error(i(130,e===null?`null`:typeof e,``)),r=null}return t=Ri(s,n,t,a),t.elementType=e,t.type=r,t.lanes=o,t}function Ui(e,t,n,r){return e=Ri(7,e,r,t),e.lanes=n,e}function Wi(e,t,n){return e=Ri(6,e,null,t),e.lanes=n,e}function Gi(e){var t=Ri(18,null,null,0);return t.stateNode=e,t}function Ki(e,t,n){return t=Ri(4,e.children===null?[]:e.children,e.key,t),t.lanes=n,t.stateNode={containerInfo:e.containerInfo,pendingChildren:null,implementation:e.implementation},t}var qi=new WeakMap;function Ji(e,t){if(typeof e==`object`&&e){var n=qi.get(e);return n===void 0?(t={value:e,source:t,stack:Ue(t)},qi.set(e,t),t):n}return{value:e,source:t,stack:Ue(t)}}var Yi=[],Xi=0,Zi=null,Qi=0,$i=[],ea=0,ta=null,na=1,ra=``;function ia(e,t){Yi[Xi++]=Qi,Yi[Xi++]=Zi,Zi=e,Qi=t}function aa(e,t,n){$i[ea++]=na,$i[ea++]=ra,$i[ea++]=ta,ta=e;var r=na;e=ra;var i=32-ot(r)-1;r&=~(1<<i),n+=1;var a=32-ot(t)+i;if(30<a){var o=i-i%5;a=(r&(1<<o)-1).toString(32),r>>=o,i-=o,na=1<<32-ot(t)+i|n<<i|r,ra=a+e}else na=1<<a|n<<i|r,ra=e}function oa(e){e.return!==null&&(ia(e,1),aa(e,1,0))}function sa(e){for(;e===Zi;)Zi=Yi[--Xi],Yi[Xi]=null,Qi=Yi[--Xi],Yi[Xi]=null;for(;e===ta;)ta=$i[--ea],$i[ea]=null,ra=$i[--ea],$i[ea]=null,na=$i[--ea],$i[ea]=null}function ca(e,t){$i[ea++]=na,$i[ea++]=ra,$i[ea++]=ta,na=t.id,ra=t.overflow,ta=e}var la=null,P=null,F=!1,ua=null,da=!1,fa=Error(i(519));function pa(e){throw ya(Ji(Error(i(418,1<arguments.length&&arguments[1]!==void 0&&arguments[1]?`text`:`HTML`,``)),e)),fa}function ma(e){var t=e.stateNode,n=e.type,r=e.memoizedProps;switch(t[At]=e,t[jt]=r,n){case`dialog`:Q(`cancel`,t),Q(`close`,t);break;case`iframe`:case`object`:case`embed`:Q(`load`,t);break;case`video`:case`audio`:for(n=0;n<zf.length;n++)Q(zf[n],t);break;case`source`:Q(`error`,t);break;case`img`:case`image`:case`link`:Q(`error`,t),Q(`load`,t);break;case`details`:Q(`toggle`,t);break;case`input`:Q(`invalid`,t),dn(t,r.value,r.defaultValue,r.checked,r.defaultChecked,r.type,r.name,!0);break;case`select`:Q(`invalid`,t);break;case`textarea`:Q(`invalid`,t),hn(t,r.value,r.defaultValue,r.children)}n=r.children,typeof n!=`string`&&typeof n!=`number`&&typeof n!=`bigint`||t.textContent===``+n||!0===r.suppressHydrationWarning||ep(t.textContent,n)?(r.popover!=null&&(Q(`beforetoggle`,t),Q(`toggle`,t)),r.onScroll!=null&&Q(`scroll`,t),r.onScrollEnd!=null&&Q(`scrollend`,t),r.onClick!=null&&(t.onclick=wn),t=!0):t=!1,t||pa(e,!0)}function ha(e){for(la=e.return;la;)switch(la.tag){case 5:case 31:case 13:da=!1;return;case 27:case 3:da=!0;return;default:la=la.return}}function ga(e){if(e!==la)return!1;if(!F)return ha(e),F=!0,!1;var t=e.tag,n;if((n=t!==3&&t!==27)&&((n=t===5)&&(n=e.type,n=n===`form`||n===`button`||pp(e.type,e.memoizedProps)),n=!n),n&&P&&pa(e),ha(e),t===13){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));P=dm(e)}else if(t===31){if(e=e.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(317));P=dm(e)}else t===27?(t=P,Sp(e.type)?(e=um,um=null,P=e):P=t):P=la?lm(e.stateNode.nextSibling):null;return!0}function _a(){P=la=null,F=!1}function va(){var e=ua;return e!==null&&(pd===null?pd=e:pd.push.apply(pd,e),ua=null),e}function ya(e){ua===null?ua=[e]:ua.push(e)}var ba=De(null),xa=null,Sa=null;function Ca(e,t,n){k(ba,t._currentValue),t._currentValue=n}function wa(e){e._currentValue=ba.current,Oe(ba)}function Ta(e,t,n){for(;e!==null;){var r=e.alternate;if((e.childLanes&t)===t?r!==null&&(r.childLanes&t)!==t&&(r.childLanes|=t):(e.childLanes|=t,r!==null&&(r.childLanes|=t)),e===n)break;e=e.return}}function Ea(e,t,n,r){var a=e.child;for(a!==null&&(a.return=e);a!==null;){var o=a.dependencies;if(o!==null){var s=a.child;o=o.firstContext;a:for(;o!==null;){var c=o;o=a;for(var l=0;l<t.length;l++)if(c.context===t[l]){o.lanes|=n,c=o.alternate,c!==null&&(c.lanes|=n),Ta(o.return,n,e),r||(s=null);break a}o=c.next}}else if(a.tag===18){if(s=a.return,s===null)throw Error(i(341));s.lanes|=n,o=s.alternate,o!==null&&(o.lanes|=n),Ta(s,n,e),s=null}else a.tag===13&&a.memoizedState!==null&&a.memoizedState.dehydrated===null?(a.lanes|=n,s=a.alternate,s!==null&&(s.lanes|=n),Ta(a.return,n,e),s=a.child,s=s===null?null:s.sibling):s=a.child;if(s!==null)s.return=a;else for(s=a;s!==null;){if(s===e){s=null;break}if(a=s.sibling,a!==null){a.return=s.return,s=a;break}s=s.return}a=s}}function Da(e,t,n,r){e=null;for(var a=t,o=!1;a!==null;){if(!o){if(a.flags&524288)o=!0;else if(a.flags&262144)break}if(a.tag===10){var s=a.alternate;if(s===null)throw Error(i(387));if(s=s.memoizedProps,s!==null){var c=a.type;qr(a.pendingProps.value,s.value)||(e===null?e=[c]:e.push(c))}}else if(a===Me.current){if(s=a.alternate,s===null)throw Error(i(387));s.memoizedState.memoizedState!==a.memoizedState.memoizedState&&(e===null?e=[sh]:e.push(sh))}a=a.return}return e!==null&&Ea(t,e,n,r),t.flags|=262144,e!==null}function Oa(e){for(e=e.firstContext;e!==null;){if(!qr(e.context._currentValue,e.memoizedValue))return!0;e=e.next}return!1}function ka(e){xa=e,Sa=null,e=e.dependencies,e!==null&&(e.firstContext=null)}function Aa(e){return Ma(xa,e)}function ja(e,t){return xa===null&&ka(e),Ma(e,t)}function Ma(e,t){var n=t._currentValue;if(t={context:t,memoizedValue:n,next:null},Sa===null){if(e===null)throw Error(i(308));Sa=t,e.dependencies={lanes:0,firstContext:t},e.flags|=524288}else Sa=Sa.next=t;return n}var Na=typeof AbortController<`u`?AbortController:function(){var e=[],t=this.signal={aborted:!1,addEventListener:function(t,n){e.push(n)}};this.abort=function(){t.aborted=!0,e.forEach(function(e){return e()})}},Pa=t.unstable_scheduleCallback,Fa=t.unstable_NormalPriority,I={$$typeof:ue,Consumer:null,Provider:null,_currentValue:null,_currentValue2:null,_threadCount:0};function Ia(){return{controller:new Na,data:new Map,refCount:0}}function La(e){e.refCount--,e.refCount===0&&Pa(Fa,function(){e.controller.abort()})}function Ra(e,t){if(e.pendingLanes&4194048){var n=e.transitionTypes;for(n===null&&(n=e.transitionTypes=[]),e=0;e<t.length;e++){var r=t[e];n.indexOf(r)===-1&&n.push(r)}}}var za=null;function Ba(e){var t=e.transitionTypes;return e.transitionTypes=null,t}var Va=null,Ha=0,Ua=0,Wa=null;function Ga(e,t){if(Va===null){var n=Va=[];Ha=0,Ua=Pf(),Wa={status:`pending`,value:void 0,then:function(e){n.push(e)}}}return Ha++,t.then(Ka,Ka),t}function Ka(){if(--Ha===0&&(za=null,Va!==null)){Wa!==null&&(Wa.status=`fulfilled`);var e=Va;Va=null,Ua=0,Wa=null;for(var t=0;t<e.length;t++)(0,e[t])()}}function qa(e,t){var n=[],r={status:`pending`,value:null,reason:null,then:function(e){n.push(e)}};return e.then(function(){r.status=`fulfilled`,r.value=t;for(var e=0;e<n.length;e++)(0,n[e])(t)},function(e){for(r.status=`rejected`,r.reason=e,e=0;e<n.length;e++)(0,n[e])(void 0)}),r}var Ja=D.S;D.S=function(e,t){if(gd=Je(),typeof t==`object`&&t&&typeof t.then==`function`&&Ga(e,t),za!==null)for(var n=bf;n!==null;)Ra(n,za),n=n.next;if(n=e.types,n!==null){for(var r=bf;r!==null;)Ra(r,n),r=r.next;if(Ua!==0){r=za,r===null&&(r=za=[]);for(var i=0;i<n.length;i++){var a=n[i];r.indexOf(a)===-1&&r.push(a)}}}Ja!==null&&Ja(e,t)};var Ya=De(null);function Xa(){var e=Ya.current;return e===null?G.pooledCache:e}function Za(e,t){t===null?k(Ya,Ya.current):k(Ya,t.pool)}function Qa(){var e=Xa();return e===null?null:{parent:I._currentValue,pool:e}}var $a=Error(i(460)),eo=Error(i(474)),to=Error(i(542)),no={then:function(){}};function ro(e){return e=e.status,e===`fulfilled`||e===`rejected`}function io(e,t,n){switch(n=e[n],n===void 0?e.push(t):n!==t&&(t.then(wn,wn),t=n),t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,co(e),e===void 0&&!(`reason`in t)?Error(i(600)):e;default:if(typeof t.status==`string`)t.then(wn,wn);else{if(e=G,e!==null&&100<e.shellSuspendCounter)throw Error(i(482));e=t,e.status=`pending`,e.then(function(e){if(t.status===`pending`){var n=t;n.status=`fulfilled`,n.value=e}},function(e){if(t.status===`pending`){var n=t;n.status=`rejected`,n.reason=e}})}switch(t.status){case`fulfilled`:return t.value;case`rejected`:throw e=t.reason,co(e),e}throw oo=t,$a}}function ao(e){try{var t=e._init;return t(e._payload)}catch(e){throw typeof e==`object`&&e&&typeof e.then==`function`?(oo=e,$a):e}}var oo=null;function so(){if(oo===null)throw Error(i(459));var e=oo;return oo=null,e}function co(e){if(e===$a||e===to)throw Error(i(483))}var lo=null,uo=0;function fo(e){var t=uo;return uo+=1,lo===null&&(lo=[]),io(lo,e,t)}function po(e,t){t=t.props.ref,e.ref=t===void 0?null:t}function mo(e,t){throw t.$$typeof===re?Error(i(525)):(e=Object.prototype.toString.call(t),Error(i(31,e===`[object Object]`?`object with keys {`+Object.keys(t).join(`, `)+`}`:e)))}function ho(e){function t(t,n){if(e){var r=t.deletions;r===null?(t.deletions=[n],t.flags|=16):r.push(n)}}function n(n,r){if(!e)return null;for(;r!==null;)t(n,r),r=r.sibling;return null}function r(e){for(var t=new Map;e!==null;)e.key===null?t.set(e.index,e):t.set(e.key,e),e=e.sibling;return t}function a(e,t){return e=Bi(e,t),e.index=0,e.sibling=null,e}function o(t,n,r){return t.index=r,e?(r=t.alternate,r===null?(t.flags|=134217730,n):(r=r.index,r<n?(t.flags|=2,n):r)):(t.flags|=1048576,n)}function s(t){return e&&t.alternate===null&&(t.flags|=134217730),t}function c(e,t,n,r){return t===null||t.tag!==6?(t=Wi(n,e.mode,r),t.return=e,t):(t=a(t,n),t.return=e,t)}function l(e,t,n,r){var i=n.type;return i===oe?(e=d(e,t,n.props.children,r,n.key),po(e,n),e):t!==null&&(t.elementType===i||typeof i==`object`&&i&&i.$$typeof===he&&ao(i)===t.type)?(t=a(t,n.props),po(t,n),t.return=e,t):(t=Hi(n.type,n.key,n.props,null,e.mode,r),po(t,n),t.return=e,t)}function u(e,t,n,r){return t===null||t.tag!==4||t.stateNode.containerInfo!==n.containerInfo||t.stateNode.implementation!==n.implementation?(t=Ki(n,e.mode,r),t.return=e,t):(t=a(t,n.children||[]),t.return=e,t)}function d(e,t,n,r,i){return t===null||t.tag!==7?(t=Ui(n,e.mode,r,i),t.return=e,t):(t=a(t,n),t.return=e,t)}function f(e,t,n){if(typeof t==`string`&&t!==``||typeof t==`number`||typeof t==`bigint`)return t=Wi(``+t,e.mode,n),t.return=e,t;if(typeof t==`object`&&t){switch(t.$$typeof){case ie:return n=Hi(t.type,t.key,t.props,null,e.mode,n),po(n,t),n.return=e,n;case ae:return t=Ki(t,e.mode,n),t.return=e,t;case he:return t=ao(t),f(e,t,n)}if(Ce(t)||be(t))return t=Ui(t,e.mode,n,null),t.return=e,t;if(typeof t.then==`function`)return f(e,fo(t),n);if(t.$$typeof===ue)return f(e,ja(e,t),n);mo(e,t)}return null}function p(e,t,n,r){var i=t===null?null:t.key;if(typeof n==`string`&&n!==``||typeof n==`number`||typeof n==`bigint`)return i===null?c(e,t,``+n,r):null;if(typeof n==`object`&&n){switch(n.$$typeof){case ie:return n.key===i?l(e,t,n,r):null;case ae:return n.key===i?u(e,t,n,r):null;case he:return n=ao(n),p(e,t,n,r)}if(Ce(n)||be(n))return i===null?d(e,t,n,r,null):null;if(typeof n.then==`function`)return p(e,t,fo(n),r);if(n.$$typeof===ue)return p(e,t,ja(e,n),r);mo(e,n)}return null}function m(e,t,n,r,i){if(typeof r==`string`&&r!==``||typeof r==`number`||typeof r==`bigint`)return e=e.get(n)||null,c(t,e,``+r,i);if(typeof r==`object`&&r){switch(r.$$typeof){case ie:return e=e.get(r.key===null?n:r.key)||null,l(t,e,r,i);case ae:return e=e.get(r.key===null?n:r.key)||null,u(t,e,r,i);case he:return r=ao(r),m(e,t,n,r,i)}if(Ce(r)||be(r))return e=e.get(n)||null,d(t,e,r,i,null);if(typeof r.then==`function`)return m(e,t,n,fo(r),i);if(r.$$typeof===ue)return m(e,t,n,ja(t,r),i);mo(t,r)}return null}function h(i,a,s,c){for(var l=null,u=null,d=a,h=a=0,g=null;d!==null&&h<s.length;h++){d.index>h?(g=d,d=null):g=d.sibling;var _=p(i,d,s[h],c);if(_===null){d===null&&(d=g);break}e&&d&&_.alternate===null&&t(i,d),a=o(_,a,h),u===null?l=_:u.sibling=_,u=_,d=g}if(h===s.length)return n(i,d),F&&ia(i,h),l;if(d===null){for(;h<s.length;h++)d=f(i,s[h],c),d!==null&&(a=o(d,a,h),u===null?l=d:u.sibling=d,u=d);return F&&ia(i,h),l}for(d=r(d);h<s.length;h++)g=m(d,i,h,s[h],c),g!==null&&(e&&(_=g.alternate,_!==null&&d.delete(_.key===null?h:_.key)),a=o(g,a,h),u===null?l=g:u.sibling=g,u=g);return e&&d.forEach(function(e){return t(i,e)}),F&&ia(i,h),l}function g(a,s,c,l){if(c==null)throw Error(i(151));for(var u=null,d=null,h=s,g=s=0,_=null,v=c.next();h!==null&&!v.done;g++,v=c.next()){h.index>g?(_=h,h=null):_=h.sibling;var y=p(a,h,v.value,l);if(y===null){h===null&&(h=_);break}e&&h&&y.alternate===null&&t(a,h),s=o(y,s,g),d===null?u=y:d.sibling=y,d=y,h=_}if(v.done)return n(a,h),F&&ia(a,g),u;if(h===null){for(;!v.done;g++,v=c.next())v=f(a,v.value,l),v!==null&&(s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return F&&ia(a,g),u}for(h=r(h);!v.done;g++,v=c.next())v=m(h,a,g,v.value,l),v!==null&&(e&&(_=v.alternate,_!==null&&h.delete(_.key===null?g:_.key)),s=o(v,s,g),d===null?u=v:d.sibling=v,d=v);return e&&h.forEach(function(e){return t(a,e)}),F&&ia(a,g),u}function _(e,r,o,c){if(typeof o==`object`&&o&&o.type===oe&&o.key===null&&o.props.ref===void 0&&(o=o.props.children),typeof o==`object`&&o){switch(o.$$typeof){case ie:a:{for(var l=o.key;r!==null;){if(r.key===l){if(l=o.type,l===oe){if(r.tag===7){n(e,r.sibling),c=a(r,o.props.children),po(c,o),c.return=e,e=c;break a}}else if(r.elementType===l||typeof l==`object`&&l&&l.$$typeof===he&&ao(l)===r.type){n(e,r.sibling),c=a(r,o.props),po(c,o),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}o.type===oe?(c=Ui(o.props.children,e.mode,c,o.key),po(c,o),c.return=e,e=c):(c=Hi(o.type,o.key,o.props,null,e.mode,c),po(c,o),c.return=e,e=c)}return s(e);case ae:a:{for(l=o.key;r!==null;){if(r.key===l){if(r.tag===4&&r.stateNode.containerInfo===o.containerInfo&&r.stateNode.implementation===o.implementation){n(e,r.sibling),c=a(r,o.children||[]),c.return=e,e=c;break a}n(e,r);break}t(e,r),r=r.sibling}c=Ki(o,e.mode,c),c.return=e,e=c}return s(e);case he:return o=ao(o),_(e,r,o,c)}if(Ce(o))return h(e,r,o,c);if(be(o)){if(l=be(o),typeof l!=`function`)throw Error(i(150));return o=l.call(o),g(e,r,o,c)}if(typeof o.then==`function`)return _(e,r,fo(o),c);if(o.$$typeof===ue)return _(e,r,ja(e,o),c);mo(e,o)}return typeof o==`string`&&o!==``||typeof o==`number`||typeof o==`bigint`?(o=``+o,r!==null&&r.tag===6?(n(e,r.sibling),c=a(r,o),c.return=e,e=c):(n(e,r),c=Wi(o,e.mode,c),c.return=e,e=c),s(e)):n(e,r)}return function(e,t,n,r){try{uo=0;var i=_(e,t,n,r);return lo=null,i}catch(t){if(t===$a||t===to)throw t;var a=Ri(29,t,null,e.mode);return a.lanes=r,a.return=e,a}}}var go=ho(!0),_o=ho(!1),vo=!1;function yo(e){e.updateQueue={baseState:e.memoizedState,firstBaseUpdate:null,lastBaseUpdate:null,shared:{pending:null,lanes:0,hiddenCallbacks:null},callbacks:null}}function bo(e,t){e=e.updateQueue,t.updateQueue===e&&(t.updateQueue={baseState:e.baseState,firstBaseUpdate:e.firstBaseUpdate,lastBaseUpdate:e.lastBaseUpdate,shared:e.shared,callbacks:null})}function xo(e){return{lane:e,tag:0,payload:null,callback:null,next:null}}function So(e,t,n){var r=e.updateQueue;if(r===null)return null;if(r=r.shared,W&2){var i=r.pending;return i===null?t.next=t:(t.next=i.next,i.next=t),r.pending=t,t=Fi(e),Pi(e,null,n),t}return ji(e,r,t,n),Fi(e)}function Co(e,t,n){if(t=t.updateQueue,t!==null&&(t=t.shared,n&4194048)){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Ct(e,n)}}function wo(e,t){var n=e.updateQueue,r=e.alternate;if(r!==null&&(r=r.updateQueue,n===r)){var i=null,a=null;if(n=n.firstBaseUpdate,n!==null){do{var o={lane:n.lane,tag:n.tag,payload:n.payload,callback:null,next:null};a===null?i=a=o:a=a.next=o,n=n.next}while(n!==null);a===null?i=a=t:a=a.next=t}else i=a=t;n={baseState:r.baseState,firstBaseUpdate:i,lastBaseUpdate:a,shared:r.shared,callbacks:r.callbacks},e.updateQueue=n;return}e=n.lastBaseUpdate,e===null?n.firstBaseUpdate=t:e.next=t,n.lastBaseUpdate=t}var To=!1;function Eo(){if(To){var e=Wa;if(e!==null)throw e}}function Do(e,t,n,r){To=!1;var i=e.updateQueue;vo=!1;var a=i.firstBaseUpdate,o=i.lastBaseUpdate,s=i.shared.pending;if(s!==null){i.shared.pending=null;var c=s,l=c.next;c.next=null,o===null?a=l:o.next=l,o=c;var u=e.alternate;u!==null&&(u=u.updateQueue,s=u.lastBaseUpdate,s!==o&&(s===null?u.firstBaseUpdate=l:s.next=l,u.lastBaseUpdate=c))}if(a!==null){var d=i.baseState;o=0,u=l=c=null,s=a;do{var f=s.lane&-536870913,p=f!==s.lane;if(p?(q&f)===f:(r&f)===f){f!==0&&f===Ua&&(To=!0),u!==null&&(u=u.next={lane:0,tag:s.tag,payload:s.payload,callback:null,next:null});a:{var m=e,h=s;f=t;var g=n;switch(h.tag){case 1:if(m=h.payload,typeof m==`function`){d=m.call(g,d,f);break a}d=m;break a;case 3:m.flags=m.flags&-65537|128;case 0:if(m=h.payload,f=typeof m==`function`?m.call(g,d,f):m,f==null)break a;d=w({},d,f);break a;case 2:vo=!0}}f=s.callback,f!==null&&(e.flags|=64,p&&(e.flags|=8192),p=i.callbacks,p===null?i.callbacks=[f]:p.push(f))}else p={lane:f,tag:s.tag,payload:s.payload,callback:s.callback,next:null},u===null?(l=u=p,c=d):u=u.next=p,o|=f;if(s=s.next,s===null){if(s=i.shared.pending,s===null)break;p=s,s=p.next,p.next=null,i.lastBaseUpdate=p,i.shared.pending=null}}while(1);u===null&&(c=d),i.baseState=c,i.firstBaseUpdate=l,i.lastBaseUpdate=u,a===null&&(i.shared.lanes=0),sd|=o,e.lanes=o,e.memoizedState=d}}function Oo(e,t){if(typeof e!=`function`)throw Error(i(191,e));e.call(t)}function ko(e,t){var n=e.callbacks;if(n!==null)for(e.callbacks=null,e=0;e<n.length;e++)Oo(n[e],t)}var Ao=De(null),jo=De(0);function Mo(e,t){e=od,k(jo,e),k(Ao,t),od=e|t.baseLanes}function No(){k(jo,od),k(Ao,Ao.current)}function Po(){od=jo.current,Oe(Ao),Oe(jo)}var Fo=De(null),Io=null;function Lo(e){var t=e.alternate;k(Ho,Ho.current&1),k(Fo,e),Io===null&&(t===null||Ao.current!==null||t.memoizedState!==null)&&(Io=e)}function Ro(e){k(Ho,Ho.current),k(Fo,e),Io===null&&(Io=e)}function zo(e){e.tag===22?(k(Ho,Ho.current),k(Fo,e),Io===null&&(Io=e)):Bo()}function Bo(){k(Ho,Ho.current),k(Fo,Fo.current)}function Vo(e){Oe(Fo),Io===e&&(Io=null),Oe(Ho)}var Ho=De(0);function Uo(e,t){k(Fo,Fo.current),k(Ho,t)}function Wo(e){Oe(Ho),Oe(Fo),Io===e&&(Io=null)}function Go(e){for(var t=e;t!==null;){if(t.tag===13){var n=t.memoizedState;if(n!==null&&(n=n.dehydrated,n===null||om(n)||sm(n)))return t}else if(t.tag===19&&t.memoizedProps.revealOrder!==`independent`){if(t.flags&128)return t}else if(t.child!==null){t.child.return=t,t=t.child;continue}if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return null;t=t.return}t.sibling.return=t.return,t=t.sibling}return null}var Ko=0,L=null,R=null,qo=null,Jo=!1,Yo=!1,Xo=!1,Zo=0,Qo=0,$o=null,es=0;function z(){throw Error(i(321))}function ts(e,t){if(t===null)return!1;for(var n=0;n<t.length&&n<e.length;n++)if(!qr(e[n],t[n]))return!1;return!0}function ns(e,t,n,r,i,a){return Ko=a,L=t,t.memoizedState=null,t.updateQueue=null,t.lanes=0,D.H=e===null||e.memoizedState===null?vc:yc,Xo=!1,a=n(r,i),Xo=!1,Yo&&(a=is(t,n,r,i)),rs(e),a}function rs(e){D.H=_c;var t=R!==null&&R.next!==null;if(Ko=0,qo=R=L=null,Jo=!1,Qo=0,$o=null,t)throw Error(i(300));e===null||Ic||(e=e.dependencies,e!==null&&Oa(e)&&(Ic=!0))}function is(e,t,n,r){L=e;var a=0;do{if(Yo&&($o=null),Qo=0,Yo=!1,25<=a)throw Error(i(301));if(a+=1,qo=R=null,e.updateQueue!=null){var o=e.updateQueue;o.lastEffect=null,o.events=null,o.stores=null,o.memoCache!=null&&(o.memoCache.index=0)}D.H=bc,o=t(n,r)}while(Yo);return o}function as(){var e=D.H,t=e.useState()[0];return t=typeof t.then==`function`?ds(t):t,e=e.useState()[0],(R===null?null:R.memoizedState)!==e&&(L.flags|=1024),t}function os(){var e=Zo!==0;return Zo=0,e}function ss(e,t,n){t.updateQueue=e.updateQueue,t.flags&=-2053,e.lanes&=~n}function cs(e){if(Jo){for(e=e.memoizedState;e!==null;){var t=e.queue;t!==null&&(t.pending=null),e=e.next}Jo=!1}Ko=0,qo=R=L=null,Yo=!1,Qo=Zo=0,$o=null}function ls(){var e={memoizedState:null,baseState:null,baseQueue:null,queue:null,next:null};return qo===null?L.memoizedState=qo=e:qo=qo.next=e,qo}function B(){if(R===null){var e=L.alternate;e=e===null?null:e.memoizedState}else e=R.next;var t=qo===null?L.memoizedState:qo.next;if(t!==null)qo=t,R=e;else{if(e===null)throw L.alternate===null?Error(i(467)):Error(i(310));R=e,e={memoizedState:R.memoizedState,baseState:R.baseState,baseQueue:R.baseQueue,queue:R.queue,next:null},qo===null?L.memoizedState=qo=e:qo=qo.next=e}return qo}function us(){return{lastEffect:null,events:null,stores:null,memoCache:null}}function ds(e){var t=Qo;return Qo+=1,$o===null&&($o=[]),e=io($o,e,t),t=L,(qo===null?t.memoizedState:qo.next)===null&&(t=t.alternate,D.H=t===null||t.memoizedState===null?vc:yc),e}function fs(e){if(typeof e==`object`&&e){if(typeof e.then==`function`)return ds(e);if(e.$$typeof===E)return;if(e.$$typeof===ue)return Aa(e)}throw Error(i(438,String(e)))}function ps(e){var t=null,n=L.updateQueue;if(n!==null&&(t=n.memoCache),t==null){var r=L.alternate;r!==null&&(r=r.updateQueue,r!==null&&(r=r.memoCache,r!=null&&(t={data:r.data.map(function(e){return e.slice()}),index:0})))}if(t??={data:[],index:0},n===null&&(n=us(),L.updateQueue=n),n.memoCache=t,n=t.data[t.index],n===void 0)for(n=t.data[t.index]=Array(e),r=0;r<e;r++)n[r]=ve;return t.index++,n}function ms(e,t){return typeof t==`function`?t(e):t}function hs(e){return gs(B(),R,e)}function gs(e,t,n){var r=e.queue;if(r===null)throw Error(i(311));r.lastRenderedReducer=n;var a=e.baseQueue,o=r.pending;if(o!==null){if(a!==null){var s=a.next;a.next=o.next,o.next=s}t.baseQueue=a=o,r.pending=null}if(o=e.baseState,a===null)e.memoizedState=o;else{t=a.next;var c=s=null,l=null,u=t,d=!1;do{var f=u.lane&-536870913;if(f===u.lane?(Ko&f)===f:(q&f)===f){var p=u.revertLane;if(p===0)l!==null&&(l=l.next={lane:0,revertLane:0,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null}),f===Ua&&(d=!0);else if((Ko&p)===p){u=u.next,p===Ua&&(d=!0);continue}else f={lane:0,revertLane:u.revertLane,gesture:null,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=f,s=o):l=l.next=f,L.lanes|=p,sd|=p;f=u.action,Xo&&n(o,f),o=u.hasEagerState?u.eagerState:n(o,f)}else p={lane:f,revertLane:u.revertLane,gesture:u.gesture,action:u.action,hasEagerState:u.hasEagerState,eagerState:u.eagerState,next:null},l===null?(c=l=p,s=o):l=l.next=p,L.lanes|=f,sd|=f;u=u.next}while(u!==null&&u!==t);if(l===null?s=o:l.next=c,!qr(o,e.memoizedState)&&(Ic=!0,d&&(n=Wa,n!==null)))throw n;e.memoizedState=o,e.baseState=s,e.baseQueue=l,r.lastRenderedState=o}return a===null&&(r.lanes=0),[e.memoizedState,r.dispatch]}function _s(e){var t=B(),n=t.queue;if(n===null)throw Error(i(311));n.lastRenderedReducer=e;var r=n.dispatch,a=n.pending,o=t.memoizedState;if(a!==null){n.pending=null;var s=a=a.next;do o=e(o,s.action),s=s.next;while(s!==a);qr(o,t.memoizedState)||(Ic=!0),t.memoizedState=o,t.baseQueue===null&&(t.baseState=o),n.lastRenderedState=o}return[o,r]}function vs(e,t,n){var r=L,a=B(),o=F;if(o){if(n===void 0)throw Error(i(407));n=n()}else n=t();var s=!qr((R||a).memoizedState,n);if(s&&(a.memoizedState=n,Ic=!0),a=a.queue,Us(xs.bind(null,r,a,e),[e]),e=a.getSnapshot!==t||s||qo!==null&&!!(qo.memoizedState.tag&1),Rs(e?9:8,{destroy:void 0},bs.bind(null,r,a,n,t),null),e){if(r.flags|=2048,G===null)throw Error(i(349));o||Ko&127||ys(r,t,n)}return n}function ys(e,t,n){e.flags|=16384,e={getSnapshot:t,value:n},t=L.updateQueue,t===null?(t=us(),L.updateQueue=t,t.stores=[e]):(n=t.stores,n===null?t.stores=[e]:n.push(e))}function bs(e,t,n,r){t.value=n,t.getSnapshot=r,Ss(t)&&Cs(e)}function xs(e,t,n){return n(function(){Ss(t)&&Cs(e)})}function Ss(e){var t=e.getSnapshot;e=e.value;try{var n=t();return!qr(e,n)}catch{return!0}}function Cs(e){var t=Ni(e,2);t!==null&&Pd(t,e,2)}function ws(e){var t=ls();if(typeof e==`function`){var n=e;if(e=n(),Xo){at(!0);try{n()}finally{at(!1)}}}return t.memoizedState=t.baseState=e,t.queue={pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:e},t}function Ts(e,t,n,r){return e.baseState=n,gs(e,R,typeof r==`function`?r:ms)}function Es(e,t,n,r,a){if(mc(e))throw Error(i(485));if(e=t.action,e!==null){var o={payload:a,action:e,next:null,isTransition:!0,status:`pending`,value:null,reason:null,listeners:[],then:function(e){o.listeners.push(e)}};D.T===null?o.isTransition=!1:n(!0),r(o),n=t.pending,n===null?(o.next=t.pending=o,Ds(t,o)):(o.next=n.next,t.pending=n.next=o)}}function Ds(e,t){var n=t.action,r=t.payload,i=e.state;if(t.isTransition){var a=D.T,o={};o.types=a===null?null:a.types,D.T=o;try{var s=n(i,r),c=D.S;c!==null&&c(o,s),Os(e,t,s)}catch(n){As(e,t,n)}finally{a!==null&&o.types!==null&&(a.types=o.types),D.T=a}}else try{a=n(i,r),Os(e,t,a)}catch(n){As(e,t,n)}}function Os(e,t,n){typeof n==`object`&&n&&typeof n.then==`function`?n.then(function(n){ks(e,t,n)},function(n){return As(e,t,n)}):ks(e,t,n)}function ks(e,t,n){t.status=`fulfilled`,t.value=n,js(t),e.state=n,t=e.pending,t!==null&&(n=t.next,n===t?e.pending=null:(n=n.next,t.next=n,Ds(e,n)))}function As(e,t,n){var r=e.pending;if(e.pending=null,r!==null){r=r.next;do t.status=`rejected`,t.reason=n,js(t),t=t.next;while(t!==r)}e.action=null}function js(e){e=e.listeners;for(var t=0;t<e.length;t++)(0,e[t])()}function Ms(e,t){return t}function Ns(e,t){if(F){var n=G.formState;if(n!==null){a:{var r=L;if(F){if(P){b:{for(var i=P,a=da;i.nodeType!==8;){if(!a){i=null;break b}if(i=lm(i.nextSibling),i===null){i=null;break b}}a=i.data,i=a===`F!`||a===`F`?i:null}if(i){P=lm(i.nextSibling),r=i.data===`F!`;break a}}pa(r)}r=!1}r&&(t=n[0])}}return n=ls(),n.memoizedState=n.baseState=t,r={pending:null,lanes:0,dispatch:null,lastRenderedReducer:Ms,lastRenderedState:t},n.queue=r,n=dc.bind(null,L,r),r.dispatch=n,r=ws(!1),a=pc.bind(null,L,!1,r.queue),r=ls(),i={state:t,dispatch:null,action:e,pending:null},r.queue=i,n=Es.bind(null,L,i,a,n),i.dispatch=n,r.memoizedState=e,[t,n,!1]}function Ps(e){return Fs(B(),R,e)}function Fs(e,t,n){if(t=gs(e,t,Ms)[0],e=hs(ms)[0],typeof t==`object`&&t&&typeof t.then==`function`)try{var r=ds(t)}catch(e){throw e===$a?to:e}else r=t;t=B();var i=t.queue,a=i.dispatch;return n!==t.memoizedState&&(L.flags|=2048,Rs(9,{destroy:void 0},Is.bind(null,i,n),null)),[r,a,e]}function Is(e,t){e.action=t}function Ls(e){var t=B(),n=R;if(n!==null)return Fs(t,n,e);B(),t=t.memoizedState,n=B();var r=n.queue.dispatch;return n.memoizedState=e,[t,r,!1]}function Rs(e,t,n,r){return e={tag:e,create:n,deps:r,inst:t,next:null},t=L.updateQueue,t===null&&(t=us(),L.updateQueue=t),n=t.lastEffect,n===null?t.lastEffect=e.next=e:(r=n.next,n.next=e,e.next=r,t.lastEffect=e),e}function zs(){return B().memoizedState}function Bs(e,t,n,r){var i=ls();L.flags|=e,i.memoizedState=Rs(1|t,{destroy:void 0},n,r===void 0?null:r)}function Vs(e,t,n,r){var i=B();r=r===void 0?null:r;var a=i.memoizedState.inst;R!==null&&r!==null&&ts(r,R.memoizedState.deps)?i.memoizedState=Rs(t,a,n,r):(L.flags|=e,i.memoizedState=Rs(1|t,a,n,r))}function Hs(e,t){Bs(8390656,8,e,t)}function Us(e,t){Vs(2048,8,e,t)}function Ws(e){L.flags|=4;var t=L.updateQueue;if(t===null)t=us(),L.updateQueue=t,t.events=[e];else{var n=t.events;n===null?t.events=[e]:n.push(e)}}function Gs(e){var t=B().memoizedState;return Ws({ref:t,nextImpl:e}),function(){if(W&2)throw Error(i(440));return t.impl.apply(void 0,arguments)}}function Ks(e,t){return Vs(4,2,e,t)}function qs(e,t){return Vs(4,4,e,t)}function Js(e,t){if(typeof t==`function`){e=e();var n=t(e);return function(){typeof n==`function`?n():t(null)}}if(t!=null)return e=e(),t.current=e,function(){t.current=null}}function Ys(e,t,n){n=n==null?null:n.concat([e]),Vs(4,4,Js.bind(null,t,e),n)}function Xs(){}function Zs(e,t){var n=B();t=t===void 0?null:t;var r=n.memoizedState;return t!==null&&ts(t,r[1])?r[0]:(n.memoizedState=[e,t],e)}function Qs(e,t){var n=B();t=t===void 0?null:t;var r=n.memoizedState;if(t!==null&&ts(t,r[1]))return r[0];if(r=e(),Xo){at(!0);try{e()}finally{at(!1)}}return n.memoizedState=[r,t],r}function $s(e,t,n){return n===void 0||Ko&1073741824&&!(q&261930)?e.memoizedState=t:(e.memoizedState=n,e=Md(),L.lanes|=e,sd|=e,n)}function ec(e,t,n,r){return qr(n,t)?n:Ao.current===null?!(Ko&106)||Ko&1073741824&&!(q&261930)?(Ic=!0,e.memoizedState=n):(e=Md(),L.lanes|=e,sd|=e,t):(e=$s(e,n,r),qr(e,t)||(Ic=!0),e)}function tc(e,t,n,r,i){var a=O.p;O.p=a!==0&&8>a?a:8;var o=D.T,s={};s.types=o===null?null:o.types,D.T=s,pc(e,!1,t,n);try{var c=i(),l=D.S;l!==null&&l(s,c),typeof c==`object`&&c&&typeof c.then==`function`?fc(e,t,qa(c,r),jd(e)):fc(e,t,r,jd(e))}catch(n){fc(e,t,{then:function(){},status:`rejected`,reason:n},jd())}finally{O.p=a,o!==null&&s.types!==null&&(o.types=s.types),D.T=o}}function nc(){}function rc(e,t,n,r){if(e.tag!==5)throw Error(i(476));var a=ic(e).queue;tc(e,a,t,we,n===null?nc:function(){return ac(e),n(r)})}function ic(e){var t=e.memoizedState;if(t!==null)return t;t={memoizedState:we,baseState:we,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:we},next:null};var n={};return t.next={memoizedState:n,baseState:n,baseQueue:null,queue:{pending:null,lanes:0,dispatch:null,lastRenderedReducer:ms,lastRenderedState:n},next:null},e.memoizedState=t,e=e.alternate,e!==null&&(e.memoizedState=t),t}function ac(e){var t=ic(e);t.next===null&&(t=e.alternate.memoizedState),fc(e,t.next.queue,{},jd())}function oc(){return Aa(sh)}function sc(){return B().memoizedState}function cc(){return B().memoizedState}function lc(e){for(var t=e.return;t!==null;){switch(t.tag){case 24:case 3:var n=jd();e=xo(n);var r=So(t,e,n);r!==null&&(Pd(r,t,n),Co(r,t,n)),t={cache:Ia()},e.payload=t;return}t=t.return}}function uc(e,t,n){var r=jd();n={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null},mc(e)?hc(t,n):(n=Mi(e,t,n,r),n!==null&&(Pd(n,e,r),gc(n,t,r)))}function dc(e,t,n){fc(e,t,n,jd())}function fc(e,t,n,r){var i={lane:r,revertLane:0,gesture:null,action:n,hasEagerState:!1,eagerState:null,next:null};if(mc(e))hc(t,i);else{var a=e.alternate;if(e.lanes===0&&(a===null||a.lanes===0)&&(a=t.lastRenderedReducer,a!==null))try{var o=t.lastRenderedState,s=a(o,n);if(i.hasEagerState=!0,i.eagerState=s,qr(s,o))return ji(e,t,i,0),G===null&&Ai(),!1}catch{}if(n=Mi(e,t,i,r),n!==null)return Pd(n,e,r),gc(n,t,r),!0}return!1}function pc(e,t,n,r){if(r={lane:2,revertLane:Pf(),gesture:null,action:r,hasEagerState:!1,eagerState:null,next:null},mc(e)){if(t)throw Error(i(479))}else t=Mi(e,n,r,2),t!==null&&Pd(t,e,2)}function mc(e){var t=e.alternate;return e===L||t!==null&&t===L}function hc(e,t){Yo=Jo=!0;var n=e.pending;n===null?t.next=t:(t.next=n.next,n.next=t),e.pending=t}function gc(e,t,n){if(n&4194048){var r=t.lanes;r&=e.pendingLanes,n|=r,t.lanes=n,Ct(e,n)}}var _c={readContext:Aa,use:fs,useCallback:z,useContext:z,useEffect:z,useImperativeHandle:z,useLayoutEffect:z,useInsertionEffect:z,useMemo:z,useReducer:z,useRef:z,useState:z,useDebugValue:z,useDeferredValue:z,useTransition:z,useSyncExternalStore:z,useId:z,useHostTransitionStatus:z,useFormState:z,useActionState:z,useOptimistic:z,useMemoCache:z,useCacheRefresh:z,useEffectEvent:z},vc={readContext:Aa,use:fs,useCallback:function(e,t){return ls().memoizedState=[e,t===void 0?null:t],e},useContext:Aa,useEffect:Hs,useImperativeHandle:function(e,t,n){n=n==null?null:n.concat([e]),Bs(4194308,4,Js.bind(null,t,e),n)},useLayoutEffect:function(e,t){return Bs(4194308,4,e,t)},useInsertionEffect:function(e,t){Bs(4,2,e,t)},useMemo:function(e,t){var n=ls();t=t===void 0?null:t;var r=e();if(Xo){at(!0);try{e()}finally{at(!1)}}return n.memoizedState=[r,t],r},useReducer:function(e,t,n){var r=ls();if(n!==void 0){var i=n(t);if(Xo){at(!0);try{n(t)}finally{at(!1)}}}else i=t;return r.memoizedState=r.baseState=i,e={pending:null,lanes:0,dispatch:null,lastRenderedReducer:e,lastRenderedState:i},r.queue=e,e=e.dispatch=uc.bind(null,L,e),[r.memoizedState,e]},useRef:function(e){var t=ls();return e={current:e},t.memoizedState=e},useState:function(e){e=ws(e);var t=e.queue,n=dc.bind(null,L,t);return t.dispatch=n,[e.memoizedState,n]},useDebugValue:Xs,useDeferredValue:function(e,t){return $s(ls(),e,t)},useTransition:function(){var e=ws(!1);return e=tc.bind(null,L,e.queue,!0,!1),ls().memoizedState=e,[!1,e]},useSyncExternalStore:function(e,t,n){var r=L,a=ls();if(F){if(n===void 0)throw Error(i(407));n=n()}else{if(n=t(),G===null)throw Error(i(349));q&127||ys(r,t,n)}a.memoizedState=n;var o={value:n,getSnapshot:t};return a.queue=o,Hs(xs.bind(null,r,o,e),[e]),r.flags|=2048,Rs(9,{destroy:void 0},bs.bind(null,r,o,n,t),null),n},useId:function(){var e=ls(),t=G.identifierPrefix;if(F){var n=ra,r=na;n=(r&~(1<<32-ot(r)-1)).toString(32)+n,t=`_`+t+`R_`+n,n=Zo++,0<n&&(t+=`H`+n.toString(32)),t+=`_`}else n=es++,t=`_`+t+`r_`+n.toString(32)+`_`;return e.memoizedState=t},useHostTransitionStatus:oc,useFormState:Ns,useActionState:Ns,useOptimistic:function(e){var t=ls();t.memoizedState=t.baseState=e;var n={pending:null,lanes:0,dispatch:null,lastRenderedReducer:null,lastRenderedState:null};return t.queue=n,t=pc.bind(null,L,!0,n),n.dispatch=t,[e,t]},useMemoCache:ps,useCacheRefresh:function(){return ls().memoizedState=lc.bind(null,L)},useEffectEvent:function(e){var t=ls(),n={impl:e};return t.memoizedState=n,function(){if(W&2)throw Error(i(440));return n.impl.apply(void 0,arguments)}}},yc={readContext:Aa,use:fs,useCallback:Zs,useContext:Aa,useEffect:Us,useImperativeHandle:Ys,useInsertionEffect:Ks,useLayoutEffect:qs,useMemo:Qs,useReducer:hs,useRef:zs,useState:function(){return hs(ms)},useDebugValue:Xs,useDeferredValue:function(e,t){return ec(B(),R.memoizedState,e,t)},useTransition:function(){var e=hs(ms)[0],t=B().memoizedState;return[typeof e==`boolean`?e:ds(e),t]},useSyncExternalStore:vs,useId:sc,useHostTransitionStatus:oc,useFormState:Ps,useActionState:Ps,useOptimistic:function(e,t){return Ts(B(),R,e,t)},useMemoCache:ps,useCacheRefresh:cc,useEffectEvent:Gs},bc={readContext:Aa,use:fs,useCallback:Zs,useContext:Aa,useEffect:Us,useImperativeHandle:Ys,useInsertionEffect:Ks,useLayoutEffect:qs,useMemo:Qs,useReducer:_s,useRef:zs,useState:function(){return _s(ms)},useDebugValue:Xs,useDeferredValue:function(e,t){var n=B();return R===null?$s(n,e,t):ec(n,R.memoizedState,e,t)},useTransition:function(){var e=_s(ms)[0],t=B().memoizedState;return[typeof e==`boolean`?e:ds(e),t]},useSyncExternalStore:vs,useId:sc,useHostTransitionStatus:oc,useFormState:Ls,useActionState:Ls,useOptimistic:function(e,t){var n=B();return R===null?(n.baseState=e,[e,n.queue.dispatch]):Ts(n,R,e,t)},useMemoCache:ps,useCacheRefresh:cc,useEffectEvent:Gs};function xc(e,t,n,r){t=e.memoizedState,n=n(r,t),n=n==null?t:w({},t,n),e.memoizedState=n,e.lanes===0&&(e.updateQueue.baseState=n)}var Sc={enqueueSetState:function(e,t,n){e=e._reactInternals;var r=jd(),i=xo(r);i.payload=t,n!=null&&(i.callback=n),t=So(e,i,r),t!==null&&(Pd(t,e,r),Co(t,e,r))},enqueueReplaceState:function(e,t,n){e=e._reactInternals;var r=jd(),i=xo(r);i.tag=1,i.payload=t,n!=null&&(i.callback=n),t=So(e,i,r),t!==null&&(Pd(t,e,r),Co(t,e,r))},enqueueForceUpdate:function(e,t){e=e._reactInternals;var n=jd(),r=xo(n);r.tag=2,t!=null&&(r.callback=t),t=So(e,r,n),t!==null&&(Pd(t,e,n),Co(t,e,n))}};function Cc(e,t,n,r,i,a,o){return e=e.stateNode,typeof e.shouldComponentUpdate==`function`?e.shouldComponentUpdate(r,a,o):t.prototype&&t.prototype.isPureReactComponent?!Jr(n,r)||!Jr(i,a):!0}function wc(e,t,n,r){e=t.state,typeof t.componentWillReceiveProps==`function`&&t.componentWillReceiveProps(n,r),typeof t.UNSAFE_componentWillReceiveProps==`function`&&t.UNSAFE_componentWillReceiveProps(n,r),t.state!==e&&Sc.enqueueReplaceState(t,t.state,null)}function Tc(e,t){var n=t;if(`ref`in t)for(var r in n={},t)r!==`ref`&&(n[r]=t[r]);if(e=e.defaultProps)for(var i in n===t&&(n=w({},n)),e)n[i]===void 0&&(n[i]=e[i]);return n}function Ec(e){Ei(e)}function Dc(e){console.error(e)}function Oc(e){Ei(e)}function kc(e,t){try{var n=e.onUncaughtError;n(t.value,{componentStack:t.stack})}catch(e){setTimeout(function(){throw e})}}function Ac(e,t,n){try{var r=e.onCaughtError;r(n.value,{componentStack:n.stack,errorBoundary:t.tag===1?t.stateNode:null})}catch(e){setTimeout(function(){throw e})}}function jc(e,t,n){return n=xo(n),n.tag=3,n.payload={element:null},n.callback=function(){kc(e,t)},n}function Mc(e){return e=xo(e),e.tag=3,e}function Nc(e,t,n,r){var i=n.type.getDerivedStateFromError;if(typeof i==`function`){var a=r.value;e.payload=function(){return i(a)},e.callback=function(){Ac(t,n,r)}}var o=n.stateNode;o!==null&&typeof o.componentDidCatch==`function`&&(e.callback=function(){Ac(t,n,r),typeof i!=`function`&&(yd===null?yd=new Set([this]):yd.add(this));var e=r.stack;this.componentDidCatch(r.value,{componentStack:e===null?``:e})})}function Pc(e,t,n,r,a){if(n.flags|=32768,typeof r==`object`&&r&&typeof r.then==`function`){if(t=n.alternate,t!==null&&Da(t,n,a,!0),n=Fo.current,n!==null){switch(n.tag){case 31:case 13:case 19:return Io===null?Kd():n.alternate===null&&Y===0&&(Y=3),n.flags&=-257,n.flags|=65536,n.lanes=a,r===no?n.flags|=16384:(t=n.updateQueue,t===null?n.updateQueue=new Set([r]):t.add(r),mf(e,r,a)),!1;case 22:return n.flags|=65536,r===no?n.flags|=16384:(t=n.updateQueue,t===null?(t={transitions:null,markerInstances:null,retryQueue:new Set([r])},n.updateQueue=t):(n=t.retryQueue,n===null?t.retryQueue=new Set([r]):n.add(r)),mf(e,r,a)),!1}throw Error(i(435,n.tag))}return mf(e,r,a),Kd(),!1}if(F)return t=Fo.current,t===null?(r!==fa&&(t=Error(i(423),{cause:r}),ya(Ji(t,n))),e=e.current.alternate,e.flags|=65536,a&=-a,e.lanes|=a,r=Ji(r,n),a=jc(e.stateNode,r,a),wo(e,a),Y!==4&&(Y=2)):(!(t.flags&65536)&&(t.flags|=256),t.flags|=65536,t.lanes=a,r!==fa&&(e=Error(i(422),{cause:r}),ya(Ji(e,n)))),!1;var o=Error(i(520),{cause:r});if(o=Ji(o,n),fd===null?fd=[o]:fd.push(o),Y!==4&&(Y=2),t===null)return!0;r=Ji(r,n),n=t;do{switch(n.tag){case 3:return n.flags|=65536,e=a&-a,n.lanes|=e,e=jc(n.stateNode,r,e),wo(n,e),!1;case 1:if(t=n.type,o=n.stateNode,!(n.flags&128)&&(typeof t.getDerivedStateFromError==`function`||o!==null&&typeof o.componentDidCatch==`function`&&(yd===null||!yd.has(o))))return n.flags|=65536,a&=-a,n.lanes|=a,a=Mc(a),Nc(a,e,n,r),wo(n,a),!1;break;case 22:if(n.memoizedState!==null)return n.flags|=65536,!1}n=n.return}while(n!==null);return!1}var Fc=Error(i(461)),Ic=!1;function Lc(e,t,n,r){t.child=e===null?_o(t,null,n,r):go(t,e.child,n,r)}function Rc(e,t,n,r,i){n=n.render;var a=t.ref;if(`ref`in r){var o={};for(var s in r)s!==`ref`&&(o[s]=r[s])}else o=r;return ka(t),r=ns(e,t,n,o,a,i),s=os(),e!==null&&!Ic?(ss(e,t,i),fl(e,t,i)):(F&&s&&oa(t),t.flags|=1,Lc(e,t,r,i),t.child)}function zc(e,t,n,r,i){if(e===null){var a=n.type;return typeof a==`function`&&!zi(a)&&a.defaultProps===void 0&&n.compare===null?(t.tag=15,t.type=a,Bc(e,t,a,r,i)):(e=Hi(n.type,null,r,t,t.mode,i),e.ref=t.ref,e.return=t,t.child=e)}if(a=e.child,!pl(e,i)){var o=a.memoizedProps;if(n=n.compare,n=n===null?Jr:n,n(o,r)&&e.ref===t.ref)return fl(e,t,i)}return t.flags|=1,e=Bi(a,r),e.ref=t.ref,e.return=t,t.child=e}function Bc(e,t,n,r,i){if(e!==null){var a=e.memoizedProps;if(Jr(a,r)&&e.ref===t.ref){if(Ic=!1,t.pendingProps=r=a,pl(e,i))e.flags&131072&&(Ic=!0);else return t.lanes=e.lanes,fl(e,t,i)}}return Jc(e,t,n,r,i)}function Vc(e,t,n,r){var i=r.children,a=e===null?null:e.memoizedState;if(e===null&&t.stateNode===null&&(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),r.mode===`hidden`){if(t.flags&128){if(a=a===null?n:a.baseLanes|n,e!==null){for(r=t.child=e.child,i=0;r!==null;)i=i|r.lanes|r.childLanes,r=r.sibling;r=i&~a}else r=0,t.child=null;return Uc(e,t,a,n,r)}if(n&536870912)t.memoizedState={baseLanes:0,cachePool:null},e!==null&&Za(t,a===null?null:a.cachePool),a===null?No():Mo(t,a),zo(t);else return r=t.lanes=536870912,Uc(e,t,a===null?n:a.baseLanes|n,n,r)}else a===null?(e!==null&&Za(t,null),No(),Bo()):(Za(t,a.cachePool),Mo(t,a),Bo(),t.memoizedState=null);return Lc(e,t,i,n),t.child}function Hc(e,t){return e!==null&&e.tag===22||t.stateNode!==null||(t.stateNode={_visibility:1,_pendingMarkers:null,_retryCache:null,_transitions:null}),t.sibling}function Uc(e,t,n,r,i){var a=Xa();return a=a===null?null:{parent:I._currentValue,pool:a},t.memoizedState={baseLanes:n,cachePool:a},e!==null&&Za(t,null),No(),zo(t),e!==null&&Da(e,t,r,!0),t.childLanes=i,null}function Wc(e,t){return t=rl({mode:t.mode,children:t.children},e.mode),t.ref=e.ref,e.child=t,t.return=e,t}function Gc(e,t,n){return go(t,e.child,null,n),e=Wc(t,t.pendingProps),e.flags|=2,Vo(t),t.memoizedState=null,e}function Kc(e,t,n){var r=t.pendingProps,a=!!(t.flags&128);if(t.flags&=-129,e===null){if(F){if(r.mode===`hidden`)return e=Wc(t,r),t.lanes=536870912,e.memoizedState={baseLanes:0,cachePool:null},Hc(null,e);if(Ro(t),(e=P)?(e=am(e,da),e=e!==null&&e.data===`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:ta===null?null:{id:na,overflow:ra},retryLane:536870912,hydrationErrors:null},n=Gi(e),n.return=t,t.child=n,la=t,P=null)):e=null,e===null)throw pa(t);return t.lanes=536870912,null}return Wc(t,r)}var o=e.memoizedState;if(o!==null){var s=o.dehydrated;if(Ro(t),a){if(t.flags&256)t.flags&=-257,t=Gc(e,t,n);else if(t.memoizedState!==null)t.child=e.child,t.flags|=128,t=null;else throw Error(i(558))}else if(Ic||Da(e,t,n,!1),a=(n&e.childLanes)!==0,Ic||a){if(Ao.current===null){if(r=G,r!==null&&(s=wt(r,n),s!==0&&s!==o.retryLane))throw o.retryLane=s,Ni(e,s),Pd(r,e,s),Fc;Kd()}t=Gc(e,t,n)}else e=o.treeContext,P=lm(s.nextSibling),la=t,F=!0,ua=null,da=!1,e!==null&&ca(t,e),t=Wc(t,r),t.flags|=134221824;return t}return e=Bi(e.child,{mode:r.mode,children:r.children}),e.ref=t.ref,t.child=e,e.return=t,e}function qc(e,t){var n=t.ref;if(n===null)e!==null&&e.ref!==null&&(t.flags|=4194816);else{if(typeof n!=`function`&&typeof n!=`object`)throw Error(i(284));(e===null||e.ref!==n)&&(t.flags|=4194816)}}function Jc(e,t,n,r,i){return ka(t),n=ns(e,t,n,r,void 0,i),r=os(),e!==null&&!Ic?(ss(e,t,i),fl(e,t,i)):(F&&r&&oa(t),t.flags|=1,Lc(e,t,n,i),t.child)}function Yc(e,t,n,r,i,a){return ka(t),t.updateQueue=null,n=is(t,r,n,i),rs(e),r=os(),e!==null&&!Ic?(ss(e,t,a),fl(e,t,a)):(F&&r&&oa(t),t.flags|=1,Lc(e,t,n,a),t.child)}function Xc(e,t,n,r,i){if(ka(t),t.stateNode===null){var a=Ii,o=n.contextType;typeof o==`object`&&o&&(a=Aa(o)),a=new n(r,a),t.memoizedState=a.state!==null&&a.state!==void 0?a.state:null,a.updater=Sc,t.stateNode=a,a._reactInternals=t,a=t.stateNode,a.props=r,a.state=t.memoizedState,a.refs={},yo(t),o=n.contextType,a.context=typeof o==`object`&&o?Aa(o):Ii,a.state=t.memoizedState,o=n.getDerivedStateFromProps,typeof o==`function`&&(xc(t,n,o,r),a.state=t.memoizedState),typeof n.getDerivedStateFromProps==`function`||typeof a.getSnapshotBeforeUpdate==`function`||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(o=a.state,typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount(),o!==a.state&&Sc.enqueueReplaceState(a,a.state,null),Do(t,r,a,i),Eo(),a.state=t.memoizedState),typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!0}else if(e===null){a=t.stateNode;var s=t.memoizedProps,c=Tc(n,s);a.props=c;var l=a.context,u=n.contextType;o=Ii,typeof u==`object`&&u&&(o=Aa(u));var d=n.getDerivedStateFromProps;u=typeof d==`function`||typeof a.getSnapshotBeforeUpdate==`function`,s=t.pendingProps!==s,u||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(s||l!==o)&&wc(t,a,r,o),vo=!1;var f=t.memoizedState;a.state=f,Do(t,r,a,i),Eo(),l=t.memoizedState,s||f!==l||vo?(typeof d==`function`&&(xc(t,n,d,r),l=t.memoizedState),(c=vo||Cc(t,n,c,r,f,l,o))?(u||typeof a.UNSAFE_componentWillMount!=`function`&&typeof a.componentWillMount!=`function`||(typeof a.componentWillMount==`function`&&a.componentWillMount(),typeof a.UNSAFE_componentWillMount==`function`&&a.UNSAFE_componentWillMount()),typeof a.componentDidMount==`function`&&(t.flags|=4194308)):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),t.memoizedProps=r,t.memoizedState=l),a.props=r,a.state=l,a.context=o,r=c):(typeof a.componentDidMount==`function`&&(t.flags|=4194308),r=!1)}else{a=t.stateNode,bo(e,t),o=t.memoizedProps,u=Tc(n,o),a.props=u,d=t.pendingProps,f=a.context,l=n.contextType,c=Ii,typeof l==`object`&&l&&(c=Aa(l)),s=n.getDerivedStateFromProps,(l=typeof s==`function`||typeof a.getSnapshotBeforeUpdate==`function`)||typeof a.UNSAFE_componentWillReceiveProps!=`function`&&typeof a.componentWillReceiveProps!=`function`||(o!==d||f!==c)&&wc(t,a,r,c),vo=!1,f=t.memoizedState,a.state=f,Do(t,r,a,i),Eo();var p=t.memoizedState;o!==d||f!==p||vo||e!==null&&e.dependencies!==null&&Oa(e.dependencies)?(typeof s==`function`&&(xc(t,n,s,r),p=t.memoizedState),(u=vo||Cc(t,n,u,r,f,p,c)||e!==null&&e.dependencies!==null&&Oa(e.dependencies))?(l||typeof a.UNSAFE_componentWillUpdate!=`function`&&typeof a.componentWillUpdate!=`function`||(typeof a.componentWillUpdate==`function`&&a.componentWillUpdate(r,p,c),typeof a.UNSAFE_componentWillUpdate==`function`&&a.UNSAFE_componentWillUpdate(r,p,c)),typeof a.componentDidUpdate==`function`&&(t.flags|=4),typeof a.getSnapshotBeforeUpdate==`function`&&(t.flags|=1024)):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),t.memoizedProps=r,t.memoizedState=p),a.props=r,a.state=p,a.context=c,r=u):(typeof a.componentDidUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=4),typeof a.getSnapshotBeforeUpdate!=`function`||o===e.memoizedProps&&f===e.memoizedState||(t.flags|=1024),r=!1)}return a=r,qc(e,t),r=!!(t.flags&128),a||r?(a=t.stateNode,n=r&&typeof n.getDerivedStateFromError!=`function`?null:a.render(),t.flags|=1,e!==null&&r?(t.child=go(t,e.child,null,i),t.child=go(t,null,n,i)):Lc(e,t,n,i),t.memoizedState=a.state,e=t.child):e=fl(e,t,i),e}function Zc(e,t,n,r){return _a(),t.flags|=256,Lc(e,t,n,r),t.child}var Qc={dehydrated:null,treeContext:null,retryLane:0,hydrationErrors:null};function $c(e){return{baseLanes:e,cachePool:Qa()}}function el(e,t,n){return e=e===null?0:e.childLanes&~n,t&&(e|=ud),e}function tl(e,t,n){var r=t.pendingProps,i=!1,a=!!(t.flags&128),o;if((o=a)||(o=e!==null&&e.memoizedState===null?!1:!!(Ho.current&2)),o&&(i=!0,t.flags&=-129),o=!!(t.flags&32),t.flags&=-33,e===null){if(F){if(i?Lo(t):Bo(),(e=P)?(e=am(e,da),e=e!==null&&e.data!==`&`?e:null,e!==null&&(t.memoizedState={dehydrated:e,treeContext:ta===null?null:{id:na,overflow:ra},retryLane:536870912,hydrationErrors:null},n=Gi(e),n.return=t,t.child=n,la=t,P=null)):e=null,e===null)throw pa(t);return t.lanes=sm(e)?32:536870912,null}return a=r.children,r=r.fallback,i?(Bo(),i=t.mode,a=rl({mode:`hidden`,children:a},i),r=Ui(r,i,n,null),a.return=t,r.return=t,a.sibling=r,t.child=a,r=t.child,r.memoizedState=$c(n),r.childLanes=el(e,o,n),t.memoizedState=Qc,Hc(null,r)):(Lo(t),nl(t,a))}var s=e.memoizedState;if(s!==null){var c=s.dehydrated;if(c!==null)return al(e,t,a,o,r,c,s,n)}return i?(Bo(),i=r.fallback,a=t.mode,s=e.child,c=s.sibling,r=Bi(s,{mode:`hidden`,children:r.children}),r.subtreeFlags=s.subtreeFlags&1206910976,c===null?(i=Ui(i,a,n,null),i.flags|=2):i=Bi(c,i),i.return=t,r.return=t,r.sibling=i,t.child=r,Hc(null,r),r=t.child,i=e.child.memoizedState,i===null?i=$c(n):(a=i.cachePool,a===null?a=Qa():(s=I._currentValue,a=a.parent===s?a:{parent:s,pool:s}),i={baseLanes:i.baseLanes|n,cachePool:a}),r.memoizedState=i,r.childLanes=el(e,o,n),t.memoizedState=Qc,Hc(e.child,r)):(Lo(t),n=e.child,e=n.sibling,n=Bi(n,{mode:`visible`,children:r.children}),n.return=t,n.sibling=null,e!==null&&(o=t.deletions,o===null?(t.deletions=[e],t.flags|=16):o.push(e)),t.child=n,t.memoizedState=null,n)}function nl(e,t){return t=rl({mode:`visible`,children:t},e.mode),t.return=e,e.child=t}function rl(e,t){return e=Ri(22,e,null,t),e.lanes=0,e}function il(e,t,n){return go(t,e.child,null,n),e=nl(t,t.pendingProps.children),e.flags|=2,t.memoizedState=null,e}function al(e,t,n,r,a,o,s,c){if(n)return t.flags&256?(Lo(t),t.flags&=-257,il(e,t,c)):t.memoizedState===null?(Bo(),o=a.fallback,s=t.mode,a=rl({mode:`visible`,children:a.children},s),o=Ui(o,s,c,null),o.flags|=2,a.return=t,o.return=t,a.sibling=o,t.child=a,go(t,e.child,null,c),a=t.child,a.memoizedState=$c(c),a.childLanes=el(e,r,c),t.memoizedState=Qc,Hc(null,a)):(Bo(),t.child=e.child,t.flags|=128,null);if(Lo(t),sm(o)){if(r=o.nextSibling&&o.nextSibling.dataset,r)var l=r.dgst;return r=l,r!==``&&(a=Error(i(419)),a.stack=``,a.digest=r,ya({value:a,source:null,stack:null})),il(e,t,c)}if(Ic||Da(e,t,c,!1),r=(c&e.childLanes)!==0,Ic||r){if(Ao.current!==null)return il(e,t,c);if(r=G,r!==null&&(a=wt(r,c),a!==0&&a!==s.retryLane))throw s.retryLane=a,Ni(e,a),Pd(r,e,a),Fc;return om(o)||Kd(),il(e,t,c)}return om(o)?(t.flags|=192,t.child=e.child,null):(e=s.treeContext,P=lm(o.nextSibling),la=t,F=!0,ua=null,da=!1,e!==null&&ca(t,e),t=nl(t,a.children),t.flags|=134221824,t)}function ol(e,t,n){e.lanes|=t;var r=e.alternate;r!==null&&(r.lanes|=t),Ta(e.return,t,n)}function sl(e){for(var t=null;e!==null;){var n=e.alternate;n!==null&&Go(n)===null&&(t=e),e=e.sibling}return t}function cl(e,t,n,r,i,a){var o=e.memoizedState;o===null?e.memoizedState={isBackwards:t,rendering:null,renderingStartTime:0,last:r,tail:n,tailMode:i,treeForkCount:a}:(o.isBackwards=t,o.rendering=null,o.renderingStartTime=0,o.last=r,o.tail=n,o.tailMode=i,o.treeForkCount=a)}function ll(e){var t=e.child;for(e.child=null;t!==null;){var n=t.sibling;t.sibling=e.child,e.child=t,t=n}}function ul(e,t,n){var r=t.pendingProps,i=r.revealOrder,a=r.tail;r=r.children;var o=Ho.current;if(t.flags&128)return Uo(t,o),null;var s=!!(o&2);if(s?(o=o&1|2,t.flags|=128):o&=1,Uo(t,o),i===`backwards`&&e!==null?(ll(e),Lc(e,t,r,n),ll(e)):Lc(e,t,r,n),r=F?Qi:0,!s&&e!==null&&e.flags&128)a:for(e=t.child;e!==null;){if(e.tag===13)e.memoizedState!==null&&ol(e,n,t);else if(e.tag===19)ol(e,n,t);else if(e.child!==null){e.child.return=e,e=e.child;continue}if(e===t)break a;for(;e.sibling===null;){if(e.return===null||e.return===t)break a;e=e.return}e.sibling.return=e.return,e=e.sibling}switch(i){case`backwards`:n=sl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null,ll(t)),cl(t,!0,i,null,a,r);break;case`unstable_legacy-backwards`:for(n=null,i=t.child,t.child=null;i!==null;){if(e=i.alternate,e!==null&&Go(e)===null){t.child=i;break}e=i.sibling,i.sibling=n,n=i,i=e}cl(t,!0,n,null,a,r);break;case`together`:cl(t,!1,null,null,void 0,r);break;case`independent`:t.memoizedState=null;break;default:n=sl(t.child),n===null?(i=t.child,t.child=null):(i=n.sibling,n.sibling=null),cl(t,!1,i,n,a,r)}return t.child}function dl(e,t,n){var r=t.pendingProps;return Ca(t,t.type,r.value),Lc(e,t,r.children,n),t.child}function fl(e,t,n){if(e!==null&&(t.dependencies=e.dependencies),sd|=t.lanes,(n&t.childLanes)===0){if(e!==null){if(Da(e,t,n,!1),(n&t.childLanes)===0)return null}else return null}if(e!==null&&t.child!==e.child)throw Error(i(153));if(t.child!==null){for(e=t.child,n=Bi(e,e.pendingProps),t.child=n,n.return=t;e.sibling!==null;)e=e.sibling,n=n.sibling=Bi(e,e.pendingProps),n.return=t;n.sibling=null}return t.child}function pl(e,t){return(e.lanes&t)!==0||(e=e.dependencies,!!(e!==null&&Oa(e)))}function ml(e,t,n){switch(t.tag){case 3:Ne(t,t.stateNode.containerInfo),Ca(t,I,e.memoizedState.cache),_a();break;case 27:case 5:Fe(t);break;case 4:Ne(t,t.stateNode.containerInfo);break;case 10:Ca(t,t.type,t.memoizedProps.value);break;case 31:if(t.memoizedState!==null)return t.flags|=128,Ro(t),null;break;case 13:var r=t.memoizedState;if(r!==null){if(r.dehydrated!==null)return Lo(t),t.flags|=128,null;r=Da(e,t,n,!1);var i=t.child.childLanes;return r||(n&i)!==0?tl(e,t,n):(Lo(t),e=fl(e,t,n),e===null?null:e.sibling)}Lo(t);break;case 19:if(t.flags&128)return ul(e,t,n);if(i=!!(e.flags&128),r=(n&t.childLanes)!==0,r||=(Da(e,t,n,!1),(n&t.childLanes)!==0),i){if(r)return ul(e,t,n);t.flags|=128}if(i=t.memoizedState,i!==null&&(i.rendering=null,i.tail=null,i.lastEffect=null),Uo(t,Ho.current),r)break;return null;case 22:return t.lanes=0,Vc(e,t,n,t.pendingProps);case 24:Ca(t,I,e.memoizedState.cache)}return fl(e,t,n)}function hl(e,t,n){if(e!==null){if(e.memoizedProps!==t.pendingProps)Ic=!0;else{if(!pl(e,n)&&!(t.flags&128))return Ic=!1,ml(e,t,n);Ic=!!(e.flags&131072)}}else Ic=!1,F&&t.flags&1048576&&aa(t,Qi,t.index);switch(t.lanes=0,t.tag){case 16:a:{var r=t.pendingProps;if(e=ao(t.elementType),t.type=e,typeof e==`function`)zi(e)?(r=Tc(e,r),t.tag=1,t=Xc(null,t,e,r,n)):(t.tag=0,t=Jc(null,t,e,r,n));else{if(e!=null){var a=e.$$typeof;if(a===de){t.tag=11,t=Rc(null,t,e,r,n);break a}if(a===me){t.tag=14,t=zc(null,t,e,r,n);break a}if(a===ue){t.tag=10,t.type=e,t=dl(null,t,n);break a}}throw t=Se(e)||e,Error(i(306,t,``))}}return t;case 0:return Jc(e,t,t.type,t.pendingProps,n);case 1:return r=t.type,a=Tc(r,t.pendingProps),Xc(e,t,r,a,n);case 3:a:{if(Ne(t,t.stateNode.containerInfo),e===null)throw Error(i(387));r=t.pendingProps;var o=t.memoizedState;a=o.element,bo(e,t),Do(t,r,null,n);var s=t.memoizedState;if(r=s.cache,Ca(t,I,r),r!==o.cache&&Ea(t,[I],n,!0),Eo(),r=s.element,o.isDehydrated){if(o={element:r,isDehydrated:!1,cache:s.cache},t.updateQueue.baseState=o,t.memoizedState=o,t.flags&256){t=Zc(e,t,r,n);break a}if(r!==a){a=Ji(Error(i(424)),t),ya(a),t=Zc(e,t,r,n);break a}switch(e=t.stateNode.containerInfo,e.nodeType){case 9:e=e.body;break;default:e=e.nodeName===`HTML`?e.ownerDocument.body:e}for(P=lm(e.firstChild),la=t,F=!0,ua=null,da=!0,n=_o(t,null,r,n),t.child=n;n;)n.flags=n.flags&-3|134221824,n=n.sibling}else{if(_a(),r===a){t=fl(e,t,n);break a}Lc(e,t,r,n)}t=t.child}return t;case 26:return qc(e,t),e===null?(n=Nm(t.type,null,t.pendingProps,null))?t.memoizedState=n:F||(t.stateNode=fp(t.type,t.pendingProps,je.current,t)):t.memoizedState=Nm(t.type,e.memoizedProps,t.pendingProps,e.memoizedState),null;case 27:return Fe(t),e===null&&F&&(r=t.stateNode=hm(t.type,t.pendingProps,je.current),la=t,da=!0,a=P,Sp(t.type)?(um=a,P=lm(r.firstChild)):P=a),Lc(e,t,t.pendingProps.children,n),qc(e,t),e===null&&(t.flags|=4194304),t.child;case 5:return e===null&&F&&((a=r=P)&&(r=rm(r,t.type,t.pendingProps,da),r===null?a=!1:(t.stateNode=r,la=t,P=lm(r.firstChild),da=!1,a=!0)),a||pa(t)),Fe(t),a=t.type,o=t.pendingProps,s=e===null?null:e.memoizedProps,r=o.children,pp(a,o)?r=null:s!==null&&pp(a,s)&&(t.flags|=32),t.memoizedState!==null&&(a=ns(e,t,as,null,null,n),sh._currentValue=a),qc(e,t),Lc(e,t,r,n),t.child;case 6:return e===null&&F&&((e=n=P)&&(n=im(n,t.pendingProps,da),n===null?e=!1:(t.stateNode=n,la=t,P=null,e=!0)),e||pa(t)),null;case 13:return tl(e,t,n);case 4:return Ne(t,t.stateNode.containerInfo),r=t.pendingProps,e===null?t.child=go(t,null,r,n):Lc(e,t,r,n),t.child;case 11:return Rc(e,t,t.type,t.pendingProps,n);case 7:return r=t.pendingProps,qc(e,t),Lc(e,t,r,n),t.child;case 8:return Lc(e,t,t.pendingProps.children,n),t.child;case 12:return Lc(e,t,t.pendingProps.children,n),t.child;case 10:return dl(e,t,n);case 9:return a=t.type._context,r=t.pendingProps.children,ka(t),a=Aa(a),r=r(a),t.flags|=1,Lc(e,t,r,n),t.child;case 14:return zc(e,t,t.type,t.pendingProps,n);case 15:return Bc(e,t,t.type,t.pendingProps,n);case 19:return ul(e,t,n);case 31:return Kc(e,t,n);case 22:return Vc(e,t,n,t.pendingProps);case 24:return ka(t),r=Aa(I),e===null?(a=Xa(),a===null&&(a=G,o=Ia(),a.pooledCache=o,o.refCount++,o!==null&&(a.pooledCacheLanes|=n),a=o),t.memoizedState={parent:r,cache:a},yo(t),Ca(t,I,a)):((e.lanes&n)!==0&&(bo(e,t),Do(t,null,null,n),Eo()),a=e.memoizedState,o=t.memoizedState,a.parent===r?(r=o.cache,Ca(t,I,r),r!==a.cache&&Ea(t,[I],n,!0)):(a={parent:r,cache:r},t.memoizedState=a,t.lanes===0&&(t.memoizedState=t.updateQueue.baseState=a),Ca(t,I,r))),Lc(e,t,t.pendingProps.children,n),t.child;case 30:return t.stateNode===null&&(t.stateNode={autoName:null,paired:null,clones:null,ref:null}),r=t.pendingProps,r.name!=null&&r.name!==`auto`?t.flags|=e===null?18882560:18874368:F&&oa(t),e!==null&&e.memoizedProps.name!==r.name?t.flags|=4194816:qc(e,t),Lc(e,t,r.children,n),t.child;case 29:throw t.pendingProps}throw Error(i(156,t.tag))}function gl(e){e.flags|=4}function _l(e,t,n,r,i){var a;if((a=!!(e.mode&32))&&(a=n===null?Jm(t,r):Jm(t,r)&&(r.src!==n.src||r.srcSet!==n.srcSet)),a){if(e.flags|=16777216,(i&335544128)===i){if(e.stateNode.complete)e.flags|=8192;else if(Ud())e.flags|=8192;else throw oo=no,eo}}else e.flags&=-16777217}function vl(e,t){if(t.type!==`stylesheet`||t.state.loading&4)e.flags&=-16777217;else if(e.flags|=16777216,!Ym(t)){if(Ud())e.flags|=8192;else throw oo=no,eo}}function yl(e,t){t!==null&&(e.flags|=4),e.flags&16384&&(t=e.tag===22?536870912:vt(),e.lanes|=t,dd|=t)}function bl(e,t){if(!F)switch(e.tailMode){case`visible`:break;case`collapsed`:for(var n=e.tail,r=null;n!==null;)n.alternate!==null&&(r=n),n=n.sibling;r===null?t||e.tail===null?e.tail=null:e.tail.sibling=null:r.sibling=null;break;default:for(t=e.tail,n=null;t!==null;)t.alternate!==null&&(n=t),t=t.sibling;n===null?e.tail=null:n.sibling=null}}function V(e){var t=e.alternate!==null&&e.alternate.child===e.child,n=0,r=0;if(t)for(var i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags&1206910976,r|=i.flags&1206910976,i.return=e,i=i.sibling;else for(i=e.child;i!==null;)n|=i.lanes|i.childLanes,r|=i.subtreeFlags,r|=i.flags,i.return=e,i=i.sibling;return e.subtreeFlags|=r,e.childLanes=n,t}function xl(e,t,n){var r=t.pendingProps;switch(sa(t),t.tag){case 16:case 15:case 0:case 11:case 7:case 8:case 12:case 9:case 14:return V(t),null;case 1:return V(t),null;case 3:return n=t.stateNode,r=null,e!==null&&(r=e.memoizedState.cache),t.memoizedState.cache!==r&&(t.flags|=2048),wa(I),Pe(),n.pendingContext&&(n.context=n.pendingContext,n.pendingContext=null),(e===null||e.child===null)&&(ga(t)?gl(t):e===null||e.memoizedState.isDehydrated&&!(t.flags&256)||(t.flags|=1024,va())),V(t),null;case 26:var a=t.type,o=t.memoizedState;return e===null?(gl(t),o===null?(V(t),_l(t,a,null,r,n)):(V(t),vl(t,o))):o?o===e.memoizedState?(V(t),t.flags&=-16777217):(gl(t),V(t),vl(t,o)):(e=e.memoizedProps,e!==r&&gl(t),V(t),_l(t,a,e,r,n)),null;case 27:if(Ie(t),n=je.current,a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return V(t),t.subtreeFlags&=-33554433,null}e=ke.current,ga(t)?ma(t,e):(e=hm(a,r,n),t.stateNode=e,gl(t))}return V(t),t.subtreeFlags&=-33554433,null;case 5:if(Ie(t),a=t.type,e!==null&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(!r){if(t.stateNode===null)throw Error(i(166));return V(t),t.subtreeFlags&=-33554433,null}if(o=ke.current,ga(t))ma(t,o);else{var s=lp(je.current);switch(o){case 1:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case 2:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;default:switch(a){case`svg`:o=s.createElementNS(`http://www.w3.org/2000/svg`,a);break;case`math`:o=s.createElementNS(`http://www.w3.org/1998/Math/MathML`,a);break;case`script`:o=s.createElement(`div`),o.innerHTML=`<script><\/script>`,o=o.removeChild(o.firstChild);break;case`select`:o=typeof r.is==`string`?s.createElement(`select`,{is:r.is}):s.createElement(`select`),r.multiple?o.multiple=!0:r.size&&(o.size=r.size);break;default:o=typeof r.is==`string`?s.createElement(a,{is:r.is}):s.createElement(a)}}o[At]=t,o[jt]=r;a:for(s=t.child;s!==null;){if(s.tag===5||s.tag===6)o.appendChild(s.stateNode);else if(s.tag!==4&&s.tag!==27&&s.child!==null){s.child.return=s,s=s.child;continue}if(s===t)break a;for(;s.sibling===null;){if(s.return===null||s.return===t)break a;s=s.return}s.sibling.return=s.return,s=s.sibling}t.stateNode=o;a:switch(np(o,a,r),a){case`button`:case`input`:case`select`:case`textarea`:r=!!r.autoFocus;break a;case`img`:r=!0;break a;default:r=!1}r&&gl(t)}}return V(t),t.subtreeFlags&=-33554433,_l(t,t.type,e===null?null:e.memoizedProps,t.pendingProps,n),null;case 6:if(e&&t.stateNode!=null)e.memoizedProps!==r&&gl(t);else{if(typeof r!=`string`&&t.stateNode===null)throw Error(i(166));if(e=je.current,ga(t)){if(e=t.stateNode,n=t.memoizedProps,r=null,a=la,a!==null)switch(a.tag){case 27:case 5:r=a.memoizedProps}e[At]=t,e=!!(e.nodeValue===n||r!==null&&!0===r.suppressHydrationWarning||ep(e.nodeValue,n)),e||pa(t,!0)}else e=lp(e).createTextNode(r),e[At]=t,t.stateNode=e}return V(t),null;case 31:if(n=t.memoizedState,e===null||e.memoizedState!==null){if(r=ga(t),n!==null){if(e===null){if(!r)throw Error(i(318));if(e=t.memoizedState,e=e===null?null:e.dehydrated,!e)throw Error(i(557));e[At]=t}else _a(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;V(t),e=!1}else n=va(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=n),e=!0;if(!e)return t.flags&256?(Vo(t),t):(Vo(t),null);if(t.flags&128)throw Error(i(558))}return V(t),null;case 13:if(r=t.memoizedState,e===null||e.memoizedState!==null&&e.memoizedState.dehydrated!==null){if(a=ga(t),r!==null&&r.dehydrated!==null){if(e===null){if(!a)throw Error(i(318));if(a=t.memoizedState,a=a===null?null:a.dehydrated,!a)throw Error(i(317));a[At]=t}else _a(),!(t.flags&128)&&(t.memoizedState=null),t.flags|=4;V(t),a=!1}else a=va(),e!==null&&e.memoizedState!==null&&(e.memoizedState.hydrationErrors=a),a=!0;if(!a)return t.flags&256?(Vo(t),t):(Vo(t),null)}return Vo(t),t.flags&128?(t.lanes=n,t):(n=r!==null,e=e!==null&&e.memoizedState!==null,n&&(r=t.child,a=null,r.alternate!==null&&r.alternate.memoizedState!==null&&r.alternate.memoizedState.cachePool!==null&&(a=r.alternate.memoizedState.cachePool.pool),o=null,r.memoizedState!==null&&r.memoizedState.cachePool!==null&&(o=r.memoizedState.cachePool.pool),o!==a&&(r.flags|=2048)),n!==e&&n&&(t.child.flags|=8192),yl(t,t.updateQueue),V(t),null);case 4:return Pe(),e===null&&Wf(t.stateNode.containerInfo),t.flags|=67108864,V(t),null;case 10:return wa(t.type),V(t),null;case 19:if(Wo(t),r=t.memoizedState,r===null)return V(t),null;if(a=!!(t.flags&128),o=r.rendering,o===null){if(a)bl(r,!1);else{if(Y!==0||e!==null&&e.flags&128)for(e=t.child;e!==null;){if(o=Go(e),o!==null){for(t.flags|=128,bl(r,!1),e=o.updateQueue,t.updateQueue=e,yl(t,e),t.subtreeFlags=0,e=n,n=t.child;n!==null;)Vi(n,e),n=n.sibling;return Uo(t,Ho.current&1|2),F&&ia(t,r.treeForkCount),t.child}e=e.sibling}r.tail!==null&&Je()>_d&&(t.flags|=128,a=!0,bl(r,!1),t.lanes=4194304)}}else{if(!a){if(e=Go(o),e!==null){if(t.flags|=128,a=!0,e=e.updateQueue,t.updateQueue=e,yl(t,e),bl(r,!0),r.tail===null&&r.tailMode!==`collapsed`&&r.tailMode!==`visible`&&!o.alternate&&!F)return V(t),null}else 2*Je()-r.renderingStartTime>_d&&n!==536870912&&(t.flags|=128,a=!0,bl(r,!1),t.lanes=4194304)}r.isBackwards?(o.sibling=t.child,t.child=o):(e=r.last,e===null?t.child=o:e.sibling=o,r.last=o)}if(r.tail!==null){e=r.tail;a:{for(n=e;n!==null;){if(n.alternate!==null){n=!1;break a}n=n.sibling}n=!0}return r.rendering=e,r.tail=e.sibling,r.renderingStartTime=Je(),e.sibling=null,o=Ho.current,o=a?o&1|2:o&1,r.tailMode===`visible`||r.tailMode===`collapsed`||!n||F?Uo(t,o):(n=o,k(Fo,t),k(Ho,n),Io===null&&(Io=t)),F&&ia(t,r.treeForkCount),e}return V(t),null;case 22:case 23:return Vo(t),Po(),r=t.memoizedState!==null,e===null?r&&(t.flags|=8192):e.memoizedState!==null!==r&&(t.flags|=8192),r?n&536870912&&!(t.flags&128)&&(V(t),t.subtreeFlags&6&&(t.flags|=8192)):V(t),n=t.updateQueue,n!==null&&yl(t,n.retryQueue),n=null,e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),r=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(r=t.memoizedState.cachePool.pool),r!==n&&(t.flags|=2048),e!==null&&Oe(Ya),null;case 24:return n=null,e!==null&&(n=e.memoizedState.cache),t.memoizedState.cache!==n&&(t.flags|=2048),wa(I),V(t),null;case 25:return null;case 30:return t.flags|=33554432,V(t),null}throw Error(i(156,t.tag))}function Sl(e,t){switch(sa(t),t.tag){case 1:return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 3:return wa(I),Pe(),e=t.flags,e&65536&&!(e&128)?(t.flags=e&-65537|128,t):null;case 26:case 27:case 5:return Ie(t),null;case 31:if(t.memoizedState!==null){if(Vo(t),t.alternate===null)throw Error(i(340));_a()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 13:if(Vo(t),e=t.memoizedState,e!==null&&e.dehydrated!==null){if(t.alternate===null)throw Error(i(340));_a()}return e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 19:return Wo(t),e=t.flags,e&65536?(t.flags=e&-65537|128,e=t.memoizedState,e!==null&&(e.rendering=null,e.tail=null),t.flags|=4,t):null;case 4:return Pe(),null;case 10:return wa(t.type),null;case 22:case 23:return Vo(t),Po(),e!==null&&Oe(Ya),e=t.flags,e&65536?(t.flags=e&-65537|128,t):null;case 24:return wa(I),null;case 25:return null;default:return null}}function Cl(e,t){switch(sa(t),t.tag){case 3:wa(I),Pe();break;case 26:case 27:case 5:Ie(t);break;case 4:Pe();break;case 31:t.memoizedState!==null&&Vo(t);break;case 13:Vo(t);break;case 19:Wo(t);break;case 10:wa(t.type);break;case 22:case 23:Vo(t),Po(),e!==null&&Oe(Ya);break;case 24:wa(I)}}function wl(e,t){try{var n=t.updateQueue,r=n===null?null:n.lastEffect;if(r!==null){var i=r.next;n=i;do{if((n.tag&e)===e){r=void 0;var a=n.create,o=n.inst;r=a(),o.destroy=r}n=n.next}while(n!==i)}}catch(e){Z(t,t.return,e)}}function Tl(e,t,n){try{var r=t.updateQueue,i=r===null?null:r.lastEffect;if(i!==null){var a=i.next;r=a;do{if((r.tag&e)===e){var o=r.inst,s=o.destroy;if(s!==void 0){o.destroy=void 0,i=t;var c=n,l=s;try{l()}catch(e){Z(i,c,e)}}}r=r.next}while(r!==a)}}catch(e){Z(t,t.return,e)}}function El(e){var t=e.updateQueue;if(t!==null){var n=e.stateNode;try{ko(t,n)}catch(t){Z(e,e.return,t)}}}function Dl(e,t,n){n.props=Tc(e.type,e.memoizedProps),n.state=e.memoizedState;try{n.componentWillUnmount()}catch(n){Z(e,t,n)}}function Ol(e,t){try{var n=e.ref;if(n!==null){switch(e.tag){case 26:case 27:case 5:var r=e.stateNode;break;case 30:var i=e.stateNode,a=Ci(e.memoizedProps,i);(i.ref===null||i.ref.name!==a)&&(i.ref=Pp(a)),r=i.ref;break;case 7:if(e.stateNode===null){var o=new Fp(e);h(e.child,!1,Qp,o,void 0,void 0),e.stateNode=o}r=e.stateNode;break;default:r=e.stateNode}typeof n==`function`?e.refCleanup=n(r):n.current=r}}catch(n){Z(e,t,n)}}function kl(e,t){var n=e.ref,r=e.refCleanup;if(n!==null){if(typeof r==`function`)try{r()}catch(n){Z(e,t,n)}finally{e.refCleanup=null,e=e.alternate,e!=null&&(e.refCleanup=null)}else if(typeof n==`function`)try{n(null)}catch(n){Z(e,t,n)}else n.current=null}}function Al(e,t){if((e.tag===5||e.tag===27||e.tag===6)&&e.alternate===null&&t!==null)for(var n=0;n<t.length;n++)em(e.stateNode,t[n])}function jl(e){for(var t=e.return;t!==null&&(Pl(t)&&em(e.stateNode,t.stateNode),!Nl(t));)t=t.return}function Ml(e){for(var t=e.return;t!==null&&(Pl(t)&&tm(e.stateNode,t.stateNode),!Nl(t));)t=t.return}function Nl(e){return e.tag===5||e.tag===3||e.tag===27}function Pl(e){return e&&e.tag===7&&e.stateNode!==null}function Fl(e){var t=e.type,n=e.memoizedProps,r=e.stateNode;try{a:switch(t){case`button`:case`input`:case`select`:case`textarea`:n.autoFocus&&r.focus();break a;case`img`:n.src?r.src=n.src:n.srcSet&&(r.srcset=n.srcSet)}}catch(t){Z(e,e.return,t)}}function Il(e,t,n){try{var r=e.stateNode;ip(r,e.type,n,t),r[jt]=t}catch(t){Z(e,e.return,t)}}function Ll(e){return e.tag===5||e.tag===3||e.tag===26||e.tag===27&&Sp(e.type)||e.tag===4}function Rl(e){a:for(;;){for(;e.sibling===null;){if(e.return===null||Ll(e.return))return null;e=e.return}for(e.sibling.return=e.return,e=e.sibling;e.tag!==5&&e.tag!==6&&e.tag!==18;){if(e.tag===27&&Sp(e.type)||e.flags&2||e.child===null||e.tag===4)continue a;e.child.return=e,e=e.child}if(!(e.flags&2))return e.stateNode}}function zl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?(n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n).insertBefore(i,t):(t=n.nodeType===9?n.body:n.nodeName===`HTML`?n.ownerDocument.body:n,t.appendChild(i),n=n._reactRootContainer,n!=null||t.onclick!==null||(t.onclick=wn)),Al(e,r),M=!0;else if(i!==4&&(i===27&&(Al(e,r),r=null,Sp(e.type)&&(n=e.stateNode,t=null)),e=e.child,e!==null))for(zl(e,t,n,r),e=e.sibling;e!==null;)zl(e,t,n,r),e=e.sibling}function Bl(e,t,n,r){var i=e.tag;if(i===5||i===6)i=e.stateNode,t?n.insertBefore(i,t):n.appendChild(i),Al(e,r),M=!0;else if(i!==4&&(i===27&&(Al(e,r),r=null,Sp(e.type)&&(n=e.stateNode)),e=e.child,e!==null))for(Bl(e,t,n,r),e=e.sibling;e!==null;)Bl(e,t,n,r),e=e.sibling}function Vl(e){var t=e.stateNode,n=e.memoizedProps;try{for(var r=e.type,i=t.attributes;i.length;)t.removeAttributeNode(i[0]);np(t,r,n),t[At]=e,t[jt]=n}catch(t){Z(e,e.return,t)}}var Hl=!1,Ul=null;function Wl(e){(e.tag===30||e.subtreeFlags&33554432)&&(Hl=!0)}var Gl=null;function Kl(){var e=Gl;return Gl=null,e}var ql=0;function Jl(e,t,n,r,i){return ql=0,Yl(e.child,t,n,r,i)}function Yl(e,t,n,r,i){for(var a=!1;e!==null;){if(e.tag===5){var o=e.stateNode;if(r!==null){var s=Op(o);r.push(s),s.view&&(a=!0)}else a||Op(o).view&&(a=!0);Hl=!0,Tp(o,ql===0?t:t+`_`+ql,n),ql++}else(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&i||Yl(e.child,t,n,r,i)&&(a=!0));e=e.sibling}return a}function Xl(e,t){for(;e!==null;)e.tag===5?Ep(e.stateNode,e.memoizedProps):(e.tag!==22||e.memoizedState===null)&&(e.tag===30&&t||Xl(e.child,t)),e=e.sibling}function Zl(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if((e.tag!==22||e.memoizedState===null)&&(Zl(e),e.tag===30&&e.flags&18874368&&e.stateNode.paired)){var t=e.memoizedProps;if(t.name==null||t.name===`auto`)throw Error(i(544));var n=t.name;t=Ti(t.default,t.share),t!==`none`&&(Jl(e,n,t,null,!1)||Xl(e.child,!1))}e=e.sibling}}function Ql(e,t){if(e.tag===30){var n=e.stateNode,r=e.memoizedProps,i=Ci(r,n),a=Ti(r.default,n.paired?r.share:r.enter);a===`none`?Zl(e):Jl(e,i,a,null,!1)?(Zl(e),n.paired||t||Nd(e,r.onEnter)):Xl(e.child,!1)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)Ql(e,t),e=e.sibling;else Zl(e)}function $l(e){if(Ul!==null&&Ul.size!==0){var t=Ul;if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var n=e.memoizedProps,r=n.name;if(r!=null&&r!==`auto`){var i=t.get(r);if(i!==void 0){var a=Ti(n.default,n.share);if(a!==`none`&&(Jl(e,r,a,null,!1)?(a=e.stateNode,i.paired=a,a.paired=i,Nd(e,n.onShare)):Xl(e.child,!1)),t.delete(r),t.size===0)break}}}$l(e)}e=e.sibling}}}function eu(e){if(e.tag===30){var t=e.memoizedProps,n=Ci(t,e.stateNode),r=Ul===null?void 0:Ul.get(n),i=Ti(t.default,r===void 0?t.exit:t.share);i!==`none`&&(Jl(e,n,i,null,!1)?r===void 0?Nd(e,t.onExit):(i=e.stateNode,r.paired=i,i.paired=r,Ul.delete(n),Nd(e,t.onShare)):Xl(e.child,!1)),Ul!==null&&$l(e)}else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)eu(e),e=e.sibling;else Ul!==null&&$l(e)}function tu(e){for(e=e.child;e!==null;){if(e.tag===30){var t=e.memoizedProps,n=Ci(t,e.stateNode);t=Ti(t.default,t.update),e.flags&=-5,t!==`none`&&Jl(e,n,t,e.memoizedState=[],!1)}else e.subtreeFlags&33554432&&tu(e);e=e.sibling}}function nu(e){if(e.subtreeFlags&18874368)for(e=e.child;e!==null;){if(e.tag!==22||e.memoizedState===null){if(e.tag===30&&e.flags&18874368){var t=e.stateNode;t.paired!==null&&(t.paired=null,Xl(e.child,!1))}nu(e)}e=e.sibling}}function ru(e){if(e.tag===30)e.stateNode.paired=null,Xl(e.child,!1),nu(e);else if(e.subtreeFlags&33554432)for(e=e.child;e!==null;)ru(e),e=e.sibling;else nu(e)}function iu(e){for(e=e.child;e!==null;)e.tag===30?Xl(e.child,!1):e.subtreeFlags&33554432&&iu(e),e=e.sibling}function au(e,t,n,r,i,a,o){for(var s=!1;t!==null;){if(t.tag===5){var c=t.stateNode;if(a!==null&&ql<a.length){var l=a[ql],u=Op(c);(l.view||u.view)&&(s=!0);var d;if(d=!(e.flags&4)){if(u.clip)d=!0;else{d=l.rect;var f=u.rect;d=d.y!==f.y||d.x!==f.x||d.height!==f.height||d.width!==f.width}}d&&(e.flags|=4),u.abs?u=!l.abs:(l=l.rect,u=u.rect,u=l.height!==u.height||l.width!==u.width),u&&(e.flags|=32)}else e.flags|=32;e.flags&4&&Tp(c,ql===0?n:n+`_`+ql,i),s&&e.flags&4||(Gl===null&&(Gl=[]),Gl.push(c,ql===0?r:r+`_`+ql,t.memoizedProps)),ql++}else(t.tag!==22||t.memoizedState===null)&&(t.tag===30&&o?e.flags|=t.flags&32:au(e,t.child,n,r,i,a,o)&&(s=!0));t=t.sibling}return s}function ou(e,t){for(e=e.child;e!==null;){if(e.tag===30){var n=e.memoizedProps,r=e.stateNode,i=Ci(n,r),a=Ti(n.default,n.update);if(t){r=r.clones;var o=r===null?null:r.map(kp)}else o=e.memoizedState,e.memoizedState=null;r=e;var s=e.child;ql=0,i=au(r,s,i,i,a,o,!1),e.flags&4&&i&&(t||Nd(e,n.onUpdate))}else e.subtreeFlags&33554432&&ou(e,t);e=e.sibling}}var su=!1,H=!1,cu=!1,lu=!1,uu=typeof WeakSet==`function`?WeakSet:Set,du=null,fu=!1,pu=!1,mu=!1,hu=!1;function gu(e,t,n){if(e=e.containerInfo,sp=gh,e=$r(e),ei(e)){if(`selectionStart`in e)var r={start:e.selectionStart,end:e.selectionEnd};else a:{r=(r=e.ownerDocument)&&r.defaultView||window;var i=r.getSelection&&r.getSelection();if(i&&i.rangeCount!==0){r=i.anchorNode;var a=i.anchorOffset,o=i.focusNode;i=i.focusOffset;try{r.nodeType,o.nodeType}catch{r=null;break a}var s=0,c=-1,l=-1,u=0,d=0,f=e,p=null;b:for(;;){for(var m;f!==r||a!==0&&f.nodeType!==3||(c=s+a),f!==o||i!==0&&f.nodeType!==3||(l=s+i),f.nodeType===3&&(s+=f.nodeValue.length),(m=f.firstChild)!==null;)p=f,f=m;for(;;){if(f===e)break b;if(p===r&&++u===a&&(c=s),p===o&&++d===i&&(l=s),(m=f.nextSibling)!==null)break;f=p,p=f.parentNode}f=m}r=c===-1||l===-1?null:{start:c,end:l}}else r=null}r||={start:0,end:0}}else r=null;for(cp={focusedElem:e,selectionRange:r},gh=!1,n=(n&335544064)===n,du=t,t=n?9270:1024;du!==null;){if(e=du,n&&(r=e.deletions,r!==null))for(a=0;a<r.length;a++)n&&eu(r[a]);if(e.alternate===null&&e.flags&2)n&&Wl(e),_u(n);else{if(e.tag===22){if(r=e.alternate,e.memoizedState!==null){r!==null&&r.memoizedState===null&&n&&eu(r),_u(n);continue}if(r!==null&&r.memoizedState!==null){n&&Wl(e),_u(n);continue}}r=e.child,(e.subtreeFlags&t)!==0&&r!==null?(r.return=e,du=r):(n&&tu(e),_u(n))}}Ul=null}function _u(e){for(;du!==null;){var t=du,n=e,r=t.alternate,a=t.flags;switch(t.tag){case 0:case 11:case 15:break;case 1:if(a&1024&&r!==null){n=void 0,a=r.memoizedProps,r=r.memoizedState;var o=t.stateNode;try{var s=Tc(t.type,a);n=o.getSnapshotBeforeUpdate(s,r),o.__reactInternalSnapshotBeforeUpdate=n}catch(e){Z(t,t.return,e)}}break;case 3:if(a&1024){if(r=t.stateNode.containerInfo,n=r.nodeType,n===9)nm(r);else if(n===1)switch(r.nodeName){case`HEAD`:case`HTML`:case`BODY`:nm(r);break;default:r.textContent=``}}break;case 5:case 26:case 27:case 6:case 4:case 17:break;case 30:n&&r!==null&&(n=Ci(r.memoizedProps,r.stateNode),a=t.memoizedProps,a=Ti(a.default,a.update),a!==`none`&&Jl(r,n,a,r.memoizedState=[],!0));break;default:if(a&1024)throw Error(i(163))}if(r=t.sibling,r!==null){r.return=t.return,du=r;break}du=t.return}}function vu(e,t,n){var r=n.flags;switch(n.tag){case 0:case 11:case 15:Lu(e,n),r&4&&wl(5,n);break;case 1:if(Lu(e,n),r&4){if(e=n.stateNode,t===null)try{e.componentDidMount()}catch(e){Z(n,n.return,e)}else{var i=Tc(n.type,t.memoizedProps);t=t.memoizedState;try{e.componentDidUpdate(i,t,e.__reactInternalSnapshotBeforeUpdate)}catch(e){Z(n,n.return,e)}}}r&64&&El(n),r&512&&Ol(n,n.return);break;case 3:if(Lu(e,n),r&64&&(e=n.updateQueue,e!==null)){if(t=null,n.child!==null)switch(n.child.tag){case 27:case 5:t=n.child.stateNode;break;case 1:t=n.child.stateNode}try{ko(e,t)}catch(e){Z(n,n.return,e)}}break;case 27:t===null&&r&4&&Vl(n);case 26:case 5:Lu(e,n),t===null&&r&4&&Fl(n),r&512&&Ol(n,n.return);break;case 12:Lu(e,n);break;case 31:Lu(e,n),r&4&&Eu(e,n);break;case 13:Lu(e,n),r&4&&Du(e,n),r&64&&(e=n.memoizedState,e!==null&&(e=e.dehydrated,e!==null&&(n=_f.bind(null,n),cm(e,n))));break;case 22:if(r=n.memoizedState!==null||su,!r){var a=t!==null&&t.memoizedState!==null||H;t=su,i=H,su=r,(H=a)&&!i?(r=2,n.subtreeFlags&8772&&(r|=1),zu(e,n,r)):Lu(e,n),su=t,H=i}break;case 30:Lu(e,n),r&512&&Ol(n,n.return);break;case 7:r&512&&Ol(n,n.return);default:Lu(e,n)}}function yu(e,t){for(e=e.child;e!==null;)bu(e,t),e=e.sibling}function bu(e,t){switch(e.tag){case 5:case 26:try{var n=e.stateNode;if(t){var r=n.style;typeof r.setProperty==`function`?r.setProperty(`display`,`none`,`important`):r.display=`none`}else{var i=e.stateNode,a=e.memoizedProps.style,o=a!=null&&a.hasOwnProperty(`display`)?a.display:null;i.style.display=o==null||typeof o==`boolean`?``:(``+o).trim()}}catch(t){Z(e,e.return,t)}xu(e,t);break;case 6:try{e.stateNode.nodeValue=t?``:e.memoizedProps,M=!0}catch(t){Z(e,e.return,t)}break;case 18:try{var s=e.stateNode;t?wp(s,!0):wp(e.stateNode,!1)}catch(t){Z(e,e.return,t)}break;case 22:case 23:e.memoizedState===null&&yu(e,t);break;default:yu(e,t)}}function xu(e,t){if(e.subtreeFlags&67108864)for(e=e.child;e!==null;){a:{var n=e,r=t;switch(n.tag){case 4:bu(n,r);break a;case 22:n.memoizedState===null&&xu(n,r);break a;default:xu(n,r)}}e=e.sibling}}function Su(e){var t=e.alternate;t!==null&&(e.alternate=null,Su(t)),e.child=null,e.deletions=null,e.sibling=null,e.tag===5&&(t=e.stateNode,t!==null&&zt(t)),e.stateNode=null,e.return=null,e.dependencies=null,e.memoizedProps=null,e.memoizedState=null,e.pendingProps=null,e.stateNode=null,e.updateQueue=null}var U=null,Cu=!1;function wu(e,t,n){for(n=n.child;n!==null;)Tu(e,t,n),n=n.sibling}function Tu(e,t,n){if(it&&typeof it.onCommitFiberUnmount==`function`)try{it.onCommitFiberUnmount(rt,n)}catch{}switch(n.tag){case 26:H||kl(n,t),wu(e,t,n),n.memoizedState?n.memoizedState.count--:n.stateNode&&!H&&(n=n.stateNode,n.parentNode.removeChild(n));break;case 27:H||kl(n,t),Ml(n);var r=U,i=Cu;Sp(n.type)&&(U=n.stateNode,Cu=!1),wu(e,t,n),gm(n.stateNode,n.type,n.memoizedProps),U=r,Cu=i;break;case 5:H||kl(n,t),Ml(n);case 6:if(n.tag===6&&Ml(n),r=U,i=Cu,U=null,wu(e,t,n),U=r,Cu=i,U!==null){if(Cu)try{(U.nodeType===9?U.body:U.nodeName===`HTML`?U.ownerDocument.body:U).removeChild(n.stateNode),M=!0}catch(e){Z(n,t,e)}else try{U.removeChild(n.stateNode),M=!0}catch(e){Z(n,t,e)}}break;case 18:U!==null&&(Cu?(e=U,Cp(e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,n.stateNode),Hh(e)):Cp(U,n.stateNode));break;case 4:r=U,i=Cu,U=n.stateNode.containerInfo,Cu=!0,wu(e,t,n),U=r,Cu=i;break;case 0:case 11:case 14:case 15:Tl(2,n,t),H||Tl(4,n,t),wu(e,t,n);break;case 1:H||(kl(n,t),r=n.stateNode,typeof r.componentWillUnmount==`function`&&Dl(n,t,r)),wu(e,t,n);break;case 21:wu(e,t,n);break;case 22:H=(r=H)||n.memoizedState!==null,wu(e,t,n),H=r;break;case 30:kl(n,t),wu(e,t,n);break;case 7:H||kl(n,t),wu(e,t,n);break;default:wu(e,t,n)}}function Eu(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null))){e=e.dehydrated;try{Hh(e)}catch(e){Z(t,t.return,e)}}}function Du(e,t){if(t.memoizedState===null&&(e=t.alternate,e!==null&&(e=e.memoizedState,e!==null&&(e=e.dehydrated,e!==null))))try{Hh(e)}catch(e){Z(t,t.return,e)}}function Ou(e){switch(e.tag){case 31:case 13:case 19:var t=e.stateNode;return t===null&&(t=e.stateNode=new uu),t;case 22:return e=e.stateNode,t=e._retryCache,t===null&&(t=e._retryCache=new uu),t;default:throw Error(i(435,e.tag))}}function ku(e,t){var n=Ou(e);t.forEach(function(t){if(!n.has(t)){n.add(t);var r=vf.bind(null,e,t);t.then(r,r)}})}function Au(e,t,n){var r=t.deletions;if(r!==null)for(var a=0;a<r.length;a++){var o=r[a],s=e,c=t,l=c;a:for(;l!==null;){switch(l.tag){case 27:if(Sp(l.type)){U=l.stateNode,Cu=!1;break a}break;case 5:U=l.stateNode,Cu=!1;break a;case 3:case 4:U=l.stateNode.containerInfo,Cu=!0;break a}l=l.return}if(U===null)throw Error(i(160));Tu(s,c,o),U=null,Cu=!1,s=o.alternate,s!==null&&(s.return=null),o.return=null}if(t.subtreeFlags&13886)for(t=t.child;t!==null;)Mu(t,e,n),t=t.sibling}var ju=null;function Mu(e,t,n){var r=e.alternate,a=e.flags;switch(e.tag){case 0:case 11:case 14:case 15:if(a&4&&(r=e.updateQueue,r=r===null?null:r.events,r!==null))for(var o=0;o<r.length;o++){var s=r[o];s.ref.impl=s.nextImpl}Au(t,e,n),Nu(e),a&4&&(Tl(3,e,e.return),wl(3,e),Tl(5,e,e.return));break;case 1:Au(t,e,n),Nu(e),a&512&&(H||r===null||kl(r,r.return)),a&64&&su&&(e=e.updateQueue,e!==null&&(t=e.callbacks,t!==null&&(n=e.shared.hiddenCallbacks,e.shared.hiddenCallbacks=n===null?t:n.concat(t))));break;case 26:if(o=ju,Au(t,e,n),Nu(e),a&512&&(H||r===null||kl(r,r.return)),a&4){if(a=r===null?null:r.memoizedState,n=e.memoizedState,r===null){if(n===null){if(e.stateNode===null){if(su)e.stateNode=fp(e.type,e.memoizedProps,t.containerInfo,e);else{a:{t=e.type,n=e.memoizedProps,a=o.ownerDocument||o;b:switch(t){case`title`:r=a.getElementsByTagName(`title`)[0],(!r||r[Lt]||r[At]||r.namespaceURI===`http://www.w3.org/2000/svg`||r.hasAttribute(`itemprop`))&&(r=a.createElement(t),a.head.insertBefore(r,a.querySelector(`head > title`))),np(r,t,n),r[At]=e,j(r),t=r;break a;case`link`:if(o=Gm(`link`,`href`,a).get(t+(n.href||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`href`)===(n.href==null||n.href===``?null:n.href)&&r.getAttribute(`rel`)===(n.rel==null?null:n.rel)&&r.getAttribute(`title`)===(n.title==null?null:n.title)&&r.getAttribute(`crossorigin`)===(n.crossOrigin==null?null:n.crossOrigin)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;case`meta`:if(o=Gm(`meta`,`content`,a).get(t+(n.content||``))){for(s=0;s<o.length;s++)if(r=o[s],r.getAttribute(`content`)===(n.content==null?null:``+n.content)&&r.getAttribute(`name`)===(n.name==null?null:n.name)&&r.getAttribute(`property`)===(n.property==null?null:n.property)&&r.getAttribute(`http-equiv`)===(n.httpEquiv==null?null:n.httpEquiv)&&r.getAttribute(`charset`)===(n.charSet==null?null:n.charSet)){o.splice(s,1);break b}}r=a.createElement(t),np(r,t,n),a.head.appendChild(r);break;default:throw Error(i(468,t))}r[At]=e,j(r),t=r}e.stateNode=t}}else su||Km(o,e.type,e.stateNode)}else e.stateNode=Bm(o,n,e.memoizedProps)}else a===n?n===null&&e.stateNode!==null&&Il(e,e.memoizedProps,r.memoizedProps):(a===null?(t=r.stateNode,t===null||H||t.parentNode.removeChild(t)):a.count--,n===null?su||Km(o,e.type,e.stateNode):Bm(o,n,e.memoizedProps))}break;case 27:Au(t,e,n),Nu(e),a&512&&(H||r===null||kl(r,r.return)),r!==null&&a&4&&Il(e,e.memoizedProps,r.memoizedProps);break;case 5:if(o=cu,cu=!1,Au(t,e,n),cu=o,Nu(e),a&512&&(H||r===null||kl(r,r.return)),e.flags&32){t=e.stateNode;try{gn(t,``),M=!0}catch(t){Z(e,e.return,t)}}a&4&&e.stateNode!=null&&(t=e.memoizedProps,Il(e,t,r===null?t:r.memoizedProps)),a&1024&&(lu=!0);break;case 6:if(Au(t,e,n),Nu(e),a&4){if(e.stateNode===null)throw Error(i(162));t=e.memoizedProps,n=e.stateNode;try{n.nodeValue=t,M=!0}catch(t){Z(e,e.return,t)}}break;case 3:if(M=!1,Wm=null,o=ju,ju=bm(t.containerInfo),Au(t,e,n),ju=o,Nu(e),a&4&&r!==null&&r.memoizedState.isDehydrated)try{Hh(t.containerInfo)}catch(t){Z(e,e.return,t)}lu&&(lu=!1,Pu(e)),M=!1;break;case 4:a=cu,cu=su,r=$t(),o=ju,ju=bm(e.stateNode.containerInfo),Au(t,e,n),Nu(e),ju=o,M&&pu&&(mu=!0),M=r,cu=a;break;case 12:Au(t,e,n),Nu(e);break;case 31:Au(t,e,n),Nu(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 13:Au(t,e,n),Nu(e),e.child.flags&8192&&e.memoizedState!==null!=(r!==null&&r.memoizedState!==null)&&(hd=Je()),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 22:o=e.memoizedState!==null,s=r!==null&&r.memoizedState!==null;var c=su,l=H,u=cu;su=c||o,cu=u||o,H=l||s,Au(t,e,n),H=l,cu=u,su=c,Nu(e),a&8192&&(t=e.stateNode,t._visibility=o?t._visibility&-2:t._visibility|1,!o||r===null||s||su||H||(t=s||H,n=su,r=H,su=o||su,H=t,Ru(e,2),su=n,H=r),!o&&cu||yu(e,o)),a&4&&(t=e.updateQueue,t!==null&&(n=t.retryQueue,n!==null&&(t.retryQueue=null,ku(e,n))));break;case 19:Au(t,e,n),Nu(e),a&4&&(t=e.updateQueue,t!==null&&(e.updateQueue=null,ku(e,t)));break;case 30:a&512&&(H||r===null||kl(r,r.return)),a=$t(),o=pu,s=(n&335544064)===n,c=e.memoizedProps,pu=s&&Ti(c.default,c.update)!==`none`,Au(t,e,n),Nu(e),s&&r!==null&&M&&(e.flags|=4),pu=o,M=a;break;case 21:break;case 7:a&512&&(H||r===null||kl(r,r.return)),r&&r.stateNode!==null&&(r.stateNode._fragmentFiber=e);default:Au(t,e,n),Nu(e)}}function Nu(e){var t=e.flags;if(t&2){try{for(var n,r=e.return;r!==null;){if(Ll(r)){n=r;break}r=r.return}r=null;for(var a=e.return;a!==null;){if(Pl(a)){var o=a.stateNode;r===null?r=[o]:r.push(o)}if(Nl(a))break;a=a.return}var s=r;if(n==null)throw Error(i(160));switch(n.tag){case 27:var c=n.stateNode;Bl(e,Rl(e),c,s);break;case 5:var l=n.stateNode;n.flags&32&&(gn(l,``),n.flags&=-33),Bl(e,Rl(e),l,s);break;case 3:case 4:var u=n.stateNode.containerInfo;zl(e,Rl(e),u,s);break;default:throw Error(i(161))}}catch(t){Z(e,e.return,t)}e.flags&=-3}t&4096&&(e.flags&=-4097)}function Pu(e){if(e.subtreeFlags&1024)for(e=e.child;e!==null;){var t=e;Pu(t),t.tag===5&&t.flags&1024&&(t=t.stateNode,gh=!0,t.reset(),gh=!1),e=e.sibling}}function Fu(e,t){if(t.subtreeFlags&9270)for(t=t.child;t!==null;)Iu(t,e),t=t.sibling;else ou(t,!1)}function Iu(e,t){var n=e.alternate;if(n===null)Ql(e,!1);else switch(e.tag){case 3:if(hu=fu=!1,Kl(),Fu(t,e),!fu&&!mu){if(e=Gl,e!==null)for(var r=0;r<e.length;r+=3){n=e[r];var i=e[r+1];Ep(n,e[r+2]),n=n.ownerDocument.documentElement,n!==null&&n.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(`+i+`)`})}e=t.containerInfo,e=e.nodeType===9?e.documentElement:e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===``&&(e.style.viewTransitionName=`none`,e.animate({opacity:[0,0],pointerEvents:[`none`,`none`]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition-group(root)`}),e.animate({width:[0,0],height:[0,0]},{duration:0,fill:`forwards`,pseudoElement:`::view-transition`})),hu=!0}Gl=null;break;case 5:Fu(t,e);break;case 4:r=fu,fu=!1,Fu(t,e),fu&&(mu=!0),fu=r;break;case 22:e.memoizedState===null&&(n.memoizedState===null?Fu(t,e):Ql(e,!1));break;case 30:r=fu,i=Kl(),fu=!1,Fu(t,e),fu&&(e.flags|=4);var a=e.memoizedProps,o=e.stateNode;t=Ci(a,o),o=Ci(n.memoizedProps,o);var s=Ti(a.default,a.update);s===`none`?t=!1:(a=n.memoizedState,n.memoizedState=null,n=e.child,ql=0,t=au(e,n,t,o,s,a,!0),ql!==(a===null?0:a.length)&&(e.flags|=32)),e.flags&4&&t?(Nd(e,e.memoizedProps.onUpdate),Gl=i):i!==null&&(i.push.apply(i,Gl),Gl=i),fu=e.flags&32?!0:r;break;default:Fu(t,e)}}function Lu(e,t){if(t.subtreeFlags&8772)for(t=t.child;t!==null;)vu(e,t.alternate,t),t=t.sibling}function Ru(e,t){for(e=e.child;e!==null;){var n=e,r=t;switch(n.tag){case 0:case 11:case 14:case 15:Tl(4,n,n.return),Ru(n,r);break;case 1:kl(n,n.return);var i=n.stateNode;typeof i.componentWillUnmount==`function`&&Dl(n,n.return,i),Ru(n,r);break;case 27:r&2&&gm(n.stateNode,n.type,n.memoizedProps);case 5:kl(n,n.return),n.tag!==5&&n.tag!==27||Ml(n),Ru(n,r);break;case 6:Ml(n);break;case 26:kl(n,n.return),i=n.stateNode,n.memoizedState!==null||i===null||H||i.parentNode.removeChild(i),Ru(n,r);break;case 22:n.memoizedState===null&&Ru(n,r);break;case 30:kl(n,n.return),Ru(n,r);break;case 7:kl(n,n.return);default:Ru(n,r)}e=e.sibling}}function zu(e,t,n){for(n=t.subtreeFlags&8772?n:n&-2,t=t.child;t!==null;){var r=t.alternate,i=e,a=t,o=a.flags,s=!!(n&1);switch(a.tag){case 0:case 11:case 15:zu(i,a,n),wl(4,a);break;case 1:if(zu(i,a,n),r=a,i=r.stateNode,typeof i.componentDidMount==`function`)try{i.componentDidMount()}catch(e){Z(r,r.return,e)}if(r=a,i=r.updateQueue,i!==null){var c=r.stateNode;try{var l=i.shared.hiddenCallbacks;if(l!==null)for(i.shared.hiddenCallbacks=null,i=0;i<l.length;i++)Oo(l[i],c)}catch(e){Z(r,r.return,e)}}s&&o&64&&El(a),Ol(a,a.return);break;case 27:n&2&&Vl(a);case 5:a.tag!==5&&a.tag!==27||jl(a),zu(i,a,n),s&&r===null&&o&4&&Fl(a),Ol(a,a.return);break;case 6:jl(a);break;case 26:c=a.stateNode,a.memoizedState!==null||c===null||su||Km(bm(c.ownerDocument),a.type,c),zu(i,a,n),s&&r===null&&o&4&&Fl(a),Ol(a,a.return);break;case 12:zu(i,a,n);break;case 31:zu(i,a,n),s&&o&4&&Eu(i,a);break;case 13:zu(i,a,n),s&&o&4&&Du(i,a);break;case 22:a.memoizedState===null&&zu(i,a,n),Ol(a,a.return);break;case 30:zu(i,a,n),Ol(a,a.return);break;case 7:Ol(a,a.return);default:zu(i,a,n)}t=t.sibling}}function Bu(e,t){var n=null;e!==null&&e.memoizedState!==null&&e.memoizedState.cachePool!==null&&(n=e.memoizedState.cachePool.pool),e=null,t.memoizedState!==null&&t.memoizedState.cachePool!==null&&(e=t.memoizedState.cachePool.pool),e!==n&&(e!=null&&e.refCount++,n!=null&&La(n))}function Vu(e,t){e=null,t.alternate!==null&&(e=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==e&&(t.refCount++,e!=null&&La(e))}function Hu(e,t,n,r){var i=(n&335544064)===n;if(t.subtreeFlags&(i?10262:10256))for(t=t.child;t!==null;)Uu(e,t,n,r),t=t.sibling;else i&&iu(t)}function Uu(e,t,n,r){var i=(n&335544064)===n;i&&t.alternate===null&&t.return!==null&&t.return.alternate!==null&&ru(t);var a=t.flags;switch(t.tag){case 0:case 11:case 15:Hu(e,t,n,r),a&2048&&wl(9,t);break;case 1:Hu(e,t,n,r);break;case 3:Hu(e,t,n,r),i&&hu&&(e=e.containerInfo,e=e.nodeType===9?e.body:e.nodeName===`HTML`?e.ownerDocument.body:e,e.style.viewTransitionName===`root`&&(e.style.viewTransitionName=``),e=e.ownerDocument.documentElement,e!==null&&e.style.viewTransitionName===`none`&&(e.style.viewTransitionName=``)),a&2048&&(a=null,t.alternate!==null&&(a=t.alternate.memoizedState.cache),t=t.memoizedState.cache,t!==a&&(t.refCount++,a!=null&&La(a)));break;case 12:if(a&2048){Hu(e,t,n,r),a=t.stateNode;try{var o=t.memoizedProps,s=o.id,c=o.onPostCommit;typeof c==`function`&&c(s,t.alternate===null?`mount`:`update`,a.passiveEffectDuration,-0)}catch(e){Z(t,t.return,e)}}else Hu(e,t,n,r);break;case 31:Hu(e,t,n,r);break;case 13:Hu(e,t,n,r);break;case 23:break;case 22:o=t.stateNode,s=t.alternate,t.memoizedState===null?(i&&s!==null&&s.memoizedState!==null&&ru(t),o._visibility&2?Hu(e,t,n,r):(o._visibility|=2,Wu(e,t,n,r,!!(t.subtreeFlags&10256)||!1))):(i&&s!==null&&s.memoizedState===null&&ru(s),o._visibility&2?Hu(e,t,n,r):Gu(e,t)),a&2048&&Bu(s,t);break;case 24:Hu(e,t,n,r),a&2048&&Vu(t.alternate,t);break;case 30:i&&(a=t.alternate,a!==null&&(Xl(a.child,!0),Xl(t.child,!0))),Hu(e,t,n,r);break;default:Hu(e,t,n,r)}}function Wu(e,t,n,r,i){for(i&&=!!(t.subtreeFlags&10256)||!1,t=t.child;t!==null;){var a=e,o=t,s=n,c=r,l=o.flags;switch(o.tag){case 0:case 11:case 15:Wu(a,o,s,c,i),wl(8,o);break;case 23:break;case 22:var u=o.stateNode;o.memoizedState===null?(u._visibility|=2,Wu(a,o,s,c,i)):u._visibility&2?Wu(a,o,s,c,i):Gu(a,o),i&&l&2048&&Bu(o.alternate,o);break;case 24:Wu(a,o,s,c,i),i&&l&2048&&Vu(o.alternate,o);break;default:Wu(a,o,s,c,i)}t=t.sibling}}function Gu(e,t){if(t.subtreeFlags&10256)for(t=t.child;t!==null;){var n=e,r=t,i=r.flags;switch(r.tag){case 22:Gu(n,r),i&2048&&Bu(r.alternate,r);break;case 24:Gu(n,r),i&2048&&Vu(r.alternate,r);break;default:Gu(n,r)}t=t.sibling}}var Ku=8192;function qu(e,t,n){if(e.subtreeFlags&Ku)for(e=e.child;e!==null;)Ju(e,t,n),e=e.sibling}function Ju(e,t,n){switch(e.tag){case 26:qu(e,t,n),e.flags&Ku&&(e.memoizedState===null?(e=e.stateNode,(t&335544128)===t&&Zm(n,e)):Qm(n,ju,e.memoizedState,e.memoizedProps));break;case 5:qu(e,t,n),e.flags&Ku&&(e=e.stateNode,(t&335544128)===t&&Zm(n,e));break;case 3:case 4:var r=ju;ju=bm(e.stateNode.containerInfo),qu(e,t,n),ju=r;break;case 22:e.memoizedState===null&&(r=e.alternate,r!==null&&r.memoizedState!==null?(r=Ku,Ku=16777216,qu(e,t,n),Ku=r):qu(e,t,n));break;case 30:if((e.flags&Ku)!==0&&(r=e.memoizedProps.name,r!=null&&r!==`auto`)){var i=e.stateNode;i.paired=null,Ul===null&&(Ul=new Map),Ul.set(r,i)}qu(e,t,n);break;default:qu(e,t,n)}}function Yu(e){var t=e.alternate;if(t!==null&&(e=t.child,e!==null)){t.child=null;do t=e.sibling,e.sibling=null,e=t;while(e!==null)}}function Xu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];du=r,$u(r,e)}Yu(e)}if(e.subtreeFlags&10256)for(e=e.child;e!==null;)Zu(e),e=e.sibling}function Zu(e){switch(e.tag){case 0:case 11:case 15:Xu(e),e.flags&2048&&Tl(9,e,e.return);break;case 3:Xu(e);break;case 12:Xu(e);break;case 22:var t=e.stateNode;e.memoizedState!==null&&t._visibility&2&&(e.return===null||e.return.tag!==13)?(t._visibility&=-3,Qu(e)):Xu(e);break;default:Xu(e)}}function Qu(e){var t=e.deletions;if(e.flags&16){if(t!==null)for(var n=0;n<t.length;n++){var r=t[n];du=r,$u(r,e)}Yu(e)}for(e=e.child;e!==null;){switch(t=e,t.tag){case 0:case 11:case 15:Tl(8,t,t.return),Qu(t);break;case 22:n=t.stateNode,n._visibility&2&&(n._visibility&=-3,Qu(t));break;default:Qu(t)}e=e.sibling}}function $u(e,t){for(;du!==null;){var n=du;switch(n.tag){case 0:case 11:case 15:Tl(8,n,t);break;case 23:case 22:if(n.memoizedState!==null&&n.memoizedState.cachePool!==null){var r=n.memoizedState.cachePool.pool;r!=null&&r.refCount++}break;case 24:La(n.memoizedState.cache)}if(r=n.child,r!==null)r.return=n,du=r;else a:for(n=e;du!==null;){r=du;var i=r.sibling,a=r.return;if(Su(r),r===n){du=null;break a}if(i!==null){i.return=a,du=i;break a}du=a}}}var ed={getCacheForType:function(e){var t=Aa(I),n=t.data.get(e);return n===void 0&&(n=e(),t.data.set(e,n)),n},cacheSignal:function(){return Aa(I).controller.signal}},td=typeof WeakMap==`function`?WeakMap:Map,W=0,G=null,K=null,q=0,J=0,nd=null,rd=!1,id=!1,ad=!1,od=0,Y=0,sd=0,cd=0,ld=0,ud=0,dd=0,fd=null,pd=null,md=!1,hd=0,gd=0,_d=1/0,vd=null,yd=null,X=0,bd=null,xd=null,Sd=0,Cd=0,wd=null,Td=null,Ed=null,Dd=null,Od=null,kd=0,Ad=null;function jd(){return W&2&&q!==0?q&-q:D.T===null?Dt():Pf()}function Md(){if(ud===0){if(!(q&536870912)||F){var e=dt;dt<<=1,!(dt&3932160)&&(dt=262144),ud=e}else ud=536870912}return e=Fo.current,e!==null&&(e.flags|=32),ud}function Nd(e,t){if(t!=null){var n=e.stateNode,r=n.ref;r===null&&(r=n.ref=Pp(Ci(e.memoizedProps,n))),Dd===null&&(Dd=[]),Dd.push(t.bind(null,r))}}function Pd(e,t,n){(e===G&&(J===2||J===9)||e.cancelPendingCommit!==null)&&(Vd(e,0),Rd(e,q,ud,!1)),bt(e,n),(!(W&2)||e!==G)&&(e===G&&(!(W&2)&&(cd|=n),Y===4&&Rd(e,q,ud,!1)),Ef(e))}function Fd(e,t,n){if(W&6)throw Error(i(327));var r=!n&&!(t&127)&&(t&e.expiredLanes)===0||ht(e,t),a=r?Yd(e,t):qd(e,t,!0),o=r;do{if(a===0){id&&!r&&Rd(e,t,0,!1);break}if(n=e.current.alternate,o&&!Ld(n)){a=qd(e,t,!1),o=!1;continue}if(a===2){if(o=t,e.errorRecoveryDisabledLanes&o)var s=0;else s=e.pendingLanes&-536870913,s=s===0?s&536870912?536870912:0:s;if(s!==0){t=s;a:{var c=e;a=fd;var l=c.current.memoizedState.isDehydrated;if(l&&(Vd(c,s).flags|=256),s=qd(c,s,!1),s!==2&&s!==6){if(ad&&!l){c.errorRecoveryDisabledLanes|=o,cd|=o,a=4;break a}o=pd,pd=a,o!==null&&(pd===null?pd=o:pd.push.apply(pd,o))}a=s}if(o=!1,a!==2)continue}}if(a===1){Vd(e,0),Rd(e,t,0,!0);break}a:{switch(r=e,o=a,o){case 0:case 1:throw Error(i(345));case 4:if((t&4194048)!==t&&(t&62914560)!==t)break;case 6:Rd(r,t,ud,!rd);break a;case 2:pd=null;break;case 3:case 5:break;default:throw Error(i(329))}if((t&62914560)===t&&(a=hd+300-Je(),10<a)){if(Rd(r,t,ud,!rd),mt(r,0,!0)!==0)break a;Sd=t,r.timeoutHandle=gp(Id.bind(null,r,n,pd,vd,md,t,ud,cd,dd,rd,o,`Throttled`,-0,0),a);break a}Id(r,n,pd,vd,md,t,ud,cd,dd,rd,o,null,-0,0)}break}while(1);Ef(e)}function Id(e,t,n,r,i,a,o,s,c,l,u,d,f,p){e.timeoutHandle=-1;var m=t.subtreeFlags,h=(a&335544064)===a;if(d=null,(h||m&8192||(m&16785408)==16785408)&&(d={stylesheets:null,count:0,imgCount:0,imgBytes:0,suspenseyImages:[],waitingForImages:!0,waitingForViewTransition:!1,unsuspend:wn},Ul=null,Ju(t,a,d),h&&(m=d,h=e.containerInfo,h=(h.nodeType===9?h:h.ownerDocument).__reactViewTransition,h!=null&&(m.count++,m.waitingForViewTransition=!0,m=nh.bind(m),h.finished.then(m,m))),m=(a&62914560)===a?hd-Je():(a&4194048)===a?gd-Je():0,m=eh(d,m),m!==null)){Sd=a,e.cancelPendingCommit=m(nf.bind(null,e,t,a,n,r,i,o,s,c,l,u,d,null,f,p)),Rd(e,a,o,!l);return}nf(e,t,a,n,r,i,o,s,c,l,u,d)}function Ld(e){for(var t=e;;){var n=t.tag;if((n===0||n===11||n===15)&&t.flags&16384&&(n=t.updateQueue,n!==null&&(n=n.stores,n!==null)))for(var r=0;r<n.length;r++){var i=n[r],a=i.getSnapshot;i=i.value;try{if(!qr(a(),i))return!1}catch{return!1}}if(n=t.child,t.subtreeFlags&16384&&n!==null)n.return=t,t=n;else{if(t===e)break;for(;t.sibling===null;){if(t.return===null||t.return===e)return!0;t=t.return}t.sibling.return=t.return,t=t.sibling}}return!0}function Rd(e,t,n,r){t=gt(e,t),t&=~ld,t&=~cd,e.suspendedLanes|=t,e.pingedLanes&=~t,r&&(e.warmLanes|=t),r=e.expirationTimes;for(var i=t;0<i;){var a=31-ot(i),o=1<<a;r[a]=-1,i&=~o}n!==0&&St(e,n,t)}function zd(){return W&6?!0:(Df(0,!1),!1)}function Bd(){if(K!==null){if(J===0)var e=K.return;else e=K,Sa=xa=null,cs(e),lo=null,uo=0,e=K;for(;e!==null;)Cl(e.alternate,e),e=e.return;K=null}}function Vd(e,t){var n=e.timeoutHandle;return n!==-1&&(e.timeoutHandle=-1,_p(n)),n=e.cancelPendingCommit,n!==null&&(e.cancelPendingCommit=null,n()),Sd=0,Bd(),G=e,K=n=Bi(e.current,null),q=t,J=0,nd=null,rd=!1,id=ht(e,t),ad=!1,dd=ud=ld=cd=sd=Y=0,pd=fd=null,md=!1,od=gt(e,t),Ai(),n}function Hd(e,t){L=null,D.H=_c,t===$a||t===to?(t=so(),J=3):t===eo?(t=so(),J=4):J=t===Fc?8:typeof t==`object`&&t&&typeof t.then==`function`?6:1,nd=t,K===null&&(Y=1,kc(e,Ji(t,e.current)))}function Ud(){var e=Fo.current;return e===null?!0:(q&4194048)===q?Io===null:(q&62914560)===q||q&536870912?e===Io:!1}function Wd(){var e=D.H;return D.H=_c,e===null?_c:e}function Gd(){var e=D.A;return D.A=ed,e}function Kd(){Y=4,rd||(q&4194048)!==q&&Fo.current!==null||(id=!0),!(sd&134217727)&&!(cd&134217727)||G===null||Rd(G,q,ud,!1)}function qd(e,t,n){var r=W;W|=2;var i=Wd(),a=Gd();(G!==e||q!==t)&&(vd=null,Vd(e,t)),t=!1;var o=Y;a:do try{if(J!==0&&K!==null){var s=K,c=nd;switch(J){case 8:Bd(),o=6;break a;case 3:case 2:case 9:case 6:Fo.current===null&&(t=!0);var l=J;if(J=0,nd=null,$d(e,s,c,l),n&&id){o=0;break a}break;default:l=J,J=0,nd=null,$d(e,s,c,l)}}Jd(),o=Y;break}catch(t){Hd(e,t)}while(1);return t&&e.shellSuspendCounter++,Sa=xa=null,W=r,D.H=i,D.A=a,K===null&&(G=null,q=0,Ai()),o}function Jd(){for(;K!==null;)Zd(K)}function Yd(e,t){var n=W;W|=2;var r=Wd(),a=Gd();G!==e||q!==t?(vd=null,_d=Je()+500,Vd(e,t)):id=ht(e,t);a:do try{if(J!==0&&K!==null){t=K;var o=nd;b:switch(J){case 1:J=0,nd=null,$d(e,t,o,1);break;case 2:case 9:if(ro(o)){J=0,nd=null,Qd(t);break}t=function(){J!==2&&J!==9||G!==e||(J=7),Ef(e)},o.then(t,t);break a;case 3:J=7;break a;case 4:J=5;break a;case 7:ro(o)?(J=0,nd=null,Qd(t)):(J=0,nd=null,$d(e,t,o,7));break;case 5:var s=null;switch(K.tag){case 26:s=K.memoizedState;case 5:case 27:var c=K;if(s?Ym(s):c.stateNode.complete){J=0,nd=null;var l=c.sibling;if(l!==null)K=l;else{var u=c.return;u===null?K=null:(K=u,ef(u))}break b}}J=0,nd=null,$d(e,t,o,5);break;case 6:J=0,nd=null,$d(e,t,o,6);break;case 8:Bd(),Y=6;break a;default:throw Error(i(462))}}Xd();break}catch(t){Hd(e,t)}while(1);return Sa=xa=null,D.H=r,D.A=a,W=n,K===null?(G=null,q=0,Ai(),Y):0}function Xd(){for(;K!==null&&!Ke();)Zd(K)}function Zd(e){var t=hl(e.alternate,e,od);e.memoizedProps=e.pendingProps,t===null?ef(e):K=t}function Qd(e){var t=e,n=t.alternate;switch(t.tag){case 15:case 0:t=Yc(n,t,t.pendingProps,t.type,void 0,q);break;case 11:t=Yc(n,t,t.pendingProps,t.type.render,t.ref,q);break;case 5:cs(t);var r=t;r===la&&(F?(ha(r),r.tag===5&&r.stateNode!=null&&(P=r.stateNode)):(ha(r),F=!0));default:Cl(n,t),t=K=Vi(t,od),t=hl(n,t,od)}e.memoizedProps=e.pendingProps,t===null?ef(e):K=t}function $d(e,t,n,r){Sa=xa=null,cs(t),lo=null,uo=0;var i=t.return;try{if(Pc(e,i,t,n,q)){Y=1,kc(e,Ji(n,e.current)),K=null;return}}catch(t){if(i!==null)throw K=i,t;Y=1,kc(e,Ji(n,e.current)),K=null;return}t.flags&32768?(F||r===1?e=!0:id||q&536870912?e=!1:(rd=e=!0,(r===2||r===9||r===3||r===6)&&(r=Fo.current,r!==null&&r.tag===13&&(r.flags|=16384))),tf(t,e)):ef(t)}function ef(e){var t=e;do{if(t.flags&32768){tf(t,rd);return}e=t.return;var n=xl(t.alternate,t,od);if(n!==null){K=n;return}if(t=t.sibling,t!==null){K=t;return}K=t=e}while(t!==null);Y===0&&(Y=5)}function tf(e,t){do{var n=Sl(e.alternate,e);if(n!==null){n.flags&=32767,K=n;return}if(n=e.return,n!==null&&(n.flags|=32768,n.subtreeFlags=0,n.deletions=null),!t&&(e=e.sibling,e!==null)){K=e;return}K=e=n}while(e!==null);Y=6,K=null}function nf(e,t,n,r,a,o,s,c,l,u,d,f){e.cancelPendingCommit=null;do df();while(X!==0);if(W&6)throw Error(i(327));if(t!==null){if(t===e.current)throw Error(i(177));e===G&&(K=G=null,q=0),xd=t,bd=e,Sd=n,wd=a,Td=r,rf(e,t,n,s,c,l,f)}}function rf(e,t,n,r,i,a,o){var s=t.lanes|t.childLanes;if(Cd=s,s|=ki,xt(e,n,s,r,i,a),Dd=null,(n&335544064)===n?(Od=Ba(e),r=10262):(Od=null,r=10256),(t.subtreeFlags&r)!==0||(t.flags&r)!==0?(e.callbackNode=null,e.callbackPriority=0,yf(Qe,function(){return ff(),null})):(e.callbackNode=null,e.callbackPriority=0),Hl=!1,r=!!(t.flags&13878),t.subtreeFlags&13878||r){r=D.T,D.T=null,i=O.p,O.p=2,a=W,W|=4;try{gu(e,t,n)}finally{W=a,O.p=i,D.T=r}}X=1,Hl?Ed=Mp(o,e.containerInfo,Od,sf,cf,of,lf,ff,af,null,null):(sf(),cf(),lf())}function af(e){if(X!==0){var t=bd.onRecoverableError;t(e,{componentStack:null})}}function of(){X===3&&(X=0,Iu(xd,bd),X=4)}function sf(){if(X===1){X=0;var e=bd,t=xd,n=Sd,r=!!(t.flags&13878);if(t.subtreeFlags&13878||r){r=D.T,D.T=null;var i=O.p;O.p=2;var a=W;W|=4;try{pu=mu=!1,Mu(t,e,n),n=cp;var o=$r(e.containerInfo),s=n.focusedElem,c=n.selectionRange;if(o!==s&&s&&s.ownerDocument&&Qr(s.ownerDocument.documentElement,s)){if(c!==null&&ei(s)){var l=c.start,u=c.end;if(u===void 0&&(u=l),`selectionStart`in s)s.selectionStart=l,s.selectionEnd=Math.min(u,s.value.length);else{var d=s.ownerDocument||document,f=d&&d.defaultView||window;if(f.getSelection){var p=f.getSelection(),m=s.textContent.length,h=Math.min(c.start,m),g=c.end===void 0?h:Math.min(c.end,m);!p.extend&&h>g&&(o=g,g=h,h=o);var _=Zr(s,h),v=Zr(s,g);if(_&&v&&(p.rangeCount!==1||p.anchorNode!==_.node||p.anchorOffset!==_.offset||p.focusNode!==v.node||p.focusOffset!==v.offset)){var y=d.createRange();y.setStart(_.node,_.offset),p.removeAllRanges(),h>g?(p.addRange(y),p.extend(v.node,v.offset)):(y.setEnd(v.node,v.offset),p.addRange(y))}}}}for(d=[],p=s;p=p.parentNode;)p.nodeType===1&&d.push({element:p,left:p.scrollLeft,top:p.scrollTop});for(typeof s.focus==`function`&&s.focus(),s=0;s<d.length;s++){var b=d[s];b.element.scrollLeft=b.left,b.element.scrollTop=b.top}}gh=!!sp,cp=sp=null}finally{W=a,O.p=i,D.T=r}}e.current=t,X=2}}function cf(){if(X===2){X=0;var e=bd,t=xd,n=!!(t.flags&8772);if(t.subtreeFlags&8772||n){n=D.T,D.T=null;var r=O.p;O.p=2;var i=W;W|=4;try{vu(e,t.alternate,t)}finally{W=i,O.p=r,D.T=n}}X=3}}function lf(){if(X===4||X===3){X=0;var e=Ed;Ed=null,qe();var t=bd,n=xd,r=Sd,i=Td,a=(r&335544064)===r?10262:10256;if((n.subtreeFlags&a)!==0||(n.flags&a)!==0?X=5:(X=0,xd=bd=null,uf(t,t.pendingLanes)),a=t.pendingLanes,a===0&&(yd=null),Et(r),n=n.stateNode,it&&typeof it.onCommitFiberRoot==`function`)try{it.onCommitFiberRoot(rt,n,void 0,(n.current.flags&128)==128)}catch{}if(i!==null){n=D.T,a=O.p,O.p=2,D.T=null;try{for(var o=t.onRecoverableError,s=0;s<i.length;s++){var c=i[s];o(c.value,{componentStack:c.stack})}}finally{D.T=n,O.p=a}}if(i=Dd,o=Od,Od=null,i!==null&&(Dd=null,o===null&&(o=[]),e!==null))for(c=0;c<i.length;c++)n=(0,i[c])(o),n!==void 0&&e.finished.finally(n);Sd&3&&df(),Ef(t),a=t.pendingLanes,r&261930&&a&42?t===Ad?kd++:(kd=0,Ad=t):(kd=0,Ad=null),Df(0,!1)}}function uf(e,t){(e.pooledCacheLanes&=t)===0&&(t=e.pooledCache,t!=null&&(e.pooledCache=null,La(t)))}function df(){return Ed!==null&&(Ed.skipTransition(),Ed=null),sf(),cf(),lf(),ff()}function ff(){if(X!==5)return!1;var e=bd,t=Cd;Cd=0;var n=Et(Sd),r=D.T,a=O.p;try{O.p=32>n?32:n,D.T=null,n=wd,wd=null;var o=bd,s=Sd;if(X=0,xd=bd=null,Sd=0,W&6)throw Error(i(331));var c=W;if(W|=4,Zu(o.current),Uu(o,o.current,s,n),W=c,Df(0,!1),it&&typeof it.onPostCommitFiberRoot==`function`)try{it.onPostCommitFiberRoot(rt,o)}catch{}return!0}finally{O.p=a,D.T=r,uf(e,t)}}function pf(e,t,n){t=Ji(n,t),t=jc(e.stateNode,t,2),e=So(e,t,2),e!==null&&(bt(e,2),Ef(e))}function Z(e,t,n){if(e.tag===3)pf(e,e,n);else for(;t!==null;){if(t.tag===3){pf(t,e,n);break}if(t.tag===1){var r=t.stateNode;if(typeof t.type.getDerivedStateFromError==`function`||typeof r.componentDidCatch==`function`&&(yd===null||!yd.has(r))){e=Ji(n,e),n=Mc(2),r=So(t,n,2),r!==null&&(Nc(n,r,t,e),bt(r,2),Ef(r));break}}t=t.return}}function mf(e,t,n){var r=e.pingCache;if(r===null){r=e.pingCache=new td;var i=new Set;r.set(t,i)}else i=r.get(t),i===void 0&&(i=new Set,r.set(t,i));i.has(n)||(ad=!0,i.add(n),e=hf.bind(null,e,t,n),t.then(e,e))}function hf(e,t,n){var r=e.pingCache;r!==null&&r.delete(t),e.pingedLanes|=e.suspendedLanes&n,e.warmLanes&=~n,G===e&&(q&n)===n&&(Y===4||Y===3&&(q&62914560)===q&&300>Je()-hd?W&2?ld|=n:Vd(e,0):ld|=n,dd===q&&(dd=0)),Ef(e)}function gf(e,t){t===0&&(t=vt()),e=Ni(e,t),e!==null&&(bt(e,t),Ef(e))}function _f(e){var t=e.memoizedState,n=0;t!==null&&(n=t.retryLane),gf(e,n)}function vf(e,t){var n=0;switch(e.tag){case 31:case 13:var r=e.stateNode,a=e.memoizedState;a!==null&&(n=a.retryLane);break;case 19:r=e.stateNode;break;case 22:r=e.stateNode._retryCache;break;default:throw Error(i(314))}r!==null&&r.delete(t),gf(e,n)}function yf(e,t){return Ge(e,t)}var bf=null,xf=null,Sf=!1,Cf=!1,wf=!1,Tf=0;function Ef(e){e!==xf&&e.next===null&&(xf===null?bf=xf=e:xf=xf.next=e),Cf=!0,Sf||(Sf=!0,Nf())}function Df(e,t){if(!wf&&Cf){wf=!0;do for(var n=!1,r=bf;r!==null;){if(!t){if(e!==0){var i=r.pendingLanes;if(i===0)var a=0;else{var o=r.suspendedLanes,s=r.pingedLanes;a=(1<<31-ot(42|e)+1)-1,a&=i&~(o&~s),a=a&201326741?a&201326741|1:a?a|2:0}a!==0&&(n=!0,Mf(r,a))}else a=q,a=mt(r,r===G?a:0,r.cancelPendingCommit!==null||r.timeoutHandle!==-1),!(a&3)||ht(r,a)||(n=!0,Mf(r,a))}r=r.next}while(n);wf=!1}}function Of(){kf()}function kf(){Cf=Sf=!1;var e=0;Tf!==0&&hp()&&(e=Tf);for(var t=Je(),n=null,r=bf;r!==null;){var i=r.next,a=Af(r,t);a===0?(r.next=null,n===null?bf=i:n.next=i,i===null&&(xf=n)):(n=r,(e!==0||a&3)&&(Cf=!0)),r=i}X!==0&&X!==5||Df(e,!1),Tf!==0&&(Tf=0)}function Af(e,t){for(var n=e.suspendedLanes,r=e.pingedLanes,i=e.expirationTimes,a=e.pendingLanes&-62914561;0<a;){var o=31-ot(a),s=1<<o,c=i[o];c===-1?((s&n)===0||(s&r)!==0)&&(i[o]=_t(s,t)):c<=t&&(e.expiredLanes|=s),a&=~s}if(t=G,n=q,n=mt(e,e===t?n:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r=e.callbackNode,n===0||e===t&&(J===2||J===9)||e.cancelPendingCommit!==null)return r!==null&&r!==null&&A(r),e.callbackNode=null,e.callbackPriority=0;if(!(n&3)||ht(e,n)){if(t=n&-n,t===e.callbackPriority)return t;switch(r!==null&&A(r),Et(n)){case 2:case 8:n=Ze;break;case 32:n=Qe;break;case 268435456:n=et;break;default:n=Qe}return r=jf.bind(null,e),n=Ge(n,r),e.callbackPriority=t,e.callbackNode=n,t}return r!==null&&r!==null&&A(r),e.callbackPriority=2,e.callbackNode=null,2}function jf(e,t){if(X!==0&&X!==5)return e.callbackNode=null,e.callbackPriority=0,null;var n=e.callbackNode;if(df()&&e.callbackNode!==n)return null;var r=q;return r=mt(e,e===G?r:0,e.cancelPendingCommit!==null||e.timeoutHandle!==-1),r===0?null:(Fd(e,r,t),Af(e,Je()),e.callbackNode!=null&&e.callbackNode===n?jf.bind(null,e):null)}function Mf(e,t){if(df())return null;Fd(e,t,!0)}function Nf(){bp(function(){W&6?Ge(Xe,Of):kf()})}function Pf(){if(Tf===0){var e=Ua;e===0&&(e=ut,ut<<=1,!(ut&261888)&&(ut=256)),Tf=e}return Tf}function Ff(e){return e==null||typeof e==`symbol`||typeof e==`boolean`?null:typeof e==`function`?e:Cn(e)}function If(e,t,n,r,i){if(t===`submit`&&n&&n.stateNode===i){var a=Ff((i[jt]||null).action),o=r.submitter;o&&(t=(t=o[jt]||null)?Ff(t.formAction):o.getAttribute(`formAction`),t!==null&&(a=t,o=null));var s=new Gn(`action`,`action`,null,r,i);e.push({event:s,listeners:[{instance:null,listener:function(){if(r.defaultPrevented){if(Tf!==0){var e=new FormData(i,o);rc(n,{pending:!0,data:e,method:i.method,action:a},null,e)}}else typeof a==`function`&&(s.preventDefault(),e=new FormData(i,o),rc(n,{pending:!0,data:e,method:i.method,action:a},a,e))},currentTarget:i}]})}}for(var Lf=0;Lf<bi.length;Lf++){var Rf=bi[Lf];xi(Rf.toLowerCase(),`on`+(Rf[0].toUpperCase()+Rf.slice(1)))}xi(fi,`onAnimationEnd`),xi(pi,`onAnimationIteration`),xi(mi,`onAnimationStart`),xi(`dblclick`,`onDoubleClick`),xi(`focusin`,`onFocus`),xi(`focusout`,`onBlur`),xi(hi,`onTransitionRun`),xi(gi,`onTransitionStart`),xi(_i,`onTransitionCancel`),xi(vi,`onTransitionEnd`),Jt(`onMouseEnter`,[`mouseout`,`mouseover`]),Jt(`onMouseLeave`,[`mouseout`,`mouseover`]),Jt(`onPointerEnter`,[`pointerout`,`pointerover`]),Jt(`onPointerLeave`,[`pointerout`,`pointerover`]),qt(`onChange`,`change click focusin focusout input keydown keyup selectionchange`.split(` `)),qt(`onSelect`,`focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange`.split(` `)),qt(`onBeforeInput`,[`compositionend`,`keypress`,`textInput`,`paste`]),qt(`onCompositionEnd`,`compositionend focusout keydown keypress keyup mousedown`.split(` `)),qt(`onCompositionStart`,`compositionstart focusout keydown keypress keyup mousedown`.split(` `)),qt(`onCompositionUpdate`,`compositionupdate focusout keydown keypress keyup mousedown`.split(` `));var zf=`abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting`.split(` `),Bf=new Set(`beforetoggle cancel close invalid load scroll scrollend toggle`.split(` `).concat(zf));function Vf(e,t){t=!!(t&4);for(var n=0;n<e.length;n++){var r=e[n],i=r.event;r=r.listeners;a:{var a=void 0;if(t)for(var o=r.length-1;0<=o;o--){var s=r[o],c=s.instance,l=s.currentTarget;if(s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){Ei(e)}i.currentTarget=null,a=c}else for(o=0;o<r.length;o++){if(s=r[o],c=s.instance,l=s.currentTarget,s=s.listener,c!==a&&i.isPropagationStopped())break a;a=s,i.currentTarget=l;try{a(i)}catch(e){Ei(e)}i.currentTarget=null,a=c}}}}function Q(e,t){var n=t[Nt];n===void 0&&(n=t[Nt]=new Set);var r=e+`__bubble`;n.has(r)||(Gf(t,e,2,!1),n.add(r))}function Hf(e,t,n){var r=0;t&&(r|=4),Gf(n,e,r,t)}var Uf=`_reactListening`+Math.random().toString(36).slice(2);function Wf(e){if(!e[Uf]){e[Uf]=!0,Gt.forEach(function(t){t!==`selectionchange`&&(Bf.has(t)||Hf(t,!1,e),Hf(t,!0,e))});var t=e.nodeType===9?e:e.ownerDocument;t===null||t[Uf]||(t[Uf]=!0,Hf(`selectionchange`,!1,t))}}function Gf(e,t,n,r){switch(Ch(t)){case 2:var i=_h;break;case 8:i=vh;break;default:i=yh}n=i.bind(null,t,n,e),i=void 0,!Pn||t!==`touchstart`&&t!==`touchmove`&&t!==`wheel`||(i=!0),r?i===void 0?e.addEventListener(t,n,!0):e.addEventListener(t,n,{capture:!0,passive:i}):i===void 0?e.addEventListener(t,n,!1):e.addEventListener(t,n,{passive:i})}function Kf(e,t,n,r,i){var a=r;if(!(t&1)&&!(t&2)&&r!==null)a:for(;;){if(r===null)return;var s=r.tag;if(s===3||s===4){var c=r.stateNode.containerInfo;if(c===i)break;if(s===4)for(s=r.return;s!==null;){var l=s.tag;if((l===3||l===4)&&s.stateNode.containerInfo===i)return;s=s.return}for(;c!==null;){if(s=Bt(c),s===null)return;if(l=s.tag,l===5||l===6||l===26||l===27){r=a=s;continue a}c=c.parentNode}}r=r.return}jn(function(){var r=a,i=En(n),s=[];a:{var c=yi.get(e);if(c!==void 0){var l=Gn,u=e;switch(e){case`keypress`:if(Bn(n)===0)break a;case`keydown`:case`keyup`:l=lr;break;case`focusin`:u=`focus`,l=er;break;case`focusout`:u=`blur`,l=er;break;case`beforeblur`:case`afterblur`:l=er;break;case`click`:if(n.button===2)break a;case`auxclick`:case`dblclick`:case`mousedown`:case`mousemove`:case`mouseup`:case`mouseout`:case`mouseover`:case`contextmenu`:l=Qn;break;case`drag`:case`dragend`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`dragstart`:case`drop`:l=$n;break;case`touchcancel`:case`touchend`:case`touchmove`:case`touchstart`:l=fr;break;case fi:case pi:case mi:l=tr;break;case vi:l=pr;break;case`scroll`:case`scrollend`:l=qn;break;case`wheel`:l=mr;break;case`copy`:case`cut`:case`paste`:l=nr;break;case`gotpointercapture`:case`lostpointercapture`:case`pointercancel`:case`pointerdown`:case`pointermove`:case`pointerout`:case`pointerover`:case`pointerup`:l=ur;break;case`submit`:l=dr;break;case`toggle`:case`beforetoggle`:l=hr}var d=!!(t&4),f=!d&&(e===`scroll`||e===`scrollend`),p=d?c===null?null:c+`Capture`:c;d=[];for(var m=r,h;m!==null;){var g=m;if(h=g.stateNode,g=g.tag,g!==5&&g!==26&&g!==27||h===null||p===null||(g=Mn(m,p),g!=null&&d.push(qf(m,g,h))),f)break;m=m.return}0<d.length&&(c=new l(c,u,null,n,i),s.push({event:c,listeners:d}))}}if(!(t&7)){a:{if(l=e===`mouseover`||e===`pointerover`,c=e===`mouseout`||e===`pointerout`,l&&n!==Tn&&(u=n.relatedTarget||n.fromElement)&&(Bt(u)||u[Mt]))break a;(c||l)&&(u=i.window===i?i:(l=i.ownerDocument)?l.defaultView||l.parentWindow:window,c?(l=n.relatedTarget||n.toElement,c=r,l=l?Bt(l):null,l!==null&&(f=o(l),d=l.tag,l!==f||d!==5&&d!==27&&d!==6)&&(l=null)):(c=null,l=r),c!==l&&(d=Qn,g=`onMouseLeave`,p=`onMouseEnter`,m=`mouse`,(e===`pointerout`||e===`pointerover`)&&(d=ur,g=`onPointerLeave`,p=`onPointerEnter`,m=`pointer`),f=c==null?u:Ht(c),h=l==null?u:Ht(l),u=new d(g,m+`leave`,c,n,i),u.target=f,u.relatedTarget=h,g=null,Bt(i)===r&&(d=new d(p,m+`enter`,l,n,i),d.target=h,d.relatedTarget=f,g=d),f=g,d=c&&l?ne(c,l,Yf):null,c!==null&&Xf(s,u,c,d,!1),l!==null&&f!==null&&Xf(s,f,l,d,!0)))}a:{if(c=r?Ht(r):window,l=c.nodeName&&c.nodeName.toLowerCase(),l===`select`||l===`input`&&c.type===`file`)var _=Fr;else if(kr(c)){if(Ir)_=Gr;else{_=Ur;var v=Hr}}else l=c.nodeName,!l||l.toLowerCase()!==`input`||c.type!==`checkbox`&&c.type!==`radio`?r&&bn(r.elementType)&&(_=Fr):_=Wr;if(_&&=_(e,r)){Ar(s,_,n,i);break a}v&&v(e,c,r)}switch(v=r?Ht(r):window,e){case`focusin`:(kr(v)||v.contentEditable===`true`)&&(ni=v,ri=r,ii=null);break;case`focusout`:ii=ri=ni=null;break;case`mousedown`:ai=!0;break;case`contextmenu`:case`mouseup`:case`dragend`:ai=!1,oi(s,n,i);break;case`selectionchange`:if(ti)break;case`keydown`:case`keyup`:oi(s,n,i)}var y;if(_r)b:{switch(e){case`compositionstart`:var b=`onCompositionStart`;break b;case`compositionend`:b=`onCompositionEnd`;break b;case`compositionupdate`:b=`onCompositionUpdate`;break b}b=void 0}else Tr?Cr(e,n)&&(b=`onCompositionEnd`):e===`keydown`&&n.keyCode===229&&(b=`onCompositionStart`);b&&(br&&n.locale!==`ko`&&(Tr||b!==`onCompositionStart`?b===`onCompositionEnd`&&Tr&&(y=zn()):(In=i,Ln=`value`in In?In.value:In.textContent,Tr=!0)),v=Jf(r,b),0<v.length&&(b=new rr(b,e,null,n,i),s.push({event:b,listeners:v}),y?b.data=y:(y=wr(n),y!==null&&(b.data=y)))),(y=yr?Er(e,n):Dr(e,n))&&(b=Jf(r,`onBeforeInput`),0<b.length&&(v=new rr(`onBeforeInput`,`beforeinput`,null,n,i),s.push({event:v,listeners:b}),v.data=y)),If(s,e,r,n,i)}Vf(s,t)})}function qf(e,t,n){return{instance:e,listener:t,currentTarget:n}}function Jf(e,t){for(var n=t+`Capture`,r=[];e!==null;){var i=e,a=i.stateNode;if(i=i.tag,i!==5&&i!==26&&i!==27||a===null||(i=Mn(e,n),i!=null&&r.unshift(qf(e,i,a)),i=Mn(e,t),i!=null&&r.push(qf(e,i,a))),e.tag===3)return r;e=e.return}return[]}function Yf(e){if(e===null)return null;do e=e.return;while(e&&e.tag!==5&&e.tag!==27);return e||null}function Xf(e,t,n,r,i){for(var a=t._reactName,o=[];n!==null&&n!==r;){var s=n,c=s.alternate,l=s.stateNode;if(s=s.tag,c!==null&&c===r)break;s!==5&&s!==26&&s!==27||l===null||(c=l,i?(l=Mn(n,a),l!=null&&o.unshift(qf(n,l,c))):i||(l=Mn(n,a),l!=null&&o.push(qf(n,l,c)))),n=n.return}o.length!==0&&e.push({event:t,listeners:o})}var Zf=/\r\n?/g,Qf=/\u0000|\uFFFD/g;function $f(e){return(typeof e==`string`?e:``+e).replace(Zf,`
`).replace(Qf,``)}function ep(e,t){return t=$f(t),$f(e)===t}function $(e,t,n,r,a,o){switch(n){case`children`:if(typeof r==`string`)t===`body`||t===`textarea`&&r===``||gn(e,r);else if(typeof r==`number`||typeof r==`bigint`)t!==`body`&&gn(e,``+r);else return;break;case`className`:tn(e,`class`,r);break;case`tabIndex`:tn(e,`tabindex`,r);break;case`dir`:case`role`:case`viewBox`:case`width`:case`height`:tn(e,n,r);break;case`style`:yn(e,r,o);return;case`data`:if(t!==`object`){tn(e,`data`,r);break}case`src`:case`href`:if(r===``&&(t!==`a`||n!==`href`)){e.removeAttribute(n);break}if(r==null||typeof r==`function`||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=Cn(r),e.setAttribute(n,r);break;case`action`:case`formAction`:if(typeof r==`function`){e.setAttribute(n,`javascript:throw new Error('A React form was unexpectedly submitted. If you called form.submit() manually, consider using form.requestSubmit() instead. If you\\'re trying to use event.stopPropagation() in a submit event handler, consider also calling event.preventDefault().')`);break}if(typeof o==`function`&&(n===`formAction`?(t!==`input`&&$(e,t,`name`,a.name,a,null),$(e,t,`formEncType`,a.formEncType,a,null),$(e,t,`formMethod`,a.formMethod,a,null),$(e,t,`formTarget`,a.formTarget,a,null)):($(e,t,`encType`,a.encType,a,null),$(e,t,`method`,a.method,a,null),$(e,t,`target`,a.target,a,null))),r==null||typeof r==`symbol`||typeof r==`boolean`){e.removeAttribute(n);break}r=Cn(r),e.setAttribute(n,r);break;case`onClick`:r!=null&&(e.onclick=wn);return;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`multiple`:e.multiple=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`muted`:e.muted=r&&typeof r!=`function`&&typeof r!=`symbol`;break;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`defaultValue`:case`defaultChecked`:case`innerHTML`:case`ref`:break;case`autoFocus`:break;case`xlinkHref`:if(r==null||typeof r==`function`||typeof r==`boolean`||typeof r==`symbol`){e.removeAttribute(`xlink:href`);break}n=Cn(r),e.setAttributeNS(`http://www.w3.org/1999/xlink`,`xlink:href`,n);break;case`contentEditable`:case`spellCheck`:case`draggable`:case`value`:case`autoReverse`:case`externalResourcesRequired`:case`focusable`:case`preserveAlpha`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`inert`:case`allowFullScreen`:case`async`:case`autoPlay`:case`controls`:case`credentialless`:case`default`:case`defer`:case`disabled`:case`disablePictureInPicture`:case`disableRemotePlayback`:case`formNoValidate`:case`hidden`:case`loop`:case`noModule`:case`noValidate`:case`open`:case`playsInline`:case`readOnly`:case`required`:case`reversed`:case`scoped`:case`seamless`:case`itemScope`:r&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,``):e.removeAttribute(n);break;case`capture`:case`download`:!0===r?e.setAttribute(n,``):!1!==r&&r!=null&&typeof r!=`function`&&typeof r!=`symbol`?e.setAttribute(n,r):e.removeAttribute(n);break;case`cols`:case`rows`:case`size`:case`span`:r!=null&&typeof r!=`function`&&typeof r!=`symbol`&&!isNaN(r)&&1<=r?e.setAttribute(n,r):e.removeAttribute(n);break;case`rowSpan`:case`start`:r==null||typeof r==`function`||typeof r==`symbol`||isNaN(r)?e.removeAttribute(n):e.setAttribute(n,r);break;case`popover`:Q(`beforetoggle`,e),Q(`toggle`,e),en(e,`popover`,r);break;case`xlinkActuate`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:actuate`,r);break;case`xlinkArcrole`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:arcrole`,r);break;case`xlinkRole`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:role`,r);break;case`xlinkShow`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:show`,r);break;case`xlinkTitle`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:title`,r);break;case`xlinkType`:nn(e,`http://www.w3.org/1999/xlink`,`xlink:type`,r);break;case`xmlBase`:nn(e,`http://www.w3.org/XML/1998/namespace`,`xml:base`,r);break;case`xmlLang`:nn(e,`http://www.w3.org/XML/1998/namespace`,`xml:lang`,r);break;case`xmlSpace`:nn(e,`http://www.w3.org/XML/1998/namespace`,`xml:space`,r);break;case`is`:en(e,`is`,r);break;case`innerText`:case`textContent`:return;default:if(!(2<n.length)||n[0]!==`o`&&n[0]!==`O`||n[1]!==`n`&&n[1]!==`N`)n=xn.get(n)||n,en(e,n,r);else return}M=!0}function tp(e,t,n,r,a,o){switch(n){case`style`:yn(e,r,o);return;case`dangerouslySetInnerHTML`:if(r!=null){if(typeof r!=`object`||!(`__html`in r))throw Error(i(61));if(n=r.__html,n!=null){if(a.children!=null)throw Error(i(60));o?.__html!==n&&(e.innerHTML=n)}}break;case`children`:if(typeof r==`string`)gn(e,r);else if(typeof r==`number`||typeof r==`bigint`)gn(e,``+r);else return;break;case`onScroll`:r!=null&&Q(`scroll`,e);return;case`onScrollEnd`:r!=null&&Q(`scrollend`,e);return;case`onClick`:r!=null&&(e.onclick=wn);return;case`suppressContentEditableWarning`:case`suppressHydrationWarning`:case`innerHTML`:case`ref`:return;case`innerText`:case`textContent`:return;default:if(!Kt.hasOwnProperty(n))a:{if(n[0]===`o`&&n[1]===`n`&&(a=n.endsWith(`Capture`),o=n.slice(2,a?n.length-7:void 0),t=e[jt]||null,t=t==null?null:t[n],typeof t==`function`&&e.removeEventListener(o,t,a),typeof r==`function`)){typeof t!=`function`&&t!==null&&(n in e?e[n]=null:e.hasAttribute(n)&&e.removeAttribute(n)),e.addEventListener(o,r,a);break a}M=!0,n in e?e[n]=r:!0===r?e.setAttribute(n,``):en(e,n,r)}return}M=!0}function np(e,t,n){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`img`:Q(`error`,e),Q(`load`,e);var r=!1,a=!1,o;for(o in n)if(n.hasOwnProperty(o)){var s=n[o];if(s!=null)switch(o){case`src`:r=!0;break;case`srcSet`:a=!0;break;case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,o,s,n,null)}}a&&$(e,t,`srcSet`,n.srcSet,n,null),r&&$(e,t,`src`,n.src,n,null);return;case`input`:Q(`invalid`,e);var c=o=s=a=null,l=null,u=null;for(r in n)if(n.hasOwnProperty(r)){var d=n[r];if(d!=null)switch(r){case`name`:a=d;break;case`type`:s=d;break;case`checked`:l=d;break;case`defaultChecked`:u=d;break;case`value`:o=d;break;case`defaultValue`:c=d;break;case`children`:case`dangerouslySetInnerHTML`:if(d!=null)throw Error(i(137,t));break;default:$(e,t,r,d,n,null)}}dn(e,o,c,l,u,s,a,!1);return;case`select`:for(a in Q(`invalid`,e),r=s=o=null,n)if(n.hasOwnProperty(a)&&(c=n[a],c!=null))switch(a){case`value`:o=c;break;case`defaultValue`:s=c;break;case`multiple`:r=c;default:$(e,t,a,c,n,null)}t=o,n=s,e.multiple=!!r,t==null?n!=null&&pn(e,!!r,n,!0):pn(e,!!r,t,!1);return;case`textarea`:for(s in Q(`invalid`,e),o=a=r=null,n)if(n.hasOwnProperty(s)&&(c=n[s],c!=null))switch(s){case`value`:r=c;break;case`defaultValue`:a=c;break;case`children`:o=c;break;case`dangerouslySetInnerHTML`:if(c!=null)throw Error(i(91));break;default:$(e,t,s,c,n,null)}hn(e,r,a,o);return;case`option`:for(l in n)if(n.hasOwnProperty(l)&&(r=n[l],r!=null))switch(l){case`selected`:e.selected=r&&typeof r!=`function`&&typeof r!=`symbol`;break;default:$(e,t,l,r,n,null)}return;case`dialog`:Q(`beforetoggle`,e),Q(`toggle`,e),Q(`cancel`,e),Q(`close`,e);break;case`iframe`:case`object`:Q(`load`,e);break;case`video`:case`audio`:for(r=0;r<zf.length;r++)Q(zf[r],e);break;case`image`:Q(`error`,e),Q(`load`,e);break;case`details`:Q(`toggle`,e);break;case`embed`:case`source`:case`link`:Q(`error`,e),Q(`load`,e);case`area`:case`base`:case`br`:case`col`:case`hr`:case`keygen`:case`meta`:case`param`:case`track`:case`wbr`:case`menuitem`:for(u in n)if(n.hasOwnProperty(u)&&(r=n[u],r!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:throw Error(i(137,t));default:$(e,t,u,r,n,null)}return;default:if(bn(t)){for(d in n)n.hasOwnProperty(d)&&(r=n[d],r!==void 0&&tp(e,t,d,r,n,void 0));return}}for(c in n)n.hasOwnProperty(c)&&(r=n[c],r!=null&&$(e,t,c,r,n,null))}var rp={};function ip(e,t,n,r){switch(t){case`div`:case`span`:case`svg`:case`path`:case`a`:case`g`:case`p`:case`li`:break;case`input`:var a=null,o=null,s=null,c=null,l=null,u=null,d=null;for(m in n){var f=n[m];if(n.hasOwnProperty(m)&&f!=null)switch(m){case`checked`:break;case`value`:break;case`defaultValue`:l=f;default:r.hasOwnProperty(m)||$(e,t,m,null,r,f)}}for(var p in r){var m=r[p];if(f=n[p],r.hasOwnProperty(p)&&(m!=null||f!=null))switch(p){case`type`:m!==f&&(M=!0),o=m;break;case`name`:m!==f&&(M=!0),a=m;break;case`checked`:m!==f&&(M=!0),u=m;break;case`defaultChecked`:m!==f&&(M=!0),d=m;break;case`value`:m!==f&&(M=!0),s=m;break;case`defaultValue`:m!==f&&(M=!0),c=m;break;case`children`:case`dangerouslySetInnerHTML`:if(m!=null)throw Error(i(137,t));break;default:m!==f&&$(e,t,p,m,r,f)}}un(e,s,c,l,u,d,o,a);return;case`select`:for(o in m=s=c=p=null,n)if(l=n[o],n.hasOwnProperty(o)&&l!=null)switch(o){case`value`:break;case`multiple`:m=l;default:r.hasOwnProperty(o)||$(e,t,o,null,r,l)}for(a in r)if(o=r[a],l=n[a],r.hasOwnProperty(a)&&(o!=null||l!=null))switch(a){case`value`:o!==l&&(M=!0),p=o;break;case`defaultValue`:o!==l&&(M=!0),c=o;break;case`multiple`:o!==l&&(M=!0),s=o;default:o!==l&&$(e,t,a,o,r,l)}t=c,n=s,r=m,p==null?!!r!=!!n&&(t==null?pn(e,!!n,n?[]:``,!1):pn(e,!!n,t,!0)):pn(e,!!n,p,!1);return;case`textarea`:for(c in m=p=null,n)if(a=n[c],n.hasOwnProperty(c)&&a!=null&&!r.hasOwnProperty(c))switch(c){case`value`:break;case`children`:break;default:$(e,t,c,null,r,a)}for(s in r)if(a=r[s],o=n[s],r.hasOwnProperty(s)&&(a!=null||o!=null))switch(s){case`value`:a!==o&&(M=!0),p=a;break;case`defaultValue`:a!==o&&(M=!0),m=a;break;case`children`:break;case`dangerouslySetInnerHTML`:if(a!=null)throw Error(i(91));break;default:a!==o&&$(e,t,s,a,r,o)}mn(e,p,m);return;case`option`:for(var h in n)if(p=n[h],n.hasOwnProperty(h)&&p!=null&&!r.hasOwnProperty(h))switch(h){case`selected`:e.selected=!1;break;default:$(e,t,h,null,r,p)}for(l in r)if(p=r[l],m=n[l],r.hasOwnProperty(l)&&p!==m&&(p!=null||m!=null))switch(l){case`selected`:p!==m&&(M=!0),e.selected=p&&typeof p!=`function`&&typeof p!=`symbol`;break;default:$(e,t,l,p,r,m)}return;case`img`:case`link`:case`area`:case`base`:case`br`:case`col`:case`embed`:case`hr`:case`keygen`:case`meta`:case`param`:case`source`:case`track`:case`wbr`:case`menuitem`:for(var g in n)p=n[g],n.hasOwnProperty(g)&&p!=null&&!r.hasOwnProperty(g)&&$(e,t,g,null,r,p);for(u in r)if(p=r[u],m=n[u],r.hasOwnProperty(u)&&p!==m&&(p!=null||m!=null))switch(u){case`children`:case`dangerouslySetInnerHTML`:if(p!=null)throw Error(i(137,t));break;default:$(e,t,u,p,r,m)}return;default:if(bn(t)){for(var _ in n)p=n[_],n.hasOwnProperty(_)&&p!==void 0&&!r.hasOwnProperty(_)&&tp(e,t,_,void 0,r,p);for(d in r)p=r[d],m=n[d],!r.hasOwnProperty(d)||p===m||p===void 0&&m===void 0||tp(e,t,d,p,r,m);return}}for(var v in n)p=n[v],n.hasOwnProperty(v)&&p!=null&&!r.hasOwnProperty(v)&&$(e,t,v,null,r,p);for(f in r)p=r[f],m=n[f],!r.hasOwnProperty(f)||p===m||p==null&&m==null||$(e,t,f,p,r,m)}function ap(e){switch(e){case`css`:case`script`:case`font`:case`img`:case`image`:case`input`:case`link`:return!0;default:return!1}}function op(){if(typeof performance.getEntriesByType==`function`){for(var e=0,t=0,n=performance.getEntriesByType(`resource`),r=0;r<n.length;r++){var i=n[r],a=i.transferSize,o=i.initiatorType,s=i.duration;if(a&&s&&ap(o)){for(o=0,s=i.responseEnd,r+=1;r<n.length;r++){var c=n[r],l=c.startTime;if(l>s)break;var u=c.transferSize,d=c.initiatorType;u&&ap(d)&&(c=c.responseEnd,o+=u*(c<s?1:(s-l)/(c-l)))}if(--r,t+=8*(a+o)/(i.duration/1e3),e++,10<e)break}}if(0<e)return t/e/1e6}return navigator.connection&&(e=navigator.connection.downlink,typeof e==`number`)?e:5}var sp=null,cp=null;function lp(e){return e.nodeType===9?e:e.ownerDocument}function up(e){switch(e){case`http://www.w3.org/2000/svg`:return 1;case`http://www.w3.org/1998/Math/MathML`:return 2;default:return 0}}function dp(e,t){if(e===0)switch(t){case`svg`:return 1;case`math`:return 2;default:return 0}return e===1&&t===`foreignObject`?0:e}function fp(e,t,n,r){return n=lp(n).createElement(e),n[At]=r,n[jt]=t,np(n,e,t),j(n),n}function pp(e,t){return e===`textarea`||e===`noscript`||typeof t.children==`string`||typeof t.children==`number`||typeof t.children==`bigint`||typeof t.dangerouslySetInnerHTML==`object`&&t.dangerouslySetInnerHTML!==null&&t.dangerouslySetInnerHTML.__html!=null}var mp=null;function hp(){var e=window.event;return e&&e.type===`popstate`?e!==mp&&(mp=e,!0):(mp=null,!1)}var gp=typeof setTimeout==`function`?setTimeout:void 0,_p=typeof clearTimeout==`function`?clearTimeout:void 0,vp=typeof Promise==`function`?Promise:void 0,yp=typeof requestAnimationFrame==`function`?requestAnimationFrame:gp,bp=typeof queueMicrotask==`function`?queueMicrotask:vp===void 0?gp:function(e){return vp.resolve(null).then(e).catch(xp)};function xp(e){setTimeout(function(){throw e})}function Sp(e){return e===`head`}function Cp(e,t){var n=t,r=0;do{var i=n.nextSibling;if(e.removeChild(n),i&&i.nodeType===8){if(n=i.data,n===`/$`||n===`/&`){if(r===0){e.removeChild(i),Hh(t);return}r--}else if(n===`$`||n===`$?`||n===`$~`||n===`$!`||n===`&`)r++;else if(n===`html`)_m(e.ownerDocument.documentElement);else if(n===`head`){n=e.ownerDocument.head,_m(n);for(var a=n.firstChild;a;){var o=a.nextSibling,s=a.nodeName;a[Lt]||s===`SCRIPT`||s===`STYLE`||s===`LINK`&&a.rel.toLowerCase()===`stylesheet`||n.removeChild(a),a=o}}else n===`body`&&_m(e.ownerDocument.body)}n=i}while(n);Hh(t)}function wp(e,t){var n=e;e=0;do{var r=n.nextSibling;if(n.nodeType===1?t?(n._stashedDisplay=n.style.display,n.style.display=`none`):(n.style.display=n._stashedDisplay||``,n.getAttribute(`style`)===``&&n.removeAttribute(`style`)):n.nodeType===3&&(t?(n._stashedText=n.nodeValue,n.nodeValue=``):n.nodeValue=n._stashedText||``),r&&r.nodeType===8){if(n=r.data,n===`/$`){if(e===0)break;e--}else n!==`$`&&n!==`$?`&&n!==`$~`&&n!==`$!`||e++}n=r}while(n)}function Tp(e,t,n){if(t=CSS.escape(t)===t?t:`r-`+btoa(t).replace(/=/g,``),e.style.viewTransitionName=t,n!=null&&(e.style.viewTransitionClass=n),n=getComputedStyle(e),n.display===`inline`){if(t=e.getClientRects(),t.length===1)var r=1;else for(var i=r=0;i<t.length;i++){var a=t[i];0<a.width&&0<a.height&&r++}r===1&&(e=e.style,e.display=t.length===1?`inline-block`:`block`,e.marginTop=`-`+n.paddingTop,e.marginBottom=`-`+n.paddingBottom)}}function Ep(e,t){e=e.style,t=t.style;var n=t==null?null:t.hasOwnProperty(`viewTransitionName`)?t.viewTransitionName:t.hasOwnProperty(`view-transition-name`)?t[`view-transition-name`]:null;e.viewTransitionName=n==null||typeof n==`boolean`?``:(``+n).trim(),n=t==null?null:t.hasOwnProperty(`viewTransitionClass`)?t.viewTransitionClass:t.hasOwnProperty(`view-transition-class`)?t[`view-transition-class`]:null,e.viewTransitionClass=n==null||typeof n==`boolean`?``:(``+n).trim(),e.display===`inline-block`&&(t==null?e.display=e.margin=``:(n=t.display,e.display=n==null||typeof n==`boolean`?``:n,n=t.margin,n==null?(n=t.hasOwnProperty(`marginTop`)?t.marginTop:t[`margin-top`],e.marginTop=n==null||typeof n==`boolean`?``:n,t=t.hasOwnProperty(`marginBottom`)?t.marginBottom:t[`margin-bottom`],e.marginBottom=t==null||typeof t==`boolean`?``:t):e.margin=n))}function Dp(e,t,n){return n=n.ownerDocument.defaultView,{rect:e,abs:t.position===`absolute`||t.position===`fixed`,clip:t.clipPath!==`none`||t.overflow!==`visible`||t.filter!==`none`||t.mask!==`none`||t.mask!==`none`||t.borderRadius!==`0px`,view:0<=e.bottom&&0<=e.right&&e.top<=n.innerHeight&&e.left<=n.innerWidth}}function Op(e){return Dp(e.getBoundingClientRect(),getComputedStyle(e),e)}function kp(e){var t=e.getBoundingClientRect();t=new DOMRect(t.x+2e4,t.y+2e4,t.width,t.height);var n=getComputedStyle(e);return Dp(t,n,e)}function Ap(e){return e.documentElement.clientHeight}function jp(e){this.addEventListener(`load`,e),this.addEventListener(`error`,e)}function Mp(e,t,n,r,i,a,o,s,c){var l=t.nodeType===9?t:t.ownerDocument;try{var u=l.startViewTransition({update:function(){var t=l.defaultView,n=t.navigation&&t.navigation.transition,o=l.fonts.status;r();var s=[];if(o===`loaded`&&(Ap(l),l.fonts.status===`loading`&&s.push(l.fonts.ready)),o=s.length,e!==null)for(var c=e.suspenseyImages,u=0,d=0;d<c.length;d++){var f=c[d];if(!f.complete){var p=f.getBoundingClientRect();if(0<p.bottom&&0<p.right&&p.top<t.innerHeight&&p.left<t.innerWidth){if(u+=Xm(f),u>$m){s.length=o;break}f=new Promise(jp.bind(f)),s.push(f)}}}if(0<s.length)return t=Promise.race([Promise.all(s),new Promise(function(e){return setTimeout(e,500)})]).then(i,i),(n?Promise.allSettled([n.finished,t]):t).then(a,a);if(i(),n)return n.finished.then(a,a);a()},types:n});l.__reactViewTransition=u;var d=[];return u.ready.then(function(){for(var e=l.documentElement.getAnimations({subtree:!0}),t=0;t<e.length;t++){var n=e[t],r=n.effect,i=r.pseudoElement;if(i!=null&&i.startsWith(`::view-transition`)){d.push(n),n=r.getKeyframes();for(var a=i=void 0,s=!0,c=0;c<n.length;c++){var u=n[c],f=u.width;if(i===void 0)i=f;else if(i!==f){s=!1;break}if(f=u.height,a===void 0)a=f;else if(a!==f){s=!1;break}delete u.width,delete u.height,u.transform===`none`&&delete u.transform}s&&i!==void 0&&a!==void 0&&(r.setKeyframes(n),s=getComputedStyle(r.target,r.pseudoElement),s.width!==i||s.height!==a)&&(s=n[0],s.width=i,s.height=a,s=n[n.length-1],s.width=i,s.height=a,r.setKeyframes(n))}}o()},function(e){l.__reactViewTransition===u&&(l.__reactViewTransition=null);try{if(typeof e==`object`&&e)switch(e.name){case`InvalidStateError`:(e.message===`View transition was skipped because document visibility state is hidden.`||e.message===`Skipping view transition because document visibility state has become hidden.`||e.message===`Skipping view transition because viewport size changed.`||e.message===`Transition was aborted because of invalid state`)&&(e=null)}e!==null&&c(e)}finally{r(),i(),o()}}),u.finished.finally(function(){for(var e=0;e<d.length;e++)d[e].cancel();l.__reactViewTransition===u&&(l.__reactViewTransition=null),s()}),u}catch{return r(),i(),o(),null}}function Np(e,t){this._scope=document.documentElement,this._selector=`::view-transition-`+e+`(`+t+`)`}Np.prototype.animate=function(e,t){return t=typeof t==`number`?{duration:t}:w({},t),t.pseudoElement=this._selector,this._scope.animate(e,t)},Np.prototype.getAnimations=function(){for(var e=this._scope,t=this._selector,n=e.getAnimations({subtree:!0}),r=[],i=0;i<n.length;i++){var a=n[i].effect;a!==null&&a.target===e&&a.pseudoElement===t&&r.push(n[i])}return r},Np.prototype.getComputedStyle=function(){return getComputedStyle(this._scope,this._selector)};function Pp(e){return{name:e,group:new Np(`group`,e),imagePair:new Np(`image-pair`,e),old:new Np(`old`,e),new:new Np(`new`,e)}}function Fp(e){this._fragmentFiber=e,this._observers=this._eventListeners=null}Fp.prototype.addEventListener=function(e,t,n){var r=null,i=null;if(!(n!=null&&typeof n!=`boolean`&&(r=n.signal||null,r!==null&&r.aborted))){this._eventListeners===null&&(this._eventListeners=[]);var a=this._eventListeners;if(Bp(a,e,t,n)===-1){var o=this,s=t;n!=null&&typeof n!=`boolean`&&!0===n.once&&(s=function(r){o.removeEventListener(e,t,n),typeof t==`function`?t.call(this,r):t.handleEvent(r)}),r!==null&&(i=o.removeEventListener.bind(o,e,t,n),r.addEventListener(`abort`,i,{once:!0}),i=r.removeEventListener.bind(r,`abort`,i)),r=Rp(n),a.push({type:e,listener:t,optionsOrUseCapture:n,attachedListener:s,cleanup:i}),h(this._fragmentFiber.child,!1,Ip,e,s,r)}this._eventListeners=a}};function Ip(e,t,n,r){return b(e).addEventListener(t,n,r),!1}Fp.prototype.removeEventListener=function(e,t,n){var r=this._eventListeners;if(r!==null&&(t=Bp(r,e,t,n),t!==-1)){var i=r[t];n=i.attachedListener;var a=i.cleanup;i=Rp(i.optionsOrUseCapture),h(this._fragmentFiber.child,!1,Lp,e,n,i),r.splice(t,1),a!==null&&a()}};function Lp(e,t,n,r){return b(e).removeEventListener(t,n,r),!1}function Rp(e){return e!=null&&typeof e!=`boolean`&&(!0===e.once||e.signal instanceof AbortSignal)?{capture:e.capture,passive:e.passive}:e}function zp(e){return e==null?`c=0`:typeof e==`boolean`?`c=`+(e?`1`:`0`):`c=`+(e.capture?`1`:`0`)}function Bp(e,t,n,r){if(e.length===0)return-1;r=zp(r);for(var i=0;i<e.length;i++){var a=e[i];if(a.type===t&&a.listener===n&&zp(a.optionsOrUseCapture)===r)return i}return-1}Fp.prototype.dispatchEvent=function(e){var t=g(this._fragmentFiber);if(t===null)return!0;t=b(t);var n=this._eventListeners;if(n!==null&&0<n.length||!e.bubbles){var r=t.nodeType===9?t.createComment(``):document.createTextNode(``);if(n)for(var i=0;i<n.length;i++){var a=n[i];r.addEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture))}if(t.appendChild(r),e=r.dispatchEvent(e),n)for(i=0;i<n.length;i++)a=n[i],r.removeEventListener(a.type,a.attachedListener,Rp(a.optionsOrUseCapture));return t.removeChild(r),e}return t.dispatchEvent(e)},Fp.prototype.focus=function(e){h(this._fragmentFiber.child,!0,Vp,e,void 0,void 0)};function Vp(e,t){return e.tag!==6&&(e=b(e),pm(e,t))}Fp.prototype.focusLast=function(e){var t=[];h(this._fragmentFiber.child,!0,Hp,t,void 0,void 0);for(var n=t.length-1;0<=n&&!Vp(t[n],e);n--);};function Hp(e,t){return t.push(e),!1}Fp.prototype.blur=function(){var e=g(this._fragmentFiber);e!==null&&(e=b(e),e=lp(e).activeElement,e!==null&&h(this._fragmentFiber.child,!1,Up,e,void 0,void 0))};function Up(e,t){return e.tag!==6&&(e=b(e),e===t||e.contains(t)?(t.blur(),!0):!1)}Fp.prototype.observeUsing=function(e){this._observers===null&&(this._observers=new Set),this._observers.add(e),h(this._fragmentFiber.child,!1,Wp,e,void 0,void 0)};function Wp(e,t){return e.tag!==6&&(e=b(e),t.observe(e),!1)}Fp.prototype.unobserveUsing=function(e){var t=this._observers;if(t!==null&&t.has(e)){t.delete(e),h(this._fragmentFiber.child,!1,Gp,e,void 0,void 0);for(var n=t=0;n<Kp.length;n++){var r=Kp[n];r.fragmentInstance===this&&r.observer===e?e.unobserve(r.instance):Kp[t++]=r}Kp.length=t}};function Gp(e,t){return e.tag!==6&&(e=b(e),t.unobserve(e),!1)}var Kp=[],qp=!1;function Jp(e,t,n){Kp.push({fragmentInstance:e,observer:t,instance:n}),qp||(qp=!0,mm(function(){qp=!1;var e=Kp;Kp=[];for(var t=0;t<e.length;t++){var n=e[t];n.observer.unobserve(n.instance)}}))}Fp.prototype.getClientRects=function(){var e=[];return h(this._fragmentFiber.child,!1,Yp,e,void 0,void 0),e};function Yp(e,t){if(e.tag===6){e=e.stateNode;var n=e.ownerDocument.createRange();n.selectNodeContents(e),t.push.apply(t,n.getClientRects())}else e=b(e),t.push.apply(t,e.getClientRects());return!1}Fp.prototype.getRootNode=function(e){var t=g(this._fragmentFiber);return t===null?this:b(t).getRootNode(e)},Fp.prototype.compareDocumentPosition=function(e){var t=g(this._fragmentFiber);if(t===null)return Node.DOCUMENT_POSITION_DISCONNECTED;var n=[];h(this._fragmentFiber.child,!1,Hp,n,void 0,void 0);var r=b(t);if(n.length===0){if(n=r,_(this._fragmentFiber)){a:{for(t=this._fragmentFiber.return;t!==null;){if(t.tag===4){t=t.stateNode.containerInfo;break a}if(t.tag===3||t.tag===5||t.tag===27)break;t=t.return}t=null}t!=null&&(n=t)}t=this._fragmentFiber;var i=r=n.compareDocumentPosition(e);return n===e?i=Node.DOCUMENT_POSITION_CONTAINS:r&Node.DOCUMENT_POSITION_CONTAINED_BY&&(n=v(t)[1],n===null?i=Node.DOCUMENT_POSITION_PRECEDING:(e=b(n).compareDocumentPosition(e),i=e===0||e&Node.DOCUMENT_POSITION_FOLLOWING?Node.DOCUMENT_POSITION_FOLLOWING:Node.DOCUMENT_POSITION_PRECEDING)),i|=Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC}t=b(n[0]),i=b(n[n.length-1]);var a=_(this._fragmentFiber)?t.parentElement:r;if(a==null)return Node.DOCUMENT_POSITION_DISCONNECTED;r=a.compareDocumentPosition(t)&Node.DOCUMENT_POSITION_CONTAINED_BY,a=a.compareDocumentPosition(i)&Node.DOCUMENT_POSITION_CONTAINED_BY;var o=t.compareDocumentPosition(e),s=i.compareDocumentPosition(e),c=o&Node.DOCUMENT_POSITION_CONTAINED_BY||s&Node.DOCUMENT_POSITION_CONTAINED_BY;return s=r&&a&&o&Node.DOCUMENT_POSITION_FOLLOWING&&s&Node.DOCUMENT_POSITION_PRECEDING,t=r&&t===e||a&&i===e||c||s?Node.DOCUMENT_POSITION_CONTAINED_BY:!r&&t===e||!a&&i===e?Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC:o,t&Node.DOCUMENT_POSITION_DISCONNECTED||t&Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC||Xp(t,this._fragmentFiber,n[0],n[n.length-1],e)?t:Node.DOCUMENT_POSITION_IMPLEMENTATION_SPECIFIC};function Xp(e,t,n,r,i){var a=Bt(i);if(e&Node.DOCUMENT_POSITION_CONTAINED_BY){if(n=!!a)a:{for(;a!==null;){if(a.tag===7&&(a===t||a.alternate===t)){n=!0;break a}a=a.return}n=!1}return n}if(e&Node.DOCUMENT_POSITION_CONTAINS){if(a===null)return a=i.ownerDocument,i===a||i===a.documentElement||i===a.body;a:{for(a=t,t=g(t);a!==null;){if(!(a.tag!==5&&a.tag!==3&&a.tag!==27||a!==t&&a.alternate!==t)){a=!0;break a}a=a.return}a=!1}return a}return e&Node.DOCUMENT_POSITION_PRECEDING?((t=!!a)&&!(t=a===n)&&(t=ne(n,a,C),t===null?t=!1:(h(t,!0,te,a,n),a=x,x=null,t=a!==null)),t):e&Node.DOCUMENT_POSITION_FOLLOWING?((t=!!a)&&!(t=a===r)&&(t=ne(r,a,C),t===null?t=!1:(h(t,!0,S,a,r),a=x,ee=x=null,t=a!==null)),t):!1}function Zp(e,t){var n=e.ownerDocument.createRange();n.selectNodeContents(e),e=n.getBoundingClientRect(),window.scrollTo(window.scrollX+e.left,t?window.scrollY+e.top:window.scrollY+e.bottom-window.innerHeight)}Fp.prototype.scrollIntoView=function(e){if(typeof e==`object`)throw Error(i(566));var t=[];h(this._fragmentFiber.child,!1,Hp,t,void 0,void 0);var n=!1!==e;if(t.length===0){var r=v(this._fragmentFiber);if(r=n?r[1]||r[0]||g(this._fragmentFiber):r[0]||r[1],r===null)return;if(r.tag===6){e=b(r),Zp(e,n);return}if(r=b(r),r.nodeType!==9){if(r.nodeType===11){n=`host`in r?r.host:null,n!==null&&n.scrollIntoView(e);return}r.scrollIntoView(e)}}for(r=n?t.length-1:0;r!==(n?-1:t.length);){var a=t[r];a.tag===6?(a=b(a),Zp(a,n)):b(a).scrollIntoView(e),r+=n?-1:1}};function Qp(e,t){return e=b(e),$p(e,t),!1}function $p(e,t){e.reactFragments??=new Set,e.reactFragments.add(t)}function em(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.addEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){for(var r=0,i=0;i<Kp.length;i++){var a=Kp[i];(a.fragmentInstance!==t||a.observer!==n||a.instance!==e)&&(Kp[r++]=a)}Kp.length=r,n.observe(e)}),$p(e,t))}function tm(e,t){var n=t._eventListeners;if(n!==null)for(var r=0;r<n.length;r++){var i=n[r];e.removeEventListener(i.type,i.attachedListener,Rp(i.optionsOrUseCapture))}e.nodeType!==3&&(n=t._observers,n!==null&&n.forEach(function(n){typeof n.rootMargin==`string`?Jp(t,n,e):n.unobserve(e)}),e.reactFragments!=null&&e.reactFragments.delete(t))}function nm(e){var t=e.firstChild;for(t&&t.nodeType===10&&(t=t.nextSibling);t;){var n=t;switch(t=t.nextSibling,n.nodeName){case`HTML`:case`HEAD`:case`BODY`:nm(n),zt(n);continue;case`SCRIPT`:case`STYLE`:continue;case`LINK`:if(n.rel.toLowerCase()===`stylesheet`)continue}e.removeChild(n)}}function rm(e,t,n,r){for(;e.nodeType===1;){var i=n;if(e.nodeName.toLowerCase()!==t.toLowerCase()){if(!r&&(e.nodeName!==`INPUT`||e.type!==`hidden`))break}else if(!r){if(t===`input`&&e.type===`hidden`){var a=i.name==null?null:``+i.name;if(i.type===`hidden`&&e.getAttribute(`name`)===a)return e}else return e}else if(!e[Lt])switch(t){case`meta`:if(!e.hasAttribute(`itemprop`))break;return e;case`link`:if(a=e.getAttribute(`rel`),a===`stylesheet`&&e.hasAttribute(`data-precedence`)||a!==i.rel||e.getAttribute(`href`)!==(i.href==null||i.href===``?null:i.href)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin)||e.getAttribute(`title`)!==(i.title==null?null:i.title))break;return e;case`style`:if(e.hasAttribute(`data-precedence`))break;return e;case`script`:if(a=e.getAttribute(`src`),(a!==(i.src==null?null:i.src)||e.getAttribute(`type`)!==(i.type==null?null:i.type)||e.getAttribute(`crossorigin`)!==(i.crossOrigin==null?null:i.crossOrigin))&&a&&e.hasAttribute(`async`)&&!e.hasAttribute(`itemprop`))break;return e;default:return e}if(e=lm(e.nextSibling),e===null)break}return null}function im(e,t,n){if(t===``)return null;for(;e.nodeType!==3;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!n||(e=lm(e.nextSibling),e===null))return null;return e}function am(e,t){for(;e.nodeType!==8;)if((e.nodeType!==1||e.nodeName!==`INPUT`||e.type!==`hidden`)&&!t||(e=lm(e.nextSibling),e===null))return null;return e}function om(e){return e.data===`$?`||e.data===`$~`}function sm(e){return e.data===`$!`||e.data===`$?`&&e.ownerDocument.readyState!==`loading`}function cm(e,t){var n=e.ownerDocument;if(e.data===`$~`)e._reactRetry=t;else if(e.data!==`$?`||n.readyState!==`loading`)t();else{var r=function(){t(),n.removeEventListener(`DOMContentLoaded`,r)};n.addEventListener(`DOMContentLoaded`,r),e._reactRetry=r}}function lm(e){for(;e!=null;e=e.nextSibling){var t=e.nodeType;if(t===1||t===3)break;if(t===8){if(t=e.data,t===`$`||t===`$!`||t===`$?`||t===`$~`||t===`&`||t===`F!`||t===`F`)break;if(t===`/$`||t===`/&`)return null}}return e}var um=null;function dm(e){e=e.nextSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`/$`||n===`/&`){if(t===0)return lm(e.nextSibling);t--}else n!==`$`&&n!==`$!`&&n!==`$?`&&n!==`$~`&&n!==`&`||t++}e=e.nextSibling}return null}function fm(e){e=e.previousSibling;for(var t=0;e;){if(e.nodeType===8){var n=e.data;if(n===`$`||n===`$!`||n===`$?`||n===`$~`||n===`&`){if(t===0)return e;t--}else n!==`/$`&&n!==`/&`||t++}e=e.previousSibling}return null}function pm(e,t){function n(){r=!0}if(e.ownerDocument.activeElement===e)return!0;var r=!1;try{e.ownerDocument.addEventListener(`focus`,n,!0),(e.focus||HTMLElement.prototype.focus).call(e,t)}finally{e.ownerDocument.removeEventListener(`focus`,n,!0)}return r}function mm(e){yp(function(){yp(function(t){return e(t)})})}function hm(e,t,n){switch(t=lp(n),e){case`html`:if(e=t.documentElement,!e)throw Error(i(452));return e;case`head`:if(e=t.head,!e)throw Error(i(453));return e;case`body`:if(e=t.body,!e)throw Error(i(454));return e;default:throw Error(i(451))}}function gm(e,t,n){for(var r in n){var i=n[r];n.hasOwnProperty(r)&&i!=null&&$(e,t,r,null,rp,i)}n.dangerouslySetInnerHTML!=null&&(e.textContent=``),e.onclick===wn&&(e.onclick=null),zt(e)}function _m(e){for(var t=e.attributes;t.length;)e.removeAttributeNode(t[0]);zt(e)}var vm=new Map,ym=new Set;function bm(e){if(typeof e.getRootNode==`function`){var t=e.getRootNode();if(t.nodeType===9||t.nodeType===11)return t}return e.nodeType===9?e:e.ownerDocument}var xm=O.d;O.d={f:Sm,r:Cm,D:Em,C:Dm,L:Om,m:km,X:jm,S:Am,M:Mm};function Sm(){var e=xm.f(),t=zd();return e||t}function Cm(e){var t=Vt(e);t!==null&&t.tag===5&&t.type===`form`?ac(t):xm.r(e)}var wm=typeof document>`u`?null:document;function Tm(e,t,n){var r=wm;if(r&&typeof t==`string`&&t){var i=ln(t);i=`link[rel="`+e+`"][href="`+i+`"]`,typeof n==`string`&&(i+=`[crossorigin="`+n+`"]`),ym.has(i)||(ym.add(i),e={rel:e,crossOrigin:n,href:t},r.querySelector(i)===null&&(t=r.createElement(`link`),np(t,`link`,e),j(t),r.head.appendChild(t)))}}function Em(e){xm.D(e),Tm(`dns-prefetch`,e,null)}function Dm(e,t){xm.C(e,t),Tm(`preconnect`,e,t)}function Om(e,t,n){xm.L(e,t,n);var r=wm;if(r&&e&&t){var i=`link[rel="preload"][as="`+ln(t)+`"]`;t===`image`&&n&&n.imageSrcSet?(i+=`[imagesrcset="`+ln(n.imageSrcSet)+`"]`,typeof n.imageSizes==`string`&&(i+=`[imagesizes="`+ln(n.imageSizes)+`"]`)):i+=`[href="`+ln(e)+`"]`;var a=i;switch(t){case`style`:a=Pm(e);break;case`script`:a=Rm(e)}if(!(vm.has(a)||(e=w({rel:`preload`,href:t===`image`&&n&&n.imageSrcSet?void 0:e,as:t},n),vm.set(a,e),r.querySelector(i)!==null||t===`style`&&r.querySelector(Fm(a))||t===`script`&&r.querySelector(zm(a))))){var o=r.createElement(`link`);np(o,`link`,e),t===`style`&&(o[Rt]=!0,o.onload=o.onerror=function(){Wt(o)}),j(o),r.head.appendChild(o)}}}function km(e,t){xm.m(e,t);var n=wm;if(n&&e){var r=t&&typeof t.as==`string`?t.as:`script`,i=`link[rel="modulepreload"][as="`+ln(r)+`"][href="`+ln(e)+`"]`,a=i;switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:a=Rm(e)}if(!vm.has(a)&&(e=w({rel:`modulepreload`,href:e},t),vm.set(a,e),n.querySelector(i)===null)){switch(r){case`audioworklet`:case`paintworklet`:case`serviceworker`:case`sharedworker`:case`worker`:case`script`:if(n.querySelector(zm(a)))return}r=n.createElement(`link`),np(r,`link`,e),j(r),n.head.appendChild(r)}}}function Am(e,t,n){xm.S(e,t,n);var r=wm;if(r&&e){var i=Ut(r).hoistableStyles,a=Pm(e);t||=`default`;var o=i.get(a);if(!o){var s={loading:0,preload:null};if(o=r.querySelector(Fm(a)))s.loading=5;else{e=w({rel:`stylesheet`,href:e,"data-precedence":t},n),(n=vm.get(a))&&Hm(e,n);var c=o=r.createElement(`link`);j(c),np(c,`link`,e),c._p=new Promise(function(e,t){c.onload=e,c.onerror=t}),c.addEventListener(`load`,function(){s.loading|=1}),c.addEventListener(`error`,function(){s.loading|=2}),s.loading|=4,Vm(o,t,r)}o={type:`stylesheet`,instance:o,count:1,state:s},i.set(a,o)}}}function jm(e,t){xm.X(e,t);var n=wm;if(n&&e){var r=Ut(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=w({src:e,async:!0},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),j(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Mm(e,t){xm.M(e,t);var n=wm;if(n&&e){var r=Ut(n).hoistableScripts,i=Rm(e),a=r.get(i);a||(a=n.querySelector(zm(i)),a||(e=w({src:e,async:!0,type:`module`},t),(t=vm.get(i))&&Um(e,t),a=n.createElement(`script`),j(a),np(a,`link`,e),n.head.appendChild(a)),a={type:`script`,instance:a,count:1,state:null},r.set(i,a))}}function Nm(e,t,n,r){var a=(a=je.current)?bm(a):null;if(!a)throw Error(i(446));switch(e){case`meta`:case`title`:return null;case`style`:return typeof n.precedence==`string`&&typeof n.href==`string`?(n=Pm(n.href),t=Ut(a).hoistableStyles,r=t.get(n),r||(r={type:`style`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};case`link`:if(n.rel===`stylesheet`&&typeof n.href==`string`&&typeof n.precedence==`string`){e=Pm(n.href);var o=Ut(a).hoistableStyles,s=o.get(e);if(s||(a=a.ownerDocument||a,s={type:`stylesheet`,instance:null,count:0,state:{loading:0,preload:null}},o.set(e,s),(o=a.querySelector(Fm(e)))?o._p||(s.instance=o,s.state.loading=5):(o=vm.get(e),o||(o={rel:`preload`,as:`style`,href:n.href,crossOrigin:n.crossOrigin,integrity:n.integrity,media:n.media,hrefLang:n.hrefLang,referrerPolicy:n.referrerPolicy},vm.set(e,o)),Lm(a,e,o,s.state))),t&&r===null)throw Error(i(528,``));return s}if(t&&r!==null)throw Error(i(529,``));return null;case`script`:return t=n.async,n=n.src,typeof n==`string`&&t&&typeof t!=`function`&&typeof t!=`symbol`?(n=Rm(n),t=Ut(a).hoistableScripts,r=t.get(n),r||(r={type:`script`,instance:null,count:0,state:null},t.set(n,r)),r):{type:`void`,instance:null,count:0,state:null};default:throw Error(i(444,e))}}function Pm(e){return`href="`+ln(e)+`"`}function Fm(e){return`link[rel="stylesheet"][`+e+`]`}function Im(e){return w({},e,{"data-precedence":e.precedence,precedence:null})}function Lm(e,t,n,r){if(t=e.querySelector(`link[rel="preload"][as="style"][`+t+`]`)){if(!0!==t[Rt]){r.loading=1;return}}else t=e.createElement(`link`),t[Rt]=!0,t.onload=t.onerror=Wt.bind(null,t),np(t,`link`,n),j(t),e.head.appendChild(t);r.preload=t,t.addEventListener(`load`,function(){return r.loading|=1}),t.addEventListener(`error`,function(){return r.loading|=2})}function Rm(e){return`[src="`+ln(e)+`"]`}function zm(e){return`script[async]`+e}function Bm(e,t,n){if(t.count++,t.instance===null)switch(t.type){case`style`:var r=e.querySelector(`style[data-href~="`+ln(n.href)+`"]`);if(r)return t.instance=r,j(r),r;var a=w({},n,{"data-href":n.href,"data-precedence":n.precedence,href:null,precedence:null});return r=(e.ownerDocument||e).createElement(`style`),j(r),np(r,`style`,a),Vm(r,n.precedence,e),t.instance=r;case`stylesheet`:a=Pm(n.href);var o=e.querySelector(Fm(a));if(o)return t.state.loading|=4,t.instance=o,j(o),o;r=Im(n),(a=vm.get(a))&&Hm(r,a),o=(e.ownerDocument||e).createElement(`link`),j(o);var s=o;return s._p=new Promise(function(e,t){s.onload=e,s.onerror=t}),np(o,`link`,r),t.state.loading|=4,Vm(o,n.precedence,e),t.instance=o;case`script`:return o=Rm(n.src),(a=e.querySelector(zm(o)))?(t.instance=a,j(a),a):(r=n,(a=vm.get(o))&&(r=w({},n),Um(r,a)),e=e.ownerDocument||e,a=e.createElement(`script`),j(a),np(a,`link`,r),e.head.appendChild(a),t.instance=a);case`void`:return null;default:throw Error(i(443,t.type))}else t.type===`stylesheet`&&!(t.state.loading&4)&&(r=t.instance,t.state.loading|=4,Vm(r,n.precedence,e));return t.instance}function Vm(e,t,n){for(var r=n.querySelectorAll(`link[rel="stylesheet"][data-precedence],style[data-precedence]`),i=r.length?r[r.length-1]:null,a=i,o=0;o<r.length;o++){var s=r[o];if(s.dataset.precedence===t)a=s;else if(a!==i)break}a?a.parentNode.insertBefore(e,a.nextSibling):(t=n.nodeType===9?n.head:n,t.insertBefore(e,t.firstChild))}function Hm(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.title??=t.title}function Um(e,t){e.crossOrigin??=t.crossOrigin,e.referrerPolicy??=t.referrerPolicy,e.integrity??=t.integrity}var Wm=null;function Gm(e,t,n){if(Wm===null){var r=new Map,i=Wm=new Map;i.set(n,r)}else i=Wm,r=i.get(n),r||(r=new Map,i.set(n,r));if(r.has(e))return r;for(r.set(e,null),n=n.getElementsByTagName(e),i=0;i<n.length;i++){var a=n[i];if(!(a[Lt]||a[At]||e===`link`&&a.getAttribute(`rel`)===`stylesheet`)&&a.namespaceURI!==`http://www.w3.org/2000/svg`){var o=a.getAttribute(t)||``;o=e+o;var s=r.get(o);s?s.push(a):r.set(o,[a])}}return r}function Km(e,t,n){e=e.ownerDocument||e,e.head.insertBefore(n,t===`title`?e.querySelector(`head > title`):null)}function qm(e,t,n){if(n===1||t.itemProp!=null)return!1;switch(e){case`meta`:case`title`:return!0;case`style`:if(typeof t.precedence!=`string`||typeof t.href!=`string`||t.href===``)break;return!0;case`link`:if(typeof t.rel!=`string`||typeof t.href!=`string`||t.href===``||t.onLoad||t.onError)break;switch(t.rel){case`stylesheet`:return e=t.disabled,typeof t.precedence==`string`&&e==null;default:return!0}case`script`:if(t.async&&typeof t.async!=`function`&&typeof t.async!=`symbol`&&!t.onLoad&&!t.onError&&t.src&&typeof t.src==`string`)return!0}return!1}function Jm(e,t){return e===`img`&&t.src!=null&&t.src!==``&&t.onLoad==null&&t.loading!==`lazy`}function Ym(e){return!(e.type===`stylesheet`&&!(e.state.loading&3))}function Xm(e){return(e.width||100)*(e.height||100)*(typeof devicePixelRatio==`number`?devicePixelRatio:1)*.25}function Zm(e,t){typeof t.decode==`function`&&(e.imgCount++,t.complete||(e.imgBytes+=Xm(t),e.suspenseyImages.push(t)),e=rh.bind(e),t.decode().then(e,e))}function Qm(e,t,n,r){if(n.type===`stylesheet`&&(typeof r.media!=`string`||!1!==matchMedia(r.media).matches)&&!(n.state.loading&4)){if(n.instance===null){var i=Pm(r.href),a=t.querySelector(Fm(i));if(a){t=a._p,typeof t==`object`&&t&&typeof t.then==`function`&&(e.count++,e=nh.bind(e),t.then(e,e)),n.state.loading|=4,n.instance=a,j(a);return}a=t.ownerDocument||t,r=Im(r),(i=vm.get(i))&&Hm(r,i),a=a.createElement(`link`),j(a);var o=a;o._p=new Promise(function(e,t){o.onload=e,o.onerror=t}),np(a,`link`,r),n.instance=a}e.stylesheets===null&&(e.stylesheets=new Map),e.stylesheets.set(n,t),(t=n.state.preload)&&!(n.state.loading&3)&&(e.count++,n=nh.bind(e),t.addEventListener(`load`,n),t.addEventListener(`error`,n))}}var $m=0;function eh(e,t){return e.stylesheets&&e.count===0&&ah(e,e.stylesheets),0<e.count||0<e.imgCount?function(n){var r=setTimeout(function(){if(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}},6e4+t);0<e.imgBytes&&$m===0&&($m=62500*op());var i=setTimeout(function(){if(e.waitingForImages=!1,e.count===0&&(e.stylesheets&&ah(e,e.stylesheets),e.unsuspend)){var t=e.unsuspend;e.unsuspend=null,t()}},(e.imgBytes>$m?50:800)+t);return e.unsuspend=n,function(){e.unsuspend=null,clearTimeout(r),clearTimeout(i)}}:null}function th(e){if(e.count===0&&(e.imgCount===0||!e.waitingForImages)){if(e.stylesheets)ah(e,e.stylesheets);else if(e.unsuspend){var t=e.unsuspend;e.unsuspend=null,t()}}}function nh(){this.count--,th(this)}function rh(){this.imgCount--,th(this)}var ih=null;function ah(e,t){e.stylesheets=null,e.unsuspend!==null&&(e.count++,ih=new Map,t.forEach(oh,e),ih=null,nh.call(e))}function oh(e,t){if(!(t.state.loading&4)){var n=ih.get(e);if(n)var r=n.get(null);else{n=new Map,ih.set(e,n);for(var i=e.querySelectorAll(`link[data-precedence],style[data-precedence]`),a=0;a<i.length;a++){var o=i[a];(o.nodeName===`LINK`||o.getAttribute(`media`)!==`not all`)&&(n.set(o.dataset.precedence,o),r=o)}r&&n.set(null,r)}i=t.instance,o=i.getAttribute(`data-precedence`),a=n.get(o)||r,a===r&&n.set(null,i),n.set(o,i),this.count++,r=nh.bind(this),i.addEventListener(`load`,r),i.addEventListener(`error`,r),a?a.parentNode.insertBefore(i,a.nextSibling):(e=e.nodeType===9?e.head:e,e.insertBefore(i,e.firstChild)),t.state.loading|=4}}var sh={$$typeof:ue,Provider:null,Consumer:null,_currentValue:we,_currentValue2:we,_threadCount:0};function ch(e,t,n,r,i,a,o,s,c){this.tag=1,this.containerInfo=e,this.pingCache=this.current=this.pendingChildren=null,this.timeoutHandle=-1,this.callbackNode=this.next=this.pendingContext=this.context=this.cancelPendingCommit=null,this.callbackPriority=0,this.expirationTimes=yt(-1),this.entangledLanes=this.shellSuspendCounter=this.errorRecoveryDisabledLanes=this.expiredLanes=this.warmLanes=this.pingedLanes=this.suspendedLanes=this.pendingLanes=0,this.entanglements=yt(0),this.hiddenUpdates=yt(null),this.identifierPrefix=r,this.onUncaughtError=i,this.onCaughtError=a,this.onRecoverableError=o,this.pooledCache=null,this.pooledCacheLanes=0,this.formState=c,this.transitionTypes=null,this.incompleteTransitions=new Map}function lh(e,t,n,r,i,a,o,s,c,l,u,d){return e=new ch(e,t,n,o,c,l,u,d,s),t=1,!0===a&&(t|=24),a=Ri(3,null,null,t),e.current=a,a.stateNode=e,t=Ia(),t.refCount++,e.pooledCache=t,t.refCount++,a.memoizedState={element:r,isDehydrated:n,cache:t},yo(a),e}function uh(e){return e?(e=Ii,e):Ii}function dh(e,t,n,r,i,a){i=uh(i),r.context===null?r.context=i:r.pendingContext=i,r=xo(t),r.payload={element:n},a=a===void 0?null:a,a!==null&&(r.callback=a),n=So(e,r,t),n!==null&&(Pd(n,e,t),Co(n,e,t))}function fh(e,t){if(e=e.memoizedState,e!==null&&e.dehydrated!==null){var n=e.retryLane;e.retryLane=n!==0&&n<t?n:t}}function ph(e,t){fh(e,t),(e=e.alternate)&&fh(e,t)}function mh(e){if(e.tag===13||e.tag===31){var t=Ni(e,67108864);t!==null&&Pd(t,e,67108864),ph(e,67108864)}}function hh(e){if(e.tag===13||e.tag===31){var t=jd();t=Tt(t);var n=Ni(e,t);n!==null&&Pd(n,e,t),ph(e,t)}}var gh=!0;function _h(e,t,n,r){var i=D.T;D.T=null;var a=O.p;try{O.p=2,yh(e,t,n,r)}finally{O.p=a,D.T=i}}function vh(e,t,n,r){var i=D.T;D.T=null;var a=O.p;try{O.p=8,yh(e,t,n,r)}finally{O.p=a,D.T=i}}function yh(e,t,n,r){if(gh){var i=bh(r);if(i===null)Kf(e,t,r,xh,n),Mh(e,r);else if(Ph(i,e,t,n,r))r.stopPropagation();else if(Mh(e,r),t&4&&-1<jh.indexOf(e)){for(;i!==null;){var a=Vt(i);if(a!==null)switch(a.tag){case 3:if(a=a.stateNode,a.current.memoizedState.isDehydrated){var o=pt(a.pendingLanes);if(o!==0){var s=a;for(s.pendingLanes|=2,s.entangledLanes|=2;o;){var c=1<<31-ot(o);s.entanglements[1]|=c,o&=~c}Ef(a),!(W&6)&&(_d=Je()+500,Df(0,!1))}}break;case 31:case 13:s=Ni(a,2),s!==null&&Pd(s,a,2),zd(),ph(a,2)}if(a=bh(r),a===null&&Kf(e,t,r,xh,n),a===i)break;i=a}i!==null&&r.stopPropagation()}else Kf(e,t,r,null,n)}}function bh(e){return e=En(e),Sh(e)}var xh=null;function Sh(e){if(xh=null,e=Bt(e),e!==null){var t=o(e);if(t===null)e=null;else{var n=t.tag;if(n===13){if(e=s(t),e!==null)return e;e=null}else if(n===31){if(e=c(t),e!==null)return e;e=null}else if(n===3){if(t.stateNode.current.memoizedState.isDehydrated)return t.tag===3?t.stateNode.containerInfo:null;e=null}else t!==e&&(e=null)}}return xh=e,null}function Ch(e){switch(e){case`beforetoggle`:case`cancel`:case`click`:case`close`:case`contextmenu`:case`copy`:case`cut`:case`auxclick`:case`dblclick`:case`dragend`:case`dragstart`:case`drop`:case`focusin`:case`focusout`:case`input`:case`invalid`:case`keydown`:case`keypress`:case`keyup`:case`mousedown`:case`mouseup`:case`paste`:case`pause`:case`play`:case`pointercancel`:case`pointerdown`:case`pointerup`:case`ratechange`:case`reset`:case`seeked`:case`submit`:case`toggle`:case`touchcancel`:case`touchend`:case`touchstart`:case`volumechange`:case`change`:case`selectionchange`:case`textInput`:case`compositionstart`:case`compositionend`:case`compositionupdate`:case`beforeblur`:case`afterblur`:case`beforeinput`:case`blur`:case`fullscreenchange`:case`fullscreenerror`:case`focus`:case`hashchange`:case`popstate`:case`select`:case`selectstart`:return 2;case`drag`:case`dragenter`:case`dragexit`:case`dragleave`:case`dragover`:case`mousemove`:case`mouseout`:case`mouseover`:case`pointermove`:case`pointerout`:case`pointerover`:case`resize`:case`scroll`:case`touchmove`:case`wheel`:case`mouseenter`:case`mouseleave`:case`pointerenter`:case`pointerleave`:return 8;case`message`:switch(Ye()){case Xe:return 2;case Ze:return 8;case Qe:case $e:return 32;case et:return 268435456;default:return 32}default:return 32}}var wh=!1,Th=null,Eh=null,Dh=null,Oh=new Map,kh=new Map,Ah=[],jh=`mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset`.split(` `);function Mh(e,t){switch(e){case`focusin`:case`focusout`:Th=null;break;case`dragenter`:case`dragleave`:Eh=null;break;case`mouseover`:case`mouseout`:Dh=null;break;case`pointerover`:case`pointerout`:Oh.delete(t.pointerId);break;case`gotpointercapture`:case`lostpointercapture`:kh.delete(t.pointerId)}}function Nh(e,t,n,r,i,a){return e===null||e.nativeEvent!==a?(e={blockedOn:t,domEventName:n,eventSystemFlags:r,nativeEvent:a,targetContainers:[i]},t!==null&&(t=Vt(t),t!==null&&mh(t)),e):(e.eventSystemFlags|=r,t=e.targetContainers,i!==null&&t.indexOf(i)===-1&&t.push(i),e)}function Ph(e,t,n,r,i){switch(t){case`focusin`:return Th=Nh(Th,e,t,n,r,i),!0;case`dragenter`:return Eh=Nh(Eh,e,t,n,r,i),!0;case`mouseover`:return Dh=Nh(Dh,e,t,n,r,i),!0;case`pointerover`:var a=i.pointerId;return Oh.set(a,Nh(Oh.get(a)||null,e,t,n,r,i)),!0;case`gotpointercapture`:return a=i.pointerId,kh.set(a,Nh(kh.get(a)||null,e,t,n,r,i)),!0}return!1}function Fh(e){var t=Bt(e.target);if(t!==null){var n=o(t);if(n!==null){if(t=n.tag,t===13){if(t=s(n),t!==null){e.blockedOn=t,Ot(e.priority,function(){hh(n)});return}}else if(t===31){if(t=c(n),t!==null){e.blockedOn=t,Ot(e.priority,function(){hh(n)});return}}else if(t===3&&n.stateNode.current.memoizedState.isDehydrated){e.blockedOn=n.tag===3?n.stateNode.containerInfo:null;return}}}e.blockedOn=null}function Ih(e){if(e.blockedOn!==null)return!1;for(var t=e.targetContainers;0<t.length;){var n=bh(e.nativeEvent);if(n===null){n=e.nativeEvent;var r=new n.constructor(n.type,n);Tn=r,n.target.dispatchEvent(r),Tn=null}else return t=Vt(n),t!==null&&mh(t),e.blockedOn=n,!1;t.shift()}return!0}function Lh(e,t,n){Ih(e)&&n.delete(t)}function Rh(){wh=!1,Th!==null&&Ih(Th)&&(Th=null),Eh!==null&&Ih(Eh)&&(Eh=null),Dh!==null&&Ih(Dh)&&(Dh=null),Oh.forEach(Lh),kh.forEach(Lh)}function zh(e,n){e.blockedOn===n&&(e.blockedOn=null,wh||(wh=!0,t.unstable_scheduleCallback(t.unstable_NormalPriority,Rh)))}var Bh=null;function Vh(e){Bh!==e&&(Bh=e,t.unstable_scheduleCallback(t.unstable_NormalPriority,function(){Bh===e&&(Bh=null);for(var t=0;t<e.length;t+=3){var n=e[t],r=e[t+1],i=e[t+2];if(typeof r!=`function`){if(Sh(r||n)===null)continue;break}var a=Vt(n);a!==null&&(e.splice(t,3),t-=3,rc(a,{pending:!0,data:i,method:n.method,action:r},r,i))}}))}function Hh(e){function t(t){return zh(t,e)}Th!==null&&zh(Th,e),Eh!==null&&zh(Eh,e),Dh!==null&&zh(Dh,e),Oh.forEach(t),kh.forEach(t);for(var n=0;n<Ah.length;n++){var r=Ah[n];r.blockedOn===e&&(r.blockedOn=null)}for(;0<Ah.length&&(n=Ah[0],n.blockedOn===null);)Fh(n),n.blockedOn===null&&Ah.shift();if(n=(e.ownerDocument||e).$$reactFormReplay,n!=null)for(r=0;r<n.length;r+=3){var i=n[r],a=n[r+1],o=i[jt]||null;if(typeof a==`function`)o||Vh(n);else if(o){var s=null;if(a&&a.hasAttribute(`formAction`)){if(i=a,o=a[jt]||null)s=o.formAction;else if(Sh(i)!==null)continue}else s=o.action;typeof s==`function`?n[r+1]=s:(n.splice(r,3),r-=3),Vh(n)}}}function Uh(){function e(e){e.canIntercept&&e.info===`react-transition`&&e.intercept({handler:function(){return new Promise(function(e){return i=e})},focusReset:`manual`,scroll:`manual`})}function t(){i!==null&&(i(),i=null),r||setTimeout(n,20)}function n(){if(!r&&!navigation.transition){var e=navigation.currentEntry;e&&e.url!=null&&navigation.navigate(e.url,{state:e.getState(),info:`react-transition`,history:`replace`})}}if(typeof navigation==`object`){var r=!1,i=null;return navigation.addEventListener(`navigate`,e),navigation.addEventListener(`navigatesuccess`,t),navigation.addEventListener(`navigateerror`,t),setTimeout(n,100),function(){r=!0,navigation.removeEventListener(`navigate`,e),navigation.removeEventListener(`navigatesuccess`,t),navigation.removeEventListener(`navigateerror`,t),i!==null&&(i(),i=null)}}}function Wh(e){this._internalRoot=e}Gh.prototype.render=Wh.prototype.render=function(e){var t=this._internalRoot;if(t===null)throw Error(i(409));var n=t.current;dh(n,jd(),e,t,null,null)},Gh.prototype.unmount=Wh.prototype.unmount=function(){var e=this._internalRoot;if(e!==null){this._internalRoot=null;var t=e.containerInfo;dh(e.current,2,null,e,null,null),zd(),t[Mt]=null}};function Gh(e){this._internalRoot=e}Gh.prototype.unstable_scheduleHydration=function(e){if(e){var t=Dt();e={blockedOn:null,target:e,priority:t};for(var n=0;n<Ah.length&&t!==0&&t<Ah[n].priority;n++);Ah.splice(n,0,e),n===0&&Fh(e)}};var Kh=n.version;if(Kh!==`19.3.0`)throw Error(i(527,Kh,`19.3.0`));O.findDOMNode=function(e){var t=e._reactInternals;if(t===void 0)throw typeof e.render==`function`?Error(i(188)):(e=Object.keys(e).join(`,`),Error(i(268,e)));return e=d(t),e=e===null?null:p(e),e=e===null?null:e.stateNode,e};var qh={bundleType:0,version:`19.3.0`,rendererPackageName:`react-dom`,currentDispatcherRef:D,reconcilerVersion:`19.3.0`};if(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__<`u`){var Jh=__REACT_DEVTOOLS_GLOBAL_HOOK__;if(!Jh.isDisabled&&Jh.supportsFiber)try{rt=Jh.inject(qh),it=Jh}catch{}}e.createRoot=function(e,t){if(!a(e))throw Error(i(299));var n=!1,r=``,o=Ec,s=Dc,c=Oc;return t!=null&&(!0===t.unstable_strictMode&&(n=!0),t.identifierPrefix!==void 0&&(r=t.identifierPrefix),t.onUncaughtError!==void 0&&(o=t.onUncaughtError),t.onCaughtError!==void 0&&(s=t.onCaughtError),t.onRecoverableError!==void 0&&(c=t.onRecoverableError)),t=lh(e,1,!1,null,null,n,r,null,o,s,c,Uh),e[Mt]=t.current,Wf(e),new Wh(t)}})),g=o(((e,t)=>{function n(){if(!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__>`u`||typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE!=`function`))try{__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(n)}catch(e){console.error(e)}}n(),t.exports=h()})),_=o((e=>{var t=Symbol.for(`react.transitional.element`),n=Symbol.for(`react.fragment`);function r(e,n,r){var i=null;if(r!==void 0&&(i=``+r),n.key!==void 0&&(i=``+n.key),`key`in n)for(var a in r={},n)a!==`key`&&(r[a]=n[a]);else r=n;return n=r.ref,{$$typeof:t,type:e,key:i,ref:n===void 0?null:n,props:r}}e.Fragment=n,e.jsx=r,e.jsxs=r})),v=o(((e,t)=>{t.exports=_()})),y=g(),b=c(u(),1),x=v(),ee={models:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h4A1.5 1.5 0 0 1 12 6.5v4a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 10.5zm8 0A1.5 1.5 0 0 1 14.5 5h3A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-3a1.5 1.5 0 0 1-1.5-1.5zm-8 8A1.5 1.5 0 0 1 6.5 13h4a1.5 1.5 0 0 1 1.5 1.5v3a1.5 1.5 0 0 1-1.5 1.5h-4A1.5 1.5 0 0 1 5 17.5z`}),chat:(0,x.jsx)(`path`,{d:`M5.5 7.5A2.5 2.5 0 0 1 8 5h8a2.5 2.5 0 0 1 2.5 2.5v5A2.5 2.5 0 0 1 16 15h-3.7l-3.6 3.2a.7.7 0 0 1-1.2-.52V15A2.5 2.5 0 0 1 5.5 12.5z`}),activity:(0,x.jsx)(`path`,{d:`M4.5 13h3l1.8-5.5 3.2 9 2.1-6.5h4.9`}),settings:(0,x.jsx)(`path`,{d:`M12 8.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6Zm0-3.2v2m0 10v2m7-7h-2M7 12H5m11.95-4.95-1.42 1.42M8.47 15.53l-1.42 1.42m9.9 0-1.42-1.42M8.47 8.47 7.05 7.05`}),gallery:(0,x.jsx)(`path`,{d:`M5 6.5A1.5 1.5 0 0 1 6.5 5h11A1.5 1.5 0 0 1 19 6.5v11a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 17.5zm3 2h8m-8 3h8m-8 3h5`}),command:(0,x.jsx)(`path`,{d:`M8 8.5A2.5 2.5 0 1 1 5.5 6 2.5 2.5 0 0 1 8 8.5Zm0 7A2.5 2.5 0 1 1 5.5 13 2.5 2.5 0 0 1 8 15.5Zm8-7A2.5 2.5 0 1 1 13.5 6 2.5 2.5 0 0 1 16 8.5Zm0 7A2.5 2.5 0 1 1 13.5 13 2.5 2.5 0 0 1 16 15.5Z`}),help:(0,x.jsx)(`path`,{d:`M9.5 9a2.5 2.5 0 1 1 4.4 1.62c-.68.72-1.9 1.18-1.9 2.38v.25m0 3.25h.01M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z`}),menu:(0,x.jsx)(`path`,{d:`M5 7h14M5 12h14M5 17h14`}),close:(0,x.jsx)(`path`,{d:`m7 7 10 10M17 7 7 17`}),search:(0,x.jsx)(`path`,{d:`m16.5 16.5 3 3M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14Z`}),warning:(0,x.jsx)(`path`,{d:`M12 4.8 21 19H3zm0 5.2v4m0 2.5h.01`}),key:(0,x.jsx)(`path`,{d:`M14.5 9.5A4.5 4.5 0 1 0 11 13.9l2.1 2.1H16v2h2v2h2v-2.9l-4.6-4.6a4.5 4.5 0 0 0-.9-3Z`}),schema:(0,x.jsx)(`path`,{d:`M6 5h7l5 5v9H6zm7 0v5h5M9 13h6M9 16h6M9 10h2`})};function te(e){return(0,x.jsx)(`svg`,{className:e.className??`ds-icon`,"aria-hidden":`true`,viewBox:`0 0 24 24`,fill:`none`,stroke:`currentColor`,strokeWidth:`1.8`,strokeLinecap:`round`,strokeLinejoin:`round`,children:ee[e.name]})}function S(e){let{tone:t=`secondary`,busy:n=!1,className:r=``,children:i,disabled:a,...o}=e;return(0,x.jsxs)(`button`,{className:`ds-button ds-button-${t} ${r}`.trim(),disabled:a||n,"aria-busy":n||void 0,"data-busy":n,...o,children:[n?(0,x.jsx)(`span`,{className:`ds-spinner`,"aria-hidden":`true`}):null,(0,x.jsx)(`span`,{children:i})]})}function C(e){let{label:t,icon:n,className:r=``,busy:i=!1,disabled:a,...o}=e;return(0,x.jsx)(`button`,{className:`ds-icon-button ${r}`.trim(),"aria-label":t,title:t,disabled:a||i,"aria-busy":i||void 0,"data-busy":i,...o,children:ne(n)?(0,x.jsx)(te,{name:n}):(0,x.jsx)(`span`,{"aria-hidden":`true`,children:n})})}function ne(e){return[`models`,`chat`,`activity`,`settings`,`gallery`,`command`,`help`,`menu`,`close`,`search`,`warning`,`key`,`schema`].includes(e)}function w(e){let t=(0,b.useId)(),n=`${t}-hint`,r=`${t}-error`,i=[e.hint?n:null,e.error?r:null].filter(Boolean).join(` `)||void 0;return(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:t,"data-disabled":e.disabled||e.busy||void 0,children:[(0,x.jsx)(`span`,{children:e.label}),(0,x.jsx)(`input`,{id:t,value:e.value,placeholder:e.placeholder,disabled:e.disabled||e.busy,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":i,"data-testid":e.testId,onChange:t=>e.onChange?.(t.currentTarget.value)}),e.hint?(0,x.jsx)(`small`,{id:n,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:r,"data-tone":`error`,children:e.error}):null]})}function re(e){let t=(0,b.useId)(),n=`${t}-hint`,r=`${t}-error`,i=[e.hint?n:null,e.error?r:null].filter(Boolean).join(` `)||void 0;return(0,x.jsxs)(`label`,{className:`ds-field`,htmlFor:t,"data-disabled":e.disabled||e.busy||void 0,children:[(0,x.jsx)(`span`,{children:e.label}),(0,x.jsx)(`select`,{id:t,value:e.value,disabled:e.disabled||e.busy,"aria-busy":e.busy||void 0,"aria-invalid":e.error?`true`:void 0,"aria-describedby":i,"data-testid":e.testId,onKeyDown:t=>{if(t.key!==`ArrowDown`&&t.key!==`ArrowUp`)return;let n=e.options.findIndex(t=>t.value===e.value),r=t.key===`ArrowDown`?1:-1,i=e.options[(n+r+e.options.length)%e.options.length];i.disabled||(t.preventDefault(),e.onChange(i.value))},onChange:t=>e.onChange(t.currentTarget.value),children:e.options.map(e=>(0,x.jsx)(`option`,{value:e.value,disabled:e.disabled,children:e.label},e.value))}),e.hint?(0,x.jsx)(`small`,{id:n,"data-tone":`hint`,children:e.hint}):null,e.error?(0,x.jsx)(`small`,{id:r,"data-tone":`error`,children:e.error}):null]})}function ie(e){return(0,x.jsx)(`span`,{className:`ds-status`,"data-state":e.state,"aria-label":e.label,children:e.children})}function ae(e){let t=e.value==null?void 0:Math.max(0,Math.min(100,e.value));return(0,x.jsxs)(`div`,{className:`ds-progress`,children:[(0,x.jsx)(`div`,{role:`progressbar`,"aria-label":e.label,"aria-valuemin":0,"aria-valuemax":100,"aria-valuenow":t,"data-indeterminate":t==null,children:(0,x.jsx)(`span`,{"data-progress":t==null?`indeterminate`:String(Math.round(t/5)*5)})}),e.detail?(0,x.jsx)(`small`,{children:e.detail}):null]})}function oe(e){return(0,x.jsxs)(`section`,{className:`ds-empty`,"data-testid":e.testId,children:[(0,x.jsx)(`div`,{"aria-hidden":`true`,children:(0,x.jsx)(te,{name:`models`})}),(0,x.jsx)(`h2`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),e.action]})}function se(e){return(0,x.jsxs)(`section`,{className:`ds-banner`,"data-tone":e.tone??`error`,role:e.tone===`info`?`status`:`alert`,"data-testid":e.testId,children:[(0,x.jsx)(`strong`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),e.action]})}function ce(e){let t=(0,b.useRef)(null),n=(0,b.useRef)(null),r=(0,b.useId)(),i=e.labelledBy??r,a=(0,b.useRef)(!1),o=(0,b.useRef)(e.onClose);return o.current=e.onClose,(0,b.useEffect)(()=>{let r=t.current;r&&(e.open&&!r.open&&(n.current=document.activeElement instanceof HTMLElement?document.activeElement:null,r.showModal(),r.querySelector(`[data-autofocus], button, input, select, textarea, a[href], [tabindex]:not([tabindex="-1"])`)?.focus()),!e.open&&r.open&&(a.current=!0,r.close()))},[e.open]),(0,b.useEffect)(()=>{let e=t.current;if(!e)return;let r=()=>{let e=n.current;e?.isConnected&&e.focus(),n.current=null},i=()=>{a.current?a.current=!1:o.current(),window.setTimeout(r,0)},s=t=>{if(t.key!==`Tab`)return;let n=Array.from(e.querySelectorAll(`button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])`)).filter(e=>e.offsetParent!==null||e===document.activeElement);if(n.length===0)return;let r=n[0],i=n[n.length-1];t.shiftKey&&document.activeElement===r?(t.preventDefault(),i.focus()):!t.shiftKey&&document.activeElement===i&&(t.preventDefault(),r.focus())};return e.addEventListener(`close`,i),e.addEventListener(`keydown`,s),()=>{e.removeEventListener(`close`,i),e.removeEventListener(`keydown`,s)}},[]),(0,x.jsxs)(`dialog`,{className:`ds-dialog ${e.className??``}`.trim(),"data-position":e.position??`center`,ref:t,"aria-labelledby":i,"data-testid":e.testId,children:[(0,x.jsxs)(`header`,{children:[(0,x.jsx)(`h2`,{id:i,children:e.title}),(0,x.jsx)(C,{label:e.closeLabel??`Close`,icon:`close`,onClick:()=>{let e=n.current;a.current=!0,o.current(),t.current?.close(),window.setTimeout(()=>{e?.isConnected&&e.focus()},0)},"data-testid":`dialog-close`})]}),(0,x.jsx)(`div`,{children:e.children})]})}function le(e){return(0,x.jsx)(ce,{...e})}function ue(e){return(0,x.jsx)(ce,{...e,position:`left`,className:`ds-sheet ${e.className??``}`.trim()})}function de(e){let t=(0,b.useId)();return(0,x.jsxs)(`span`,{className:`ds-tooltip-wrap`,children:[b.cloneElement(e.children,{"aria-describedby":t}),(0,x.jsx)(`span`,{className:`ds-tooltip`,role:`tooltip`,id:t,children:e.label})]})}function fe(e){let t=(0,b.useId)(),n=e.tabs.find(t=>t.id===e.active)??e.tabs[0],r=t=>e.onChange(t);return(0,x.jsxs)(`section`,{className:`ds-tabs`,children:[(0,x.jsx)(`div`,{role:`tablist`,"aria-label":e.label??`Sections`,onKeyDown:i=>{let a=e.tabs.findIndex(e=>document.activeElement?.id===`${t}-tab-${e.id}`),o=a>=0?a:e.tabs.findIndex(e=>e.id===n.id),s=i.key===`ArrowRight`?o+1:i.key===`ArrowLeft`?o-1:i.key===`Home`?0:i.key===`End`?e.tabs.length-1:null;if(s===null)return;i.preventDefault();let c=e.tabs[(s+e.tabs.length)%e.tabs.length];r(c.id),requestAnimationFrame(()=>document.getElementById(`${t}-tab-${c.id}`)?.focus())},children:e.tabs.map(e=>(0,x.jsx)(`button`,{id:`${t}-tab-${e.id}`,role:`tab`,"aria-selected":e.id===n.id,"aria-controls":`${t}-panel-${e.id}`,tabIndex:e.id===n.id?0:-1,onClick:()=>r(e.id),children:e.label},e.id))}),(0,x.jsx)(`div`,{id:`${t}-panel-${n.id}`,role:`tabpanel`,"aria-labelledby":`${t}-tab-${n.id}`,children:n.panel})]})}function pe(e){return(0,x.jsxs)(`aside`,{className:`ds-inspector`,"aria-label":e.title,children:[(0,x.jsx)(`h2`,{children:e.title}),e.children]})}function me(e){return(0,x.jsxs)(`table`,{className:`ds-table`,children:[(0,x.jsx)(`caption`,{children:e.caption}),e.children]})}function he(e){return(0,x.jsx)(`ul`,{className:`ds-list`,"aria-label":e.label,children:e.children})}function ge(e){let[t,n]=b.useState(``),r=b.useId();return b.useEffect(()=>()=>n(``),[]),(0,x.jsxs)(`form`,{className:`ds-login`,"data-testid":e.testId,autoComplete:`off`,onSubmit:r=>{if(r.preventDefault(),e.busy||t.trim().length===0)return;let i=t;n(``),e.onSubmit(i)},children:[(0,x.jsx)(`h2`,{children:e.title}),(0,x.jsx)(`p`,{children:e.body}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsx)(`span`,{children:e.tokenLabel}),(0,x.jsx)(`input`,{type:`password`,autoComplete:`off`,spellCheck:!1,autoCapitalize:`none`,autoCorrect:`off`,value:t,disabled:e.busy,"aria-invalid":e.error?`true`:void 0,"aria-describedby":r,onChange:e=>n(e.currentTarget.value)}),(0,x.jsx)(`small`,{id:r,"data-tone":e.error?`error`:`hint`,children:e.error??e.tokenHelp})]}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(S,{tone:`primary`,type:`submit`,busy:e.busy,disabled:t.trim().length===0,children:e.submitLabel}),e.onLogout&&e.logoutLabel?(0,x.jsx)(S,{type:`button`,onClick:()=>{n(``),e.onLogout?.()},children:e.logoutLabel}):null]})]})}function _e(e){return(0,x.jsx)(se,{tone:`error`,title:e.title,body:e.body,action:(0,x.jsxs)(S,{onClick:e.onRecover,children:[(0,x.jsx)(te,{name:`schema`}),e.actionLabel]})})}var ve=new Map([{key:`app.title`,en:`mlxcel WebUI`,ko:`mlxcel 웹 UI`,test_id:`app-title`},{key:`app.subtitle`,en:`Local model control plane`,ko:`로컬 모델 제어판`,test_id:`app-subtitle`},{key:`nav.models`,en:`Models`,ko:`모델`,test_id:`nav-models`},{key:`nav.chat`,en:`Chat`,ko:`대화`,test_id:`nav-chat`},{key:`nav.activity`,en:`Activity`,ko:`활동`,test_id:`nav-activity`},{key:`nav.settings`,en:`Settings`,ko:`설정`,test_id:`nav-settings`},{key:`nav.gallery`,en:`Gallery`,ko:`갤러리`,test_id:`nav-gallery`},{key:`nav.primary`,en:`Primary navigation`,ko:`주 내비게이션`,test_id:`nav-primary`},{key:`nav.home`,en:`mlxcel home`,ko:`mlxcel 홈`,test_id:`nav-home`},{key:`toolbar.command`,en:`Command`,ko:`명령`,test_id:`toolbar-command`},{key:`toolbar.help`,en:`Keyboard help`,ko:`키보드 도움말`,test_id:`toolbar-help`},{key:`toolbar.menu`,en:`Open navigation`,ko:`내비게이션 열기`,test_id:`toolbar-menu`},{key:`connection.ready`,en:`Shell loaded; local API not connected`,ko:`셸 로드됨; 로컬 API 미연결`,test_id:`connection-ready`},{key:`connection.offline`,en:`Server connection is offline`,ko:`서버 연결이 오프라인입니다`,test_id:`connection-offline`},{key:`connection.prompt.title`,en:`Connect to the local WebUI API`,ko:`로컬 WebUI API에 연결하세요`,test_id:`connection-prompt-title`},{key:`connection.prompt.body`,en:`The shell is loaded, but catalog, chat and activity data wait for the authenticated local server connection.`,ko:`셸은 로드되었지만 카탈로그, 대화, 활동 데이터는 인증된 로컬 서버 연결을 기다립니다.`,test_id:`connection-prompt-body`},{key:`connection.prompt.detail`,en:`Start mlxcel-server with --webui, enter the terminal session key when prompted, then refresh this view.`,ko:`mlxcel-server를 --webui로 시작하고, 요청되면 터미널 세션 키를 입력한 뒤 이 화면을 새로고침하세요.`,test_id:`connection-prompt-detail`},{key:`connection.footer.connected`,en:`{mode} · {status} · v{version} · seq {sequence}`,ko:`{mode} · {status} · v{version} · seq {sequence}`,test_id:`connection-footer-connected`},{key:`connection.snapshot.pending`,en:`pending`,ko:`대기 중`,test_id:`connection-snapshot-pending`},{key:`connection.authenticated.title`,en:`Authenticated local API session`,ko:`인증된 로컬 API 세션`,test_id:`connection-authenticated-title`},{key:`connection.authenticated.body`,en:`This route is connected to the shared provider. Browsing does not load models or start inference; load, unload and chat actions remain explicit.`,ko:`이 경로는 공유 provider에 연결되어 있습니다. 탐색만으로 모델을 로드하거나 추론을 시작하지 않으며, 로드·언로드·대화 동작은 명시적으로 실행됩니다.`,test_id:`connection-authenticated-body`},{key:`connection.authenticated.detail`,en:`Backend {mode}; build {version}; state {status}; catalog {count}; operations {operations}; snapshot {sequence}.`,ko:`백엔드 {mode}; 빌드 {version}; 상태 {status}; 카탈로그 {count}; 작업 {operations}; 스냅샷 {sequence}.`,test_id:`connection-authenticated-detail`},{key:`connection.error.title`,en:`Provider connection needs attention`,ko:`Provider 연결 확인 필요`,test_id:`connection-error-title`},{key:`connection.error.stale`,en:`The server snapshot changed; retry to take a fresh catalog and operation snapshot before continuing.`,ko:`서버 스냅샷이 바뀌었습니다. 계속하기 전에 다시 시도해 새 카탈로그와 작업 스냅샷을 가져오세요.`,test_id:`connection-error-stale`},{key:`connection.error.forbidden`,en:`The authenticated session is not allowed to access this UI endpoint.`,ko:`인증된 세션이 이 UI 엔드포인트에 접근할 수 없습니다.`,test_id:`connection-error-forbidden`},{key:`connection.error.unauthorized`,en:`The server rejected the current session key. Sign in again with the latest terminal key.`,ko:`서버가 현재 세션 키를 거부했습니다. 터미널에 표시된 최신 키로 다시 로그인하세요.`,test_id:`connection-error-unauthorized`},{key:`connection.error.generic`,en:`Retry the shared provider snapshot before issuing any model control action.`,ko:`모델 제어 동작을 실행하기 전에 공유 provider 스냅샷을 다시 가져오세요.`,test_id:`connection-error-generic`},{key:`connection.status.idle`,en:`idle`,ko:`대기`,test_id:`connection-status-idle`},{key:`connection.status.bootstrapping`,en:`bootstrapping`,ko:`부트스트랩`,test_id:`connection-status-bootstrapping`},{key:`connection.status.ready`,en:`ready`,ko:`준비`,test_id:`connection-status-ready`},{key:`connection.status.streaming`,en:`streaming`,ko:`스트리밍`,test_id:`connection-status-streaming`},{key:`connection.status.polling`,en:`polling`,ko:`폴링`,test_id:`connection-status-polling`},{key:`connection.status.offline`,en:`offline`,ko:`오프라인`,test_id:`connection-status-offline`},{key:`connection.status.stale`,en:`stale`,ko:`낡음`,test_id:`connection-status-stale`},{key:`connection.status.unauthorized`,en:`unauthorized`,ko:`인증 실패`,test_id:`connection-status-unauthorized`},{key:`connection.status.forbidden`,en:`forbidden`,ko:`거부됨`,test_id:`connection-status-forbidden`},{key:`connection.status.schema_mismatch`,en:`schema mismatch`,ko:`스키마 불일치`,test_id:`connection-status-schema-mismatch`},{key:`connection.status.error`,en:`error`,ko:`오류`,test_id:`connection-status-error`},{key:`auth.login`,en:`WebUI session login`,ko:`WebUI 세션 로그인`,test_id:`auth-login`},{key:`models.title`,en:`Model library`,ko:`모델 라이브러리`,test_id:`models-title`},{key:`models.empty.title`,en:`No local models yet`,ko:`아직 로컬 모델이 없습니다`,test_id:`models-empty-title`},{key:`models.empty.body`,en:`Browse, download, and load models explicitly. The shell never autoloads a checkpoint.`,ko:`모델을 명시적으로 탐색, 다운로드, 로드하세요. 셸은 체크포인트를 자동 로드하지 않습니다.`,test_id:`models-empty-body`},{key:`models.long_name`,en:`Qwen3 Very Long Local Checkpoint Name With Mixed English and 한국어 모델 이름`,ko:`Qwen3 매우 긴 로컬 체크포인트 이름과 한국어 모델 이름`,test_id:`models-long-name`},{key:`models.unsupported.reason`,en:`Vision input is unavailable for this backend; chat remains text-only.`,ko:`이 백엔드에서는 비전 입력을 사용할 수 없어 대화는 텍스트 전용입니다.`,test_id:`models-unsupported-reason`},{key:`models.load`,en:`Load`,ko:`로드`,test_id:`models-load`},{key:`models.unload`,en:`Unload`,ko:`언로드`,test_id:`models-unload`},{key:`models.status.ready`,en:`Ready`,ko:`준비됨`,test_id:`models-status-ready`},{key:`models.status.unloaded`,en:`Unloaded`,ko:`언로드됨`,test_id:`models-status-unloaded`},{key:`models.status.failed`,en:`Failed`,ko:`실패`,test_id:`models-status-failed`},{key:`models.status.loading`,en:`Loading`,ko:`로드 중`,test_id:`models-status-loading`},{key:`models.status.draining`,en:`Draining`,ko:`drain 중`,test_id:`models-status-draining`},{key:`models.status.unloading`,en:`Unloading`,ko:`언로드 중`,test_id:`models-status-unloading`},{key:`models.delete.confirm.title`,en:`Delete model from cache?`,ko:`캐시에서 모델을 삭제할까요?`,test_id:`dialog-delete-model-title`},{key:`models.delete.confirm.body`,en:`Delete {model} from the managed cache. Loaded or non-cache models cannot be deleted.`,ko:`관리 캐시에서 {model} 모델을 삭제합니다. 로드 중이거나 캐시 모델이 아니면 삭제할 수 없습니다.`,test_id:`dialog-delete-model-body`},{key:`models.delete.confirm.token_label`,en:`Type DELETE to confirm`,ko:`확인하려면 DELETE를 입력하세요`,test_id:`dialog-delete-model-token`},{key:`models.unload.confirm.body`,en:`Unload {model} after active requests drain; browser tabs will keep their selected model.`,ko:`활성 요청이 비워진 뒤 {model} 모델을 언로드합니다. 브라우저 탭의 선택 모델은 유지됩니다.`,test_id:`dialog-unload-model-body`},{key:`chat.title`,en:`Chat`,ko:`대화`,test_id:`chat-title`},{key:`chat.placeholder`,en:`Type a message; IME composition is preserved.`,ko:`메시지를 입력하세요. IME 조합은 보존됩니다.`,test_id:`chat-placeholder`},{key:`chat.streaming.status`,en:`Streaming tokens from the selected ready model.`,ko:`선택한 준비 모델에서 토큰을 스트리밍 중입니다.`,test_id:`chat-streaming-status`},{key:`chat.reasoning`,en:`Reasoning`,ko:`추론`,test_id:`chat-reasoning`},{key:`chat.tool_call`,en:`Tool call preview`,ko:`도구 호출 미리보기`,test_id:`chat-tool-call`},{key:`downloads.cancel.confirm.body`,en:`Cancel this download only after the server acknowledges worker shutdown.`,ko:`서버가 작업자 중지를 확인한 뒤에만 이 다운로드를 취소합니다.`,test_id:`dialog-cancel-download-body`},{key:`activity.title`,en:`Activity`,ko:`활동`,test_id:`activity-title`},{key:`activity.progress.indeterminate`,en:`Downloaded {bytes}; total size is not known yet.`,ko:`{bytes} 다운로드됨; 전체 크기는 아직 알 수 없습니다.`,test_id:`activity-progress-indeterminate`},{key:`activity.sse_reset`,en:`Event stream reset; take a fresh snapshot before resuming updates.`,ko:`이벤트 스트림이 재설정되었습니다. 업데이트를 재개하기 전에 새 스냅샷을 가져오세요.`,test_id:`activity-sse-reset`},{key:`settings.title`,en:`Settings`,ko:`설정`,test_id:`settings-title`},{key:`settings.appearance`,en:`Appearance preferences`,ko:`화면 표시 설정`,test_id:`settings-appearance`},{key:`settings.theme`,en:`Theme`,ko:`테마`,test_id:`settings-theme`},{key:`settings.material`,en:`Material`,ko:`재질`,test_id:`settings-material`},{key:`settings.glass_intensity`,en:`Glass intensity`,ko:`글래스 강도`,test_id:`settings-glass-intensity`},{key:`settings.reduce_motion`,en:`Reduce motion`,ko:`동작 줄이기`,test_id:`settings-reduce-motion`},{key:`settings.reduce_transparency`,en:`Reduce transparency`,ko:`투명도 줄이기`,test_id:`settings-reduce-transparency`},{key:`settings.high_contrast`,en:`High contrast`,ko:`고대비`,test_id:`settings-high-contrast`},{key:`settings.locale`,en:`Language`,ko:`언어`,test_id:`settings-locale`},{key:`settings.partial_success`,en:`Some settings changed; fields controlled by CLI or the loaded worker were left unchanged.`,ko:`일부 설정만 변경되었습니다. CLI 또는 로드된 워커가 제어하는 필드는 변경되지 않았습니다.`,test_id:`settings-partial-success`},{key:`settings.browser_only`,en:`Browser appearance only`,ko:`브라우저 표시 설정 전용`,test_id:`settings-browser-only`},{key:`settings.browser_only.body`,en:`These preferences stay in this browser and do not claim server or worker settings changed.`,ko:`이 설정은 이 브라우저에만 남으며 서버나 워커 설정 변경을 의미하지 않습니다.`,test_id:`settings-browser-only-body`},{key:`settings.clear_history.confirm.body`,en:`Clear only the explicit local browser history store; API keys are never persisted there.`,ko:`명시적으로 켠 로컬 브라우저 기록 저장소만 지웁니다. API 키는 그곳에 저장하지 않습니다.`,test_id:`dialog-clear-history-body`},{key:`state.unauthorized.title`,en:`Authentication required`,ko:`인증이 필요합니다`,test_id:`state-unauthorized-title`},{key:`state.unauthorized.body`,en:`Enter the session key printed by the local server terminal.`,ko:`로컬 서버 터미널에 한 번 표시된 세션 키를 입력하세요.`,test_id:`state-unauthorized-body`},{key:`state.offline.title`,en:`Server offline`,ko:`서버 오프라인`,test_id:`state-offline-title`},{key:`state.offline.body`,en:`The shell is available, but model data waits for the local API.`,ko:`셸은 사용할 수 있지만 모델 데이터는 로컬 API를 기다립니다.`,test_id:`state-offline-body`},{key:`state.schema_mismatch.title`,en:`UI schema mismatch`,ko:`UI 스키마 불일치`,test_id:`state-schema-mismatch-title`},{key:`state.schema_mismatch.body`,en:`The shell loaded, but the server reports a different UI API schema version; refresh after updating the bundle or server.`,ko:`셸은 로드되었지만 서버가 다른 UI API 스키마 버전을 보고했습니다. 번들이나 서버를 업데이트한 뒤 새로고침하세요.`,test_id:`state-schema-mismatch-body`},{key:`command.title`,en:`Command palette`,ko:`명령 팔레트`,test_id:`command-title`},{key:`command.search`,en:`Search commands`,ko:`명령 검색`,test_id:`command-search`},{key:`command.no_results`,en:`No commands match this search.`,ko:`검색과 일치하는 명령이 없습니다.`,test_id:`command-no-results`},{key:`help.title`,en:`Keyboard shortcuts`,ko:`키보드 단축키`,test_id:`help-title`},{key:`help.body`,en:`Command opens search. Escape closes overlays. Brackets move navigation only while the sidebar has focus.`,ko:`Command는 검색을 엽니다. Escape는 오버레이를 닫습니다. 대괄호는 사이드바에 포커스가 있을 때만 내비게이션을 이동합니다.`,test_id:`help-body`},{key:`gallery.title`,en:`Design system gallery`,ko:`디자인 시스템 갤러리`,test_id:`gallery-title`},{key:`gallery.subtitle`,en:`Shared tokens, controls, states, and viewport fixtures for page implementers.`,ko:`페이지 구현자를 위한 공유 토큰, 컨트롤, 상태, 뷰포트 픽스처입니다.`,test_id:`gallery-subtitle`},{key:`gallery.long_cjk`,en:`Long English and 한국어 labels truncate with accessible full labels.`,ko:`긴 English 및 한국어 레이블은 접근 가능한 전체 레이블을 유지하며 줄임표 처리됩니다.`,test_id:`gallery-long-cjk`},{key:`model.selected.none`,en:`No model selected`,ko:`선택한 모델 없음`,test_id:`model-selected-none`},{key:`common.unavailable`,en:`Unavailable until server adapters connect`,ko:`서버 어댑터 연결 전에는 사용할 수 없습니다`,test_id:`common-unavailable`},{key:`common.cancel`,en:`Cancel`,ko:`취소`,test_id:`common-cancel`},{key:`common.delete`,en:`Delete`,ko:`삭제`,test_id:`common-delete`},{key:`common.retry`,en:`Retry`,ko:`다시 시도`,test_id:`common-retry`},{key:`common.reload`,en:`Reload`,ko:`새로고침`,test_id:`common-reload`},{key:`common.enter_key`,en:`Enter key`,ko:`키 입력`,test_id:`common-enter-key`},{key:`common.add_model`,en:`Add model`,ko:`모델 추가`,test_id:`common-add-model`},{key:`common.send`,en:`Send`,ko:`보내기`,test_id:`common-send`},{key:`routes.models.eyebrow`,en:`Local library`,ko:`로컬 라이브러리`,test_id:`routes-models-eyebrow`},{key:`routes.chat.eyebrow`,en:`Conversation`,ko:`대화`,test_id:`routes-chat-eyebrow`},{key:`routes.activity.eyebrow`,en:`Operations`,ko:`작업`,test_id:`routes-activity-eyebrow`},{key:`routes.settings.eyebrow`,en:`Browser only`,ko:`브라우저 전용`,test_id:`routes-settings-eyebrow`},{key:`adapters.pending.title`,en:`Local API is not connected yet`,ko:`로컬 API가 아직 연결되지 않았습니다`,test_id:`adapters-pending-title`},{key:`adapters.pending.body`,en:`This route shows the production shell only. Catalog, lifecycle, and runtime data will come from the shared typed client in the integration wave.`,ko:`이 경로는 프로덕션 셸만 보여줍니다. 카탈로그, 수명주기, 런타임 데이터는 통합 웨이브의 공유 typed client에서 제공됩니다.`,test_id:`adapters-pending-body`},{key:`chat.pending.body`,en:`Chat controls stay disabled until an authenticated ready model is selected by the shared state provider.`,ko:`공유 상태 provider가 인증된 준비 모델을 선택하기 전까지 대화 컨트롤은 비활성화됩니다.`,test_id:`chat-pending-body`},{key:`activity.empty.title`,en:`No active operations`,ko:`활성 작업 없음`,test_id:`activity-empty-title`},{key:`activity.empty.body`,en:`Loads, downloads, drains and SSE resets appear here after the lifecycle adapter connects.`,ko:`수명주기 어댑터가 연결된 뒤 로드, 다운로드, drain, SSE reset이 여기에 표시됩니다.`,test_id:`activity-empty-body`},{key:`gallery.tab.sections`,en:`Gallery sections`,ko:`갤러리 섹션`,test_id:`gallery-tab-sections`},{key:`gallery.tab.controls`,en:`Controls`,ko:`컨트롤`,test_id:`gallery-tab-controls`},{key:`gallery.tab.states`,en:`States`,ko:`상태`,test_id:`gallery-tab-states`},{key:`gallery.tab.data`,en:`Data display`,ko:`데이터 표시`,test_id:`gallery-tab-data`},{key:`gallery.controls.title`,en:`Buttons and fields`,ko:`버튼과 필드`,test_id:`gallery-controls-title`},{key:`gallery.controls.primary`,en:`Primary`,ko:`기본`,test_id:`gallery-controls-primary`},{key:`gallery.controls.secondary`,en:`Secondary`,ko:`보조`,test_id:`gallery-controls-secondary`},{key:`gallery.controls.danger`,en:`Danger`,ko:`위험`,test_id:`gallery-controls-danger`},{key:`gallery.controls.busy`,en:`Busy`,ko:`진행 중`,test_id:`gallery-controls-busy`},{key:`gallery.field.repo`,en:`Repository ID`,ko:`저장소 ID`,test_id:`gallery-field-repo`},{key:`gallery.field.repo_error`,en:`Use owner/name without a URL.`,ko:`URL 없이 owner/name 형식을 사용하세요.`,test_id:`gallery-field-repo-error`},{key:`gallery.field.repo_hint`,en:`Sample only; downloads require the later lifecycle adapter.`,ko:`샘플 전용입니다. 다운로드는 후속 수명주기 어댑터가 필요합니다.`,test_id:`gallery-field-repo-hint`},{key:`gallery.select.native`,en:`Native select combobox`,ko:`네이티브 select 콤보박스`,test_id:`gallery-select-native`},{key:`gallery.overlays.title`,en:`Overlays`,ko:`오버레이`,test_id:`gallery-overlays-title`},{key:`gallery.tooltip`,en:`Tooltips are descriptive only`,ko:`툴팁은 설명 전용입니다`,test_id:`gallery-tooltip`},{key:`gallery.dialog.open`,en:`Open dialog`,ko:`다이얼로그 열기`,test_id:`gallery-dialog-open`},{key:`gallery.states.load_failed`,en:`Load failed`,ko:`로드 실패`,test_id:`gallery-states-load-failed`},{key:`gallery.states.load_failed_body`,en:`The worker exited before becoming ready; retry after checking logs.`,ko:`워커가 준비되기 전에 종료되었습니다. 로그 확인 후 다시 시도하세요.`,test_id:`gallery-states-load-failed-body`},{key:`gallery.data.title`,en:`Model rows`,ko:`모델 행`,test_id:`gallery-data-title`},{key:`gallery.data.name`,en:`Name`,ko:`이름`,test_id:`gallery-data-name`},{key:`gallery.data.status`,en:`Status`,ko:`상태`,test_id:`gallery-data-status`},{key:`gallery.data.rate`,en:`Rate`,ko:`속도`,test_id:`gallery-data-rate`},{key:`gallery.data.inspector`,en:`Inspector`,ko:`인스펙터`,test_id:`gallery-data-inspector`},{key:`gallery.sample.reasoning_body`,en:`Reasoning text is visually separated from answer content.`,ko:`추론 텍스트는 답변 본문과 시각적으로 분리됩니다.`,test_id:`gallery-sample-reasoning-body`},{key:`gallery.sample.tool_preview`,en:`{ "tool": "display_only" }`,ko:`{ "tool": "표시_전용" }`,test_id:`gallery-sample-tool-preview`},{key:`common.close`,en:`Close`,ko:`닫기`,test_id:`common-close`},{key:`settings.theme.system`,en:`System`,ko:`시스템`,test_id:`settings-theme-system`},{key:`settings.theme.light`,en:`Light`,ko:`라이트`,test_id:`settings-theme-light`},{key:`settings.theme.dark`,en:`Dark`,ko:`다크`,test_id:`settings-theme-dark`},{key:`settings.material.glass`,en:`Glass`,ko:`글래스`,test_id:`settings-material-glass`},{key:`settings.material.tinted`,en:`Tinted`,ko:`틴트`,test_id:`settings-material-tinted`},{key:`settings.material.opaque`,en:`Opaque`,ko:`불투명`,test_id:`settings-material-opaque`},{key:`settings.locale.en`,en:`English`,ko:`영어`,test_id:`settings-locale-en`},{key:`settings.locale.ko`,en:`Korean`,ko:`한국어`,test_id:`settings-locale-ko`},{key:`settings.high_contrast.system`,en:`Follow system`,ko:`시스템 따르기`,test_id:`settings-high-contrast-system`},{key:`settings.high_contrast.on`,en:`On`,ko:`켬`,test_id:`settings-high-contrast-on`},{key:`settings.high_contrast.off`,en:`Off`,ko:`끔`,test_id:`settings-high-contrast-off`},{key:`gallery.issue`,en:`Issue #1843`,ko:`이슈 #1843`,test_id:`gallery-issue`},{key:`gallery.hover_focus`,en:`Hover or focus`,ko:`호버 또는 포커스`,test_id:`gallery-hover-focus`},{key:`gallery.download`,en:`Download`,ko:`다운로드`,test_id:`gallery-download`},{key:`gallery.lifecycle.samples`,en:`Lifecycle sample badges`,ko:`수명주기 샘플 배지`,test_id:`gallery-lifecycle-samples`},{key:`gallery.sample.caption`,en:`Sample model table`,ko:`샘플 모델 표`,test_id:`gallery-sample-caption`},{key:`gallery.sample.label`,en:`Sample fixture`,ko:`샘플 픽스처`,test_id:`gallery-sample-label`},{key:`login.token.label`,en:`Session key`,ko:`세션 키`,test_id:`login-token-label`},{key:`login.token.help`,en:`Use the key printed once by the local server. It stays in memory only.`,ko:`로컬 서버가 한 번 출력한 키를 사용하세요. 키는 메모리에만 유지됩니다.`,test_id:`login-token-help`},{key:`login.submit`,en:`Connect`,ko:`연결`,test_id:`login-submit`},{key:`login.logout`,en:`Clear key`,ko:`키 지우기`,test_id:`login-logout`},{key:`login.error.sample`,en:`Sample error: the key was not accepted by the local API.`,ko:`샘플 오류: 로컬 API가 키를 허용하지 않았습니다.`,test_id:`login-error-sample`},{key:`login.error.wrong_key`,en:`The session key was rejected. Copy the latest key printed by the local server terminal.`,ko:`세션 키가 거부되었습니다. 로컬 서버 터미널에 표시된 최신 키를 복사하세요.`,test_id:`login-error-wrong-key`},{key:`login.error.offline`,en:`Could not reach the local WebUI API. Check that mlxcel-server is still running with --webui.`,ko:`로컬 WebUI API에 연결할 수 없습니다. mlxcel-server가 --webui로 계속 실행 중인지 확인하세요.`,test_id:`login-error-offline`},{key:`login.error.forbidden`,en:`The key authenticated, but this UI endpoint is forbidden for the current server session.`,ko:`키 인증은 되었지만 현재 서버 세션에서 이 UI 엔드포인트가 금지되어 있습니다.`,test_id:`login-error-forbidden`},{key:`login.error.schema`,en:`The server response does not match this bundled UI schema. Reload after updating the server or bundle.`,ko:`서버 응답이 번들된 UI 스키마와 일치하지 않습니다. 서버나 번들을 업데이트한 뒤 새로고침하세요.`,test_id:`login-error-schema`},{key:`login.error.generic`,en:`The local API could not complete authentication. Retry with the latest terminal key.`,ko:`로컬 API 인증을 완료할 수 없습니다. 터미널에 표시된 최신 키로 다시 시도하세요.`,test_id:`login-error-generic`},{key:`gallery.delete_token`,en:`DELETE`,ko:`DELETE`,test_id:`gallery-delete-token`}].map(e=>[e.key,e]));function T(e,t,n={}){let r=ve.get(t);return r?r[e].replace(/\{(\w+)\}/g,(e,t)=>n[t]??`{${t}}`):t}function E(e){return ve.get(e)?.test_id??e.replaceAll(`.`,`-`)}var ye=[{id:`models`,key:`nav.models`,icon:`models`},{id:`chat`,key:`nav.chat`,icon:`chat`},{id:`activity`,key:`nav.activity`,icon:`activity`},{id:`settings`,key:`nav.settings`,icon:`settings`}];function be(e){let t=(0,b.useRef)(null),n=(0,b.useRef)(e.onCommand),r=(0,b.useRef)(e.onHelp),[i,a]=(0,b.useState)(!1);(0,b.useEffect)(()=>{n.current=e.onCommand,r.current=e.onHelp},[e.onCommand,e.onHelp]),(0,b.useEffect)(()=>{let e=e=>{let t=e.target,i=t instanceof HTMLElement&&(t.isContentEditable||[`INPUT`,`TEXTAREA`,`SELECT`].includes(t.tagName)),o=t instanceof HTMLElement&&!!t.closest(`dialog[open]`);e.isComposing||e.altKey||i||o||((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()===`k`&&(e.preventDefault(),a(!1),n.current()),(e.key===`?`||e.key===`/`&&e.shiftKey)&&(e.preventDefault(),a(!1),r.current()))};return window.addEventListener(`keydown`,e),()=>window.removeEventListener(`keydown`,e)},[]);let o=n=>{let r=ye[(ye.findIndex(t=>t.id===e.route)+n+ye.length)%ye.length];e.onRouteChange(r.id),requestAnimationFrame(()=>t.current?.querySelector(`a[href="#${r.id}"]`)?.focus())},s=e=>{(e.key===`[`||e.key===`]`)&&(e.altKey||e.metaKey||e.ctrlKey||e.nativeEvent.isComposing||(e.preventDefault(),o(e.key===`]`?1:-1)))},c=t=>{e.onRouteChange(t),a(!1)};return(0,x.jsxs)(`div`,{className:`app-shell`,children:[(0,x.jsx)(xe,{locale:e.locale,route:e.route,onRouteChange:c,onKeyDown:s,ref:t,className:`app-sidebar desktop-sidebar material-glass`,connectionLabel:e.connectionLabel,connectionState:e.connectionState}),(0,x.jsx)(ue,{open:i,title:T(e.locale,`nav.primary`),onClose:()=>a(!1),closeLabel:T(e.locale,`common.close`),testId:`mobile-nav-sheet`,children:(0,x.jsx)(xe,{locale:e.locale,route:e.route,onRouteChange:c,onKeyDown:s,className:`app-sidebar sheet-sidebar`,connectionLabel:e.connectionLabel,connectionState:e.connectionState})}),(0,x.jsxs)(`main`,{className:`app-main`,"aria-labelledby":`app-title`,children:[(0,x.jsxs)(`header`,{className:`app-toolbar material-glass`,children:[(0,x.jsx)(C,{className:`mobile-menu-button`,label:T(e.locale,`toolbar.menu`),icon:`menu`,onClick:()=>a(!0),"data-testid":E(`toolbar.menu`)}),(0,x.jsxs)(`div`,{className:`toolbar-title`,children:[(0,x.jsx)(`p`,{id:`app-title`,"data-testid":E(`app.title`),children:T(e.locale,`app.title`)}),(0,x.jsx)(`span`,{"data-testid":E(`app.subtitle`),children:T(e.locale,`app.subtitle`)})]}),(0,x.jsx)(`div`,{className:`selected-model`,title:e.selectedModel,children:e.selectedModel}),(0,x.jsxs)(`div`,{className:`toolbar-actions`,children:[(0,x.jsx)(C,{label:T(e.locale,`toolbar.command`),icon:`command`,onClick:e.onCommand,"data-testid":E(`toolbar.command`)}),(0,x.jsx)(C,{label:T(e.locale,`toolbar.help`),icon:`help`,onClick:e.onHelp,"data-testid":E(`toolbar.help`)})]})]}),(0,x.jsxs)(`div`,{className:`app-content-grid ${e.inspector?`has-inspector`:``}`.trim(),children:[(0,x.jsx)(`section`,{className:`app-content`,children:e.children}),e.inspector]})]})]})}var xe=b.forwardRef((e,t)=>(0,x.jsxs)(`aside`,{className:e.className,"aria-label":T(e.locale,`nav.primary`),onKeyDown:e.onKeyDown,ref:t,tabIndex:-1,children:[(0,x.jsx)(`a`,{className:`brand-mark`,href:`#models`,"aria-label":T(e.locale,`nav.home`),onClick:t=>{t.preventDefault(),e.onRouteChange(`models`)},children:`mx`}),(0,x.jsx)(`nav`,{className:`app-nav`,children:ye.map(t=>(0,x.jsxs)(`a`,{href:`#${t.id}`,"aria-label":T(e.locale,t.key),"aria-current":e.route===t.id?`page`:void 0,"data-testid":E(t.key),onClick:n=>{n.preventDefault(),e.onRouteChange(t.id)},children:[(0,x.jsx)(te,{name:t.icon}),(0,x.jsx)(`span`,{className:`app-nav-label`,children:T(e.locale,t.key)})]},t.id))}),(0,x.jsxs)(`footer`,{children:[(0,x.jsx)(`span`,{className:`connection-dot`,"data-state":e.connectionState,"aria-hidden":`true`}),(0,x.jsx)(`span`,{"data-testid":E(`connection.ready`),children:e.connectionLabel})]})]}));xe.displayName=`Sidebar`;var Se={theme:`system`,material:`glass`,glassIntensity:35,reduceMotion:!1,reduceTransparency:!1,highContrast:`system`,locale:`en`},Ce=`mlxcel.webui.appearance`;function D(e){let t=typeof e==`number`&&Number.isFinite(e)?e:Se.glassIntensity;return Math.max(0,Math.min(100,Math.round(t)))}function O(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function we(e){return e===`system`||e===`light`||e===`dark`}function Te(e){return e===`glass`||e===`tinted`||e===`opaque`}function Ee(e){return e===`system`||e===`on`||e===`off`}function De(e){return Ee(e)?e:e===!0?`on`:e===!1?`off`:Se.highContrast}function Oe(e){return e===`en`||e===`ko`}function k(){if(typeof window>`u`)return Se;let e;try{e=window.localStorage.getItem(Ce)}catch{return Se}if(!e)return Se;try{let t=JSON.parse(e);return O(t)?{theme:we(t.theme)?t.theme:Se.theme,material:Te(t.material)?t.material:Se.material,glassIntensity:D(t.glassIntensity),reduceMotion:t.reduceMotion===!0,reduceTransparency:t.reduceTransparency===!0,highContrast:De(t.highContrast),locale:Oe(t.locale)?t.locale:Se.locale}:Se}catch{return Se}}function ke(e){try{window.localStorage.setItem(Ce,JSON.stringify(e))}catch{}}function Ae(){return typeof CSS>`u`||typeof CSS.supports!=`function`?!1:CSS.supports(`backdrop-filter: blur(1px)`)||CSS.supports(`-webkit-backdrop-filter: blur(1px)`)}function je(e,t){e.dataset.theme=t.theme,e.dataset.material=t.reduceTransparency?`opaque`:t.material,e.dataset.glassIntensity=String(D(t.glassIntensity)),e.dataset.reduceMotion=String(t.reduceMotion),e.dataset.reduceTransparency=String(t.reduceTransparency),e.dataset.highContrast=t.highContrast,e.dataset.backdropFilter=Ae()?`supported`:`unsupported`,e.lang=t.locale}var Me=e=>e===`ko`?`ko-KR`:`en-US`;function Ne(e,t){if(!Number.isFinite(e)||e<0)return t===`ko`?`알 수 없음`:`unknown`;let n=[`B`,`KiB`,`MiB`,`GiB`,`TiB`],r=e,i=0;for(;r>=1024&&i<n.length-1;)r/=1024,i+=1;let a=i===0?0:1;return`${new Intl.NumberFormat(Me(t),{maximumFractionDigits:a}).format(r)} ${n[i]}`}function Pe(e,t){return e===null?t===`ko`?`아직 측정되지 않음`:`not yet measured`:`${new Intl.NumberFormat(Me(t),{maximumFractionDigits:1}).format(e)} tokens/s`}function Fe(e){let[t,n]=(0,b.useState)(!1),[r,i]=(0,b.useState)(`controls`),[a,o]=(0,b.useState)(`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`),[s,c]=(0,b.useState)(`ready`);return(0,x.jsxs)(`div`,{className:`screen-stack gallery-screen`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:T(e.locale,`gallery.issue`)}),(0,x.jsx)(`h1`,{"data-testid":E(`gallery.title`),children:T(e.locale,`gallery.title`)}),(0,x.jsx)(`p`,{"data-testid":E(`gallery.subtitle`),children:T(e.locale,`gallery.subtitle`)})]}),(0,x.jsx)(fe,{active:r,onChange:i,label:T(e.locale,`gallery.tab.sections`),tabs:[{id:`controls`,label:T(e.locale,`gallery.tab.controls`),panel:(0,x.jsx)(Ie,{locale:e.locale,value:a,onValueChange:o,selectValue:s,onSelectChange:c,onOpenDialog:()=>n(!0)})},{id:`states`,label:T(e.locale,`gallery.tab.states`),panel:(0,x.jsx)(Le,{locale:e.locale})},{id:`data`,label:T(e.locale,`gallery.tab.data`),panel:(0,x.jsx)(Re,{locale:e.locale})}]}),(0,x.jsxs)(le,{open:t,title:T(e.locale,`models.delete.confirm.title`),onClose:()=>n(!1),testId:`gallery-dialog`,closeLabel:T(e.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":E(`models.delete.confirm.body`),children:T(e.locale,`models.delete.confirm.body`,{model:a})}),(0,x.jsx)(w,{label:T(e.locale,`models.delete.confirm.token_label`),value:``,placeholder:T(e.locale,`gallery.delete_token`)}),(0,x.jsxs)(`div`,{className:`dialog-actions`,children:[(0,x.jsx)(S,{onClick:()=>n(!1),children:T(e.locale,`common.cancel`)}),(0,x.jsx)(S,{tone:`danger`,"data-autofocus":!0,children:T(e.locale,`common.delete`)})]})]})]})}function Ie(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsxs)(`article`,{className:`surface-card`,children:[(0,x.jsx)(`h2`,{children:T(e.locale,`gallery.controls.title`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(S,{tone:`primary`,children:T(e.locale,`gallery.controls.primary`)}),(0,x.jsx)(S,{children:T(e.locale,`gallery.controls.secondary`)}),(0,x.jsx)(S,{tone:`danger`,children:T(e.locale,`gallery.controls.danger`)}),(0,x.jsx)(S,{busy:!0,children:T(e.locale,`gallery.controls.busy`)})]}),(0,x.jsx)(w,{label:T(e.locale,`gallery.field.repo`),value:e.value,onChange:e.onValueChange,error:T(e.locale,`gallery.field.repo_error`),hint:T(e.locale,`gallery.field.repo_hint`),testId:`gallery-field`}),(0,x.jsx)(re,{label:T(e.locale,`gallery.select.native`),value:e.selectValue,onChange:e.onSelectChange,options:[{value:`ready`,label:T(e.locale,`models.status.ready`)},{value:`unloaded`,label:T(e.locale,`models.status.unloaded`)}]})]}),(0,x.jsxs)(`article`,{className:`surface-card`,children:[(0,x.jsx)(`h2`,{children:T(e.locale,`gallery.overlays.title`)}),(0,x.jsx)(`p`,{children:T(e.locale,`gallery.long_cjk`)}),(0,x.jsxs)(`div`,{className:`control-row`,children:[(0,x.jsx)(de,{label:T(e.locale,`gallery.tooltip`),children:(0,x.jsx)(S,{children:T(e.locale,`gallery.hover_focus`)})}),(0,x.jsx)(S,{tone:`primary`,onClick:e.onOpenDialog,children:T(e.locale,`gallery.dialog.open`)})]})]})]})}function Le(e){return(0,x.jsxs)(`div`,{className:`gallery-grid`,children:[(0,x.jsx)(ge,{title:T(e.locale,`state.unauthorized.title`),body:T(e.locale,`state.unauthorized.body`),tokenLabel:T(e.locale,`login.token.label`),tokenHelp:T(e.locale,`login.token.help`),submitLabel:T(e.locale,`login.submit`),logoutLabel:T(e.locale,`login.logout`),onSubmit:()=>void 0,onLogout:()=>void 0,testId:`gallery-login`}),(0,x.jsx)(_e,{title:T(e.locale,`state.schema_mismatch.title`),body:T(e.locale,`state.schema_mismatch.body`),actionLabel:T(e.locale,`common.reload`),onRecover:()=>void 0}),(0,x.jsx)(oe,{title:T(e.locale,`models.empty.title`),body:T(e.locale,`models.empty.body`),action:(0,x.jsx)(S,{tone:`primary`,children:T(e.locale,`common.add_model`)}),testId:`gallery-empty`}),(0,x.jsx)(se,{tone:`warning`,title:T(e.locale,`gallery.states.load_failed`),body:T(e.locale,`gallery.states.load_failed_body`),action:(0,x.jsx)(S,{children:T(e.locale,`common.retry`)})}),(0,x.jsx)(ae,{label:T(e.locale,`gallery.download`),detail:T(e.locale,`activity.progress.indeterminate`,{bytes:Ne(67108864,e.locale)})}),(0,x.jsxs)(he,{label:T(e.locale,`gallery.lifecycle.samples`),children:[(0,x.jsx)(`li`,{children:(0,x.jsx)(ie,{state:`loading`,children:T(e.locale,`models.status.loading`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(ie,{state:`draining`,children:T(e.locale,`models.status.draining`)})}),(0,x.jsx)(`li`,{children:(0,x.jsx)(ie,{state:`unloading`,children:T(e.locale,`models.status.unloading`)})})]})]})}function Re(e){return(0,x.jsxs)(`div`,{className:`gallery-grid dense-gallery`,children:[(0,x.jsxs)(`article`,{className:`surface-card table-card`,tabIndex:0,"aria-label":T(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`h2`,{children:T(e.locale,`gallery.data.title`)}),(0,x.jsxs)(me,{caption:T(e.locale,`gallery.sample.caption`),children:[(0,x.jsx)(`thead`,{children:(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`th`,{children:T(e.locale,`gallery.data.name`)}),(0,x.jsx)(`th`,{children:T(e.locale,`gallery.data.status`)}),(0,x.jsx)(`th`,{children:T(e.locale,`gallery.data.rate`)})]})}),(0,x.jsxs)(`tbody`,{children:[(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:T(e.locale,`models.long_name`),"aria-label":T(e.locale,`models.long_name`),children:T(e.locale,`models.long_name`)})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(ie,{state:`ready`,children:T(e.locale,`models.status.ready`)})}),(0,x.jsx)(`td`,{children:Pe(39.4,e.locale)})]}),(0,x.jsxs)(`tr`,{children:[(0,x.jsx)(`td`,{children:(0,x.jsx)(`span`,{className:`truncate`,title:`granite-4.0-h-tiny-4bit`,"aria-label":`granite-4.0-h-tiny-4bit`,children:`granite-4.0-h-tiny-4bit`})}),(0,x.jsx)(`td`,{children:(0,x.jsx)(ie,{state:`unloaded`,children:T(e.locale,`models.status.unloaded`)})}),(0,x.jsx)(`td`,{children:Pe(null,e.locale)})]})]})]})]}),(0,x.jsxs)(pe,{title:T(e.locale,`gallery.data.inspector`),children:[(0,x.jsx)(`p`,{className:`sample-label`,children:T(e.locale,`gallery.sample.label`)}),(0,x.jsx)(`p`,{children:T(e.locale,`models.unsupported.reason`)}),(0,x.jsx)(ie,{state:`failed`,children:T(e.locale,`models.status.failed`)}),(0,x.jsx)(`hr`,{}),(0,x.jsx)(`p`,{children:T(e.locale,`gallery.sample.reasoning_body`)}),(0,x.jsx)(`code`,{children:T(e.locale,`gallery.sample.tool_preview`)})]})]})}var ze=1048576,Be=class{decoder=new TextDecoder(`utf-8`);maxFrameBytes;onMessage;onDone;buffered=``;eventName=``;dataLines=[];id=null;retry=null;frameBytes=0;ended=!1;get done(){return this.ended}constructor(e){this.maxFrameBytes=e.maxFrameBytes??ze,this.onMessage=e.onMessage,this.onDone=e.onDone}push(e){if(!this.ended){if(this.frameBytes+=e.byteLength,this.frameBytes>this.maxFrameBytes)throw Error(`SSE frame exceeded the configured byte limit.`);this.buffered+=this.decoder.decode(e,{stream:!0}),this.drainLines(!1)}}close(){if(this.ended)return;let e=this.decoder.decode();e.length>0&&(this.buffered+=e),this.drainLines(!0),this.buffered.length>0&&(this.consumeLine(this.buffered),this.buffered=``),this.dispatch()}drainLines(e){for(;this.buffered.length>0;){let t=this.buffered.indexOf(`
`),n=this.buffered.indexOf(`\r`),r;if(r=t===-1?n:n===-1?t:Math.min(t,n),r===-1||this.buffered[r]===`\r`&&this.buffered[r+1]===void 0&&!e)break;let i=this.buffered.slice(0,r),a=this.buffered[r]===`\r`&&this.buffered[r+1]===`
`?r+2:r+1;this.buffered=this.buffered.slice(a),this.consumeLine(i)}e&&this.buffered.length===0&&this.dispatch()}consumeLine(e){if(e.length===0){this.dispatch();return}if(e.startsWith(`:`))return;let t=e.indexOf(`:`),n=t===-1?e:e.slice(0,t),r=t===-1?``:e.slice(t+1),i=r.startsWith(` `)?r.slice(1):r;n===`event`?this.eventName=i:n===`data`?this.dataLines.push(i):n===`id`?this.id=i:n===`retry`&&/^\d+$/.test(i)&&(this.retry=Number(i))}dispatch(){if(this.dataLines.length===0){this.eventName=``,this.retry=null,this.frameBytes=0;return}let e=this.dataLines.join(`
`),t=this.eventName.length===0?`message`:this.eventName;if(this.eventName=``,this.dataLines=[],this.frameBytes=0,e===`[DONE]`){this.ended=!0,this.onDone?.();return}this.onMessage({event:t,data:e,id:this.id,retry:this.retry}),this.retry=null}};function Ve(e){let t=e??He();if(t===``)return``;if(!t.startsWith(`/`)||t.startsWith(`//`))throw Error(`WebUI API base must be a same-origin absolute path.`);if(t.includes(`://`)||t.includes(`?`)||t.includes(`#`)||t.includes(`\\`))throw Error(`WebUI API base must not contain an origin, query, hash, or backslash.`);let n=t.split(`/`).filter(e=>e.length>0);if(n.some(e=>e===`.`||e===`..`))throw Error(`WebUI API base must not contain dot path segments.`);return`/${n.map(encodeURIComponent).join(`/`)}`}function He(){if(typeof document>`u`)return``;let e=document.querySelector(`meta[name="mlxcel-ui-api-base"]`)?.content;if(e!==void 0&&e.length>0)return e;let t=document.querySelector(`base`)?.getAttribute(`href`);if(t==null||t.length===0)return``;let n=new URL(t,window.location.origin);if(n.origin!==window.location.origin||n.search.length>0||n.hash.length>0)throw Error(`WebUI document base must stay on the current origin.`);return n.pathname.replace(/\/webui\/?$/,``)}function Ue(e,t,n){if(!t.startsWith(`/ui-api/v1/`)&&t!==`/v1/chat/completions`&&t!==`/v1/responses`)throw Error(`WebUI client paths must stay under /ui-api/v1/ or the approved inference stream endpoints.`);let r=new URLSearchParams;for(let[e,t]of Object.entries(n??{}))t!=null&&r.set(e,String(t));return`${e}${t}${r.size===0?``:`?${r.toString()}`}`}function We(e){if(e.length===0)throw Error(`Opaque path segment must not be empty.`);return encodeURIComponent(e)}var Ge=`{
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
          "settings"
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
      }
    }
  }
}
`,A=class extends Error{path;constructor(e,t){super(`${e}: ${t}`),this.path=e,this.name=`ValidationError`}},Ke=Ye(Ge);function qe(e,t,n=`$`){let r=ot(Ke,[`components`,`schemas`],`#/components/schemas`)[e];if(r===void 0)throw new A(n,`unknown schema ${e}`);Xe(r,t,n)}function Je(e){return JSON.parse(e)}function Ye(e){return at(Je(e),`$contract`)}function Xe(e,t,n){let r=at(e,`${n}#schema`);if(typeof r.$ref==`string`){Xe(rt(r.$ref),t,n);return}if(r.anyOf!==void 0){if(!Array.isArray(r.anyOf)||!r.anyOf.some(e=>it(e,t,n)))throw new A(n,`did not match any allowed schema`);return}if(r.oneOf!==void 0){if(!Array.isArray(r.oneOf))throw new A(n,`schema oneOf must be an array`);let e=r.oneOf.filter(e=>it(e,t,n)).length;if(e!==1)throw new A(n,`matched ${e} oneOf schemas`);return}if(r.const!==void 0&&t!==r.const)throw new A(n,`expected constant ${String(r.const)}`);if(Array.isArray(r.enum)&&!r.enum.some(e=>e===t))throw new A(n,`unexpected enum value ${String(t)}`);let i=r.type;if(Array.isArray(i)){if(!i.some(e=>nt(e,t)))throw new A(n,`expected one of ${i.join(`, `)}`)}else if(typeof i==`string`&&!nt(i,t))throw new A(n,`expected ${i}`);typeof t==`string`&&$e(r,t,n),typeof t==`number`&&tt(r,t,n),Array.isArray(t)&&Qe(r,t,n),st(t)&&Ze(r,t,n)}function Ze(e,t,n){let r=ct(e.required,`${n}#schema.required`,!1),i=e.properties===void 0?{}:at(e.properties,`${n}#schema.properties`);for(let e of r)if(!(e in t))throw new A(`${n}.${e}`,`missing required property`);if(e.maxProperties!==void 0&&Object.keys(t).length>lt(e.maxProperties,`${n}#schema.maxProperties`))throw new A(n,`too many object properties`);for(let[r,a]of Object.entries(t)){let t=i[r];if(t!==void 0)Xe(t,a,`${n}.${r}`);else if(e.additionalProperties===!1)throw new A(`${n}.${r}`,`unexpected property`);else st(e.additionalProperties)&&Xe(e.additionalProperties,a,`${n}.${r}`)}}function Qe(e,t,n){if(e.minItems!==void 0&&t.length<lt(e.minItems,`${n}#schema.minItems`))throw new A(n,`array shorter than minimum`);if(e.maxItems!==void 0&&t.length>lt(e.maxItems,`${n}#schema.maxItems`))throw new A(n,`array longer than maximum`);e.items!==void 0&&t.forEach((t,r)=>Xe(e.items,t,`${n}[${r}]`))}function $e(e,t,n){if(e.minLength!==void 0&&t.length<lt(e.minLength,`${n}#schema.minLength`))throw new A(n,`string shorter than minimum`);if(e.maxLength!==void 0&&t.length>lt(e.maxLength,`${n}#schema.maxLength`))throw new A(n,`string longer than maximum`);if(typeof e.pattern==`string`&&!new RegExp(e.pattern,`u`).test(t))throw new A(n,`string does not match pattern`);if(e.format===`date-time`&&!et(t))throw new A(n,`invalid RFC3339 date-time`)}function et(e){return/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(e)?!Number.isNaN(Date.parse(e)):!1}function tt(e,t,n){if(!Number.isFinite(t))throw new A(n,`expected finite number`);if(e.type===`integer`&&!Number.isInteger(t))throw new A(n,`expected integer`);if(e.minimum!==void 0&&t<lt(e.minimum,`${n}#schema.minimum`))throw new A(n,`number below minimum`);if(e.maximum!==void 0&&t>lt(e.maximum,`${n}#schema.maximum`))throw new A(n,`number above maximum`)}function nt(e,t){return e===`null`?t===null:e===`array`?Array.isArray(t):e===`object`?st(t):e===`integer`?typeof t==`number`&&Number.isInteger(t):e===`number`?typeof t==`number`&&Number.isFinite(t):typeof t===e}function rt(e){if(!e.startsWith(`#/`))throw new A(`$ref`,`unsupported external ref ${e}`);return e.slice(2).split(`/`).map(e=>e.replace(/~1/g,`/`).replace(/~0/g,`~`)).reduce((t,n)=>at(t,e)[n],Ke)}function it(e,t,n){try{return Xe(e,t,n),!0}catch(e){if(e instanceof A)return!1;throw e}}function at(e,t){if(!st(e))throw new A(t,`expected object`);return e}function ot(e,t,n){return t.reduce((e,t)=>at(e,n)[t],e)}function st(e){return typeof e==`object`&&!!e&&!Array.isArray(e)}function ct(e,t,n){if(e===void 0&&!n)return[];if(!Array.isArray(e)||!e.every(e=>typeof e==`string`))throw new A(t,`expected string array`);return e}function lt(e,t){if(typeof e!=`number`||!Number.isFinite(e))throw new A(t,`expected numeric schema bound`);return e}function ut(e,t,n){return qe(e,t,n),t}function dt(e,t=`$`){return ut(`ErrorEnvelope`,e,t)}function ft(e,t=`$`){return ut(`BootstrapResponse`,e,t)}function pt(e,t=`$`){return ut(`CatalogEntry`,e,t)}function mt(e,t=`$`){return ut(`CatalogListResponse`,e,t)}function ht(e,t=`$`){return ut(`OperationsListResponse`,e,t)}function gt(e,t=`$`){return ut(`Operation`,e,t)}function _t(e,t=`$`){return ut(`OperationAccepted`,e,t)}function vt(e,t=`$`){return ut(`RuntimeSnapshot`,e,t)}function yt(e,t=`$`){return ut(`UiEvent`,e,t)}var bt=class extends Error{status;envelope;constructor(e,t,n){super(n??t?.error.message??`WebUI request failed with HTTP ${e}`),this.status=e,this.envelope=t,this.name=`WebUiHttpError`}},xt=2097152,St=class{apiBase;fetchImpl;bearerToken=null;onUnauthorized;controllers=new Set;constructor(e={}){this.apiBase=Ve(e.apiBase),this.fetchImpl=e.fetchImpl??fetch.bind(globalThis),this.onUnauthorized=e.onUnauthorized}setBearerToken(e){this.bearerToken=e}abortAll(){for(let e of this.controllers)e.abort();this.controllers.clear()}async bootstrap(e){return this.request(`/ui-api/v1/bootstrap`,ft,{method:`GET`,signal:e})}async catalog(e={},t){return this.request(`/ui-api/v1/catalog`,mt,{method:`GET`,query:{...e},signal:t})}async catalogEntry(e,t){return this.request(`/ui-api/v1/catalog/${We(e)}`,pt,{method:`GET`,signal:t})}async refreshCatalog(e,t){return this.request(`/ui-api/v1/catalog/refresh`,_t,{method:`POST`,body:{idempotency_key:e},signal:t})}async modelAction(e,t){return this.request(`/ui-api/v1/model-actions`,_t,{method:`POST`,body:e,signal:t})}async download(e,t){return this.request(`/ui-api/v1/downloads`,_t,{method:`POST`,body:e,signal:t})}async removeModel(e,t){return this.request(`/ui-api/v1/model-removals`,_t,{method:`POST`,body:e,signal:t})}async operationsPage(e={},t){return this.request(`/ui-api/v1/operations`,ht,{method:`GET`,query:e,signal:t})}async operations(e){let t=[],n,r=null;for(;;){let i=await this.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null)r=i.server_instance_id;else if(i.server_instance_id!==r)throw Error(`Operations pagination crossed a server restart; refresh the authoritative snapshot.`);if(t.push(...i.items),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operation(e,t){return this.request(`/ui-api/v1/operations/${We(e)}`,gt,{method:`GET`,signal:t})}async cancelOperation(e,t){return this.request(`/ui-api/v1/operations/${We(e)}/cancel`,_t,{method:`POST`,body:{},signal:t})}async runtime(e,t){let n=await this.request(`/ui-api/v1/runtime`,vt,{method:`GET`,query:{model_id:e,autoload:!1},signal:t});if(n.model_id!==e)throw Error(`Runtime response model_id did not match the requested model.`);return n}async events(e,t,n){let r=n?.serverInstanceId!==void 0&&n.serverInstanceId!==null&&n.afterSequence!==null&&n.afterSequence!==void 0,i=r?{server_instance_id:n.serverInstanceId,after_sequence:n.afterSequence}:void 0,a=new Headers({Accept:`text/event-stream`});!r&&n?.lastEventId!==void 0&&n.lastEventId!==null&&a.set(`Last-Event-ID`,n.lastEventId),await this.sse(Ue(this.apiBase,`/ui-api/v1/events`,i),{method:`GET`,signal:t,headers:a},{onDone:e.onDone,onFrame:t=>{t.retry!==null&&e.onRetryAfter?.(t.retry),t.data.length>0&&e.onEvent(yt(Je(t.data)))}})}async chatCompletions(e,t,n,r){await this.sse(Ue(this.apiBase,`/v1/chat/completions`,{autoload:!1}),{method:`POST`,body:Tt(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async responses(e,t,n,r){await this.sse(Ue(this.apiBase,`/v1/responses`,{autoload:!1}),{method:`POST`,body:Tt(t,e),signal:r,headers:{Accept:`text/event-stream`}},n)}async request(e,t,n){let r=await this.fetchWithAuth(Ue(this.apiBase,e,n.query),n);try{return r.response.ok||await this.throwHttp(r.response,r.signal),t(Je(await this.readBody(r.response,r.signal)))}finally{this.controllers.delete(r.controller),r.cleanup()}}async sse(e,t,n){let r=await this.fetchWithAuth(e,t);try{let e=r.response;if(e.ok||await this.throwHttp(e,r.signal),e.body===null)throw Error(`WebUI event stream response has no body.`);let t=new Be({onDone:n.onDone,onMessage:n.onFrame}),i=e.body.getReader();try{for(;;){let e=await Et(i,r.signal);if(e.done)break;t.push(e.value)}if(t.close(),!t.done&&r.signal.aborted!==!0)throw Error(`WebUI event stream ended before a DONE frame.`)}catch(e){throw await i.cancel().catch(()=>void 0),e}finally{i.releaseLock()}}finally{this.controllers.delete(r.controller),r.cleanup()}}async fetchWithAuth(e,t){let n=new AbortController,{signal:r,cleanup:i}=Dt(n,t.signal);this.controllers.add(n);let a=new Headers(t.headers);a.set(`Accept`,a.get(`Accept`)??`application/json`),this.bearerToken!==null&&a.set(`Authorization`,`Bearer ${this.bearerToken}`),t.body!==void 0&&a.set(`Content-Type`,`application/json`);try{return{response:await this.fetchImpl(e,{method:t.method,headers:a,body:t.body===void 0?void 0:JSON.stringify(t.body),signal:r,credentials:`same-origin`,cache:`no-store`,redirect:`error`}),controller:n,signal:r,cleanup:i}}catch(e){throw this.controllers.delete(n),i(),e}}async throwHttp(e,t){let n=this.bearerToken;e.status===401&&(this.bearerToken=null,this.abortAll(),this.onUnauthorized?.());let r=null;try{let n=await this.readBody(e,t);n.length>0&&(r=dt(Je(n)))}catch{r=null}throw new bt(e.status,Ct(r,n),wt(r?.error.message??`WebUI request failed with HTTP ${e.status}`,n))}async readBody(e,t){if(e.body===null)return``;let n=e.body.getReader(),r=[],i=0;try{for(;;){let e=await Et(n,t);if(e.done)break;if(i+=e.value.byteLength,i>xt)throw Error(`WebUI JSON response exceeded the configured byte limit.`);r.push(e.value)}}catch(e){throw await n.cancel().catch(()=>void 0),e}finally{n.releaseLock()}let a=new Uint8Array(i),o=0;for(let e of r)a.set(e,o),o+=e.byteLength;return new TextDecoder().decode(a)}};function Ct(e,t){return e===null?null:{...e,error:{...e.error,message:wt(e.error.message,t),field_errors:e.error.field_errors?.map(e=>({...e,message:wt(e.message,t)}))}}}function wt(e,t){let n=e.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`).replace(/token[=:]\s*\S+/gi,`token=[redacted]`);return t!==null&&t.length>0&&(n=n.split(t).join(`[redacted]`)),n}function Tt(e,t){if(t.length===0)throw Error(`Inference model id is required for streaming inference.`);if(typeof e!=`object`||!e||Array.isArray(e))throw Error(`Streaming inference bodies must be JSON objects.`);return{...e,model:t,stream:!0}}async function Et(e,t){if(t.aborted)throw await e.cancel().catch(()=>void 0),Ot();let n=null;try{return await Promise.race([e.read(),new Promise((r,i)=>{n=()=>{e.cancel().catch(()=>void 0),i(Ot())},t.addEventListener(`abort`,n,{once:!0})})])}finally{n!==null&&t.removeEventListener(`abort`,n)}}function Dt(e,t){if(t===void 0)return{signal:e.signal,cleanup:()=>void 0};let n=()=>e.abort(t.reason);return t.aborted?n():t.addEventListener(`abort`,n,{once:!0}),{signal:e.signal,cleanup:()=>t.removeEventListener(`abort`,n)}}function Ot(){return typeof DOMException==`function`?new DOMException(`The WebUI request was aborted.`,`AbortError`):Error(`The WebUI request was aborted.`)}function kt(e){return e instanceof bt?e.status===401?`wrong-key`:e.status===403?`forbidden`:`generic`:e instanceof A||e instanceof Error&&e.name===`ValidationError`?`schema`:e instanceof TypeError||e instanceof Error&&/fetch|network|offline|connection/i.test(e.message)?`offline`:`generic`}function At(e,t){let n=Bt(t);return n===null?T(e,`model.selected.none`):`${n.identity.display_name} · ${Vt(e,n.lifecycle.state)}`}function jt(e,t){return t.bootstrap===null?T(e,`connection.ready`):T(e,`connection.footer.connected`,{mode:t.bootstrap.server.mode,version:t.bootstrap.server.build.version,sequence:Ht(e,t),status:Mt(e,t.connection)})}function Mt(e,t){return T(e,{idle:`connection.status.idle`,bootstrapping:`connection.status.bootstrapping`,ready:`connection.status.ready`,streaming:`connection.status.streaming`,polling:`connection.status.polling`,offline:`connection.status.offline`,stale:`connection.status.stale`,unauthorized:`connection.status.unauthorized`,forbidden:`connection.status.forbidden`,"schema-mismatch":`connection.status.schema_mismatch`,error:`connection.status.error`}[t])}function Nt(e){return e.snapshot.auth.status===`authenticated`?e.snapshot.connection===`schema-mismatch`?(0,x.jsx)(It,{...e}):(0,x.jsx)(Ft,{...e}):(0,x.jsx)(Pt,{...e})}function Pt(e){return e.authFailure===`schema`?(0,x.jsx)(It,{...e}):(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title}),(0,x.jsx)(`p`,{"data-testid":E(`connection.prompt.body`),children:T(e.locale,`connection.prompt.body`)})]}),(0,x.jsx)(ge,{title:T(e.locale,`state.unauthorized.title`),body:T(e.locale,`state.unauthorized.body`),tokenLabel:T(e.locale,`login.token.label`),tokenHelp:T(e.locale,`login.token.help`),submitLabel:T(e.locale,`login.submit`),logoutLabel:T(e.locale,`login.logout`),error:e.authFailure===null?void 0:zt(e.locale,e.authFailure),busy:e.snapshot.auth.status===`authenticating`,onSubmit:e.onLogin,onLogout:e.onLogout,testId:E(`auth.login`)})]})}function Ft(e){let t=e.snapshot.connection===`offline`||e.snapshot.connection===`stale`||e.snapshot.connection===`error`||e.snapshot.connection===`unauthorized`||e.snapshot.connection===`forbidden`;return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title}),(0,x.jsx)(`p`,{"data-testid":E(`connection.authenticated.body`),children:T(e.locale,`connection.authenticated.body`)})]}),(0,x.jsx)(se,{tone:`info`,title:T(e.locale,`connection.authenticated.title`),body:Lt(e.locale,e.snapshot),testId:E(`connection.authenticated.detail`)}),t?(0,x.jsx)(se,{title:T(e.locale,`connection.error.title`),body:Rt(e.locale,e.snapshot.connection),action:(0,x.jsx)(S,{onClick:e.onRetry,children:T(e.locale,`common.retry`)}),testId:E(`connection.error.title`)}):null]})}function It(e){return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading connection-prompt`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:e.eyebrow}),(0,x.jsx)(`h1`,{"data-testid":e.titleTestId,children:e.title})]}),(0,x.jsx)(_e,{title:T(e.locale,`state.schema_mismatch.title`),body:T(e.locale,`state.schema_mismatch.body`),actionLabel:T(e.locale,`common.reload`),onRecover:e.onRecoverSchema})]})}function Lt(e,t){let n=t.bootstrap;return n===null?T(e,`connection.prompt.detail`):T(e,`connection.authenticated.detail`,{mode:n.server.mode,version:n.server.build.version,status:Mt(e,t.connection),count:String(t.catalog.length),operations:String(t.operations.size),sequence:Ht(e,t)})}function Rt(e,t){return t===`offline`?T(e,`state.offline.body`):t===`stale`?T(e,`connection.error.stale`):t===`forbidden`?T(e,`connection.error.forbidden`):t===`unauthorized`?T(e,`connection.error.unauthorized`):T(e,`connection.error.generic`)}function zt(e,t){return t===`wrong-key`?T(e,`login.error.wrong_key`):t===`offline`?T(e,`login.error.offline`):t===`forbidden`?T(e,`login.error.forbidden`):t===`schema`?T(e,`login.error.schema`):T(e,`login.error.generic`)}function Bt(e){return e.selectedModelId===null?null:e.catalog.find(t=>t.identity.id===e.selectedModelId)??null}function Vt(e,t){return T(e,{unloaded:`models.status.unloaded`,loading:`models.status.loading`,ready:`models.status.ready`,draining:`models.status.draining`,unloading:`models.status.unloading`,failed:`models.status.failed`}[t])}function Ht(e,t){return String(t.lastSequence??t.catalogSequence??t.resourceFences.operationsSnapshot??T(e,`connection.snapshot.pending`))}var Ut=`webui.ui-api.v1`;function j(){return{schemaVersion:Ut,auth:{status:`signed-out`,tokenPresent:!1},connection:`idle`,bootstrap:null,catalog:[],catalogSequence:null,operations:new Map,runtimes:new Map,selectedModelId:null,serverInstanceId:null,lastEventId:null,lastSequence:null,lastUpdatedAt:null,lastSuccessfulAt:null,error:null,pendingReconciliations:new Map,resourceFences:{catalog:null,operationsSnapshot:null,operations:new Map,models:new Map,runtimes:new Map}}}function Wt(e,t){return t.type===`login-start`?{...e,auth:{status:`authenticating`,tokenPresent:!0},connection:`bootstrapping`,error:null}:t.type===`login-success`?Gt(e,t.bootstrap,t.now):t.type===`logout`?{...j(),lastUpdatedAt:t.now}:t.type===`select-model`?{...e,selectedModelId:t.modelId}:t.type===`catalog`?Kt(e,t.response,t.now):t.type===`operations-snapshot`?qt(e,t.response,t.now):t.type===`operation`?Jt(e,t.operation,t.sequence,t.now):t.type===`runtime`?Yt(e,t.runtime,t.sequence,t.now):t.type===`event`?Xt(e,t.event,t.now):t.type===`pending`?{...e,pendingReconciliations:N(e.pendingReconciliations,t.item.idempotencyKey,t.item)}:t.type===`reconciled`?{...e,pendingReconciliations:rn(e.pendingReconciliations,t.idempotencyKey)}:{...e,connection:t.connection,error:t.error??null,lastUpdatedAt:t.now}}function Gt(e,t,n){let r=t.server.server_instance_id;return e.serverInstanceId!==null&&e.serverInstanceId!==r?{...j(),auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n}:{...e,auth:{status:`authenticated`,tokenPresent:!0},connection:`ready`,bootstrap:t,serverInstanceId:r,lastUpdatedAt:n,error:null}}function Kt(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return $t(e,t.server_instance_id,n);if(e.catalogSequence!==null&&t.snapshot_sequence<e.catalogSequence)return e;let r=e.catalogSequence===t.snapshot_sequence;return en({...e,connection:e.connection===`bootstrapping`?`ready`:e.connection,catalog:tn(r?e.catalog:[],t),catalogSequence:t.snapshot_sequence,serverInstanceId:t.server_instance_id,lastSequence:nn({...e.resourceFences,catalog:t.snapshot_sequence}),error:null,resourceFences:{...e.resourceFences,catalog:t.snapshot_sequence}},n)}function qt(e,t,n){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return $t(e,t.server_instance_id,n);if(e.resourceFences.operationsSnapshot!==null&&t.snapshot_sequence<e.resourceFences.operationsSnapshot)return e;let r=e.resourceFences.operationsSnapshot===t.snapshot_sequence,i=r?new Map(e.operations):new Map,a=r?new Map(e.resourceFences.operations):new Map;for(let e of t.items)i.set(e.operation_id,e),a.set(e.operation_id,t.snapshot_sequence);let o={...e.resourceFences,operationsSnapshot:t.snapshot_sequence,operations:a};return en({...e,operations:i,serverInstanceId:t.server_instance_id,lastSequence:nn(o),error:null,resourceFences:o},n)}function Jt(e,t,n,r){if(n!==null&&e.resourceFences.operationsSnapshot!==null&&n<=e.resourceFences.operationsSnapshot)return e;let i=e.resourceFences.operations.get(t.operation_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=N(e.operations,t.operation_id,t),o=n===null?e.resourceFences.operations:N(e.resourceFences.operations,t.operation_id,n),s={...e.resourceFences,operations:o};return en({...e,operations:a,lastSequence:nn(s),error:null,resourceFences:s},r)}function Yt(e,t,n,r){if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return $t(e,t.server_instance_id,r);let i=e.resourceFences.runtimes.get(t.model_id);if(n!==null&&i!==void 0&&n<=i)return e;let a=n===null?e.resourceFences.runtimes:N(e.resourceFences.runtimes,t.model_id,n),o={...e.resourceFences,runtimes:a};return en({...e,runtimes:N(e.runtimes,t.model_id,t),serverInstanceId:t.server_instance_id,lastSequence:nn(o),error:null,resourceFences:o},r)}function Xt(e,t,n){if(t.schema_version!==`webui.ui-api.v1`)return{...e,connection:`schema-mismatch`,error:{code:`schema_mismatch`,message:`Unsupported WebUI schema ${t.schema_version}`,retryable:!1},lastUpdatedAt:n};if(e.serverInstanceId!==null&&t.server_instance_id!==e.serverInstanceId)return $t(e,t.server_instance_id,n);if(t.type===`heartbeat`)return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:e.connection===`polling`?`polling`:`streaming`,lastUpdatedAt:n};if(t.type===`server_restart`||t.type===`gap`||t.type===`reset`)return M(e,t,n);if(t.type===`operation`)return{...Jt(e,t.payload.operation,t.sequence,n),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`};if(t.type===`model_revision`)return Zt(e,t,n);if(t.type===`runtime`)return Qt(e,t,n);if(t.type===`snapshot`){let r={...e.resourceFences,catalog:t.payload.catalog_changed?t.sequence:e.resourceFences.catalog};return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:nn(r),connection:`streaming`,lastUpdatedAt:n,resourceFences:r}}return{...e,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,connection:`streaming`,lastUpdatedAt:n}}function Zt(e,t,n){let r=t.payload.model_id,i=e.resourceFences.models.get(r);if(i!==void 0&&t.sequence<=i)return e;let a=e.catalog.map(e=>e.identity.id===r&&t.payload.revision>=e.identity.revision?{...e,identity:{...e.identity,revision:t.payload.revision},lifecycle:t.payload.lifecycle}:e),o={...e.resourceFences,models:N(e.resourceFences.models,r,t.sequence)};return en({...e,catalog:a,serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:nn(o),connection:`streaming`,resourceFences:o},n)}function Qt(e,t,n){let r=t.payload.runtime,i=e.resourceFences.runtimes.get(r.model_id);if(i!==void 0&&t.sequence<=i)return e;let a={...e.resourceFences,runtimes:N(e.resourceFences.runtimes,r.model_id,t.sequence)};return en({...e,runtimes:N(e.runtimes,r.model_id,r),serverInstanceId:t.server_instance_id,lastEventId:t.event_id,lastSequence:nn(a),connection:`streaming`,resourceFences:a},n)}function M(e,t,n){let r=t.type===`server_restart`?null:e.lastSuccessfulAt;return{...j(),auth:e.auth,connection:`stale`,serverInstanceId:t.server_instance_id,selectedModelId:e.selectedModelId,lastEventId:t.event_id,error:{code:t.type,message:t.payload.reason,retryable:t.payload.resnapshot},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n,lastSuccessfulAt:r}}function $t(e,t,n){return{...j(),auth:e.auth,connection:`stale`,serverInstanceId:t,selectedModelId:e.selectedModelId,error:{code:`server_restarted`,message:`The mlxcel server restarted; refresh the authoritative snapshot before continuing.`,retryable:!0},pendingReconciliations:e.pendingReconciliations,lastUpdatedAt:n}}function en(e,t){return{...e,lastUpdatedAt:t,lastSuccessfulAt:t}}function tn(e,t){if(e.length>0){let n=new Map(e.map(e=>[e.identity.id,e]));for(let e of t.items)n.set(e.identity.id,e);return Array.from(n.values()).sort((e,t)=>e.identity.display_name.localeCompare(t.identity.display_name))}return t.items}function nn(e){let t=[e.catalog,e.operationsSnapshot,...e.operations.values(),...e.models.values(),...e.runtimes.values()].filter(e=>e!==null);return t.length===0?null:Math.min(...t)}function N(e,t,n){let r=new Map(e);return r.set(t,n),r}function rn(e,t){let n=new Map(e);return n.delete(t),n}var an=2e3,on=3e4,sn=3e4,cn=6e4,ln=class{client;dispatch;getSnapshot;clock;visibility;random;timer=null;inflight=null;eventAbort=null;stopped=!0;failures=0;generation=0;unsubscribe;constructor(e){this.client=e.client,this.dispatch=e.dispatch,this.getSnapshot=e.getSnapshot,this.clock=e.clock??hn,this.visibility=e.visibility??gn,this.random=e.random??Math.random,this.unsubscribe=this.visibility.subscribe(()=>this.reschedule(0))}start(){this.stopped&&(this.stopped=!1,this.generation+=1,this.reschedule(0))}stop(){this.generation+=1,this.stopped=!0,this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=null,this.inflight?.abort(),this.inflight=null,this.eventAbort?.abort(),this.eventAbort=null}dispose(){this.stop(),this.unsubscribe()}async refresh(){if(this.inflight!==null)return;let e=new AbortController;this.inflight=e;let t=this.generation,n=this.clock.now();try{if(this.getSnapshot().auth.status===`signed-out`)return;let r=await this.client.bootstrap(e.signal);if(!this.isCurrent(e,t)||this.getSnapshot().auth.status===`signed-out`)return;this.dispatch({type:`login-success`,bootstrap:r,now:n});let i=await this.catalogSnapshot(e.signal);if(!this.isCurrent(e,t))return;for(let e of i)this.dispatch({type:`catalog`,response:e,now:n});let a=await this.operationSnapshot(e.signal);if(!this.isCurrent(e,t))return;let o=i.at(-1),s=a[0];if(o!==void 0&&s!==void 0&&s.server_instance_id!==o.server_instance_id){this.dispatch({type:`connection`,connection:`stale`,error:{code:`snapshot_mismatch`,message:`Operation and catalog snapshots came from different server instances.`,retryable:!0},now:n});return}let c=a.flatMap(e=>(this.dispatch({type:`operations-snapshot`,response:e,now:n}),e.items));if(await this.refreshSelectedRuntime(e.signal,n,t),!this.isCurrent(e,t))return;let l=this.reconcilePending(c,n);this.failures=0,l||this.dispatch({type:`connection`,connection:`ready`,now:n}),this.eventAbort===null&&this.startEvents()}catch(n){!e.signal.aborted&&this.generation===t&&(this.failures+=1,this.dispatch({type:`connection`,connection:dn(n),error:fn(n),now:this.clock.now()}))}finally{this.inflight===e&&(this.inflight=null),!this.stopped&&this.generation===t&&this.reschedule(this.pollDelay())}}noteUnknownPost(e){this.dispatch({type:`pending`,item:e}),this.reschedule(0)}startEvents(){this.eventAbort?.abort();let e=new AbortController;this.eventAbort=e;let t=this.generation,n=un(this.getSnapshot());this.client.events({onEvent:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.handleEvent(n)},onRetryAfter:n=>{this.generation===t&&!e.signal.aborted&&!this.stopped&&this.reschedule(n)}},e.signal,n).then(()=>{this.generation===t&&(this.eventAbort===e&&(this.eventAbort=null),!e.signal.aborted&&!this.stopped&&this.reschedule(this.backoffDelay()))}).catch(n=>{e.signal.aborted||this.stopped||this.generation!==t||(this.failures+=1,this.eventAbort=null,this.dispatch({type:`connection`,connection:dn(n),error:fn(n),now:this.clock.now()}),this.reschedule(this.backoffDelay()))})}handleEvent(e){this.dispatch({type:`event`,event:e,now:this.clock.now()}),(e.type===`server_restart`||e.type===`gap`||e.type===`reset`)&&e.payload.resnapshot&&(this.eventAbort?.abort(),this.eventAbort=null,this.refresh())}async catalogSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.catalog(n===void 0?{}:{cursor:n},e);if(r===null?r=i:mn(`catalog`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async operationSnapshot(e){let t=[],n,r=null;for(;;){let i=await this.client.operationsPage(n===void 0?{}:{cursor:n},e);if(r===null?r=i:mn(`operations`,r,i),t.push(i),i.pagination.next_cursor===null)return t;n=i.pagination.next_cursor}}async refreshSelectedRuntime(e,t,n){let r=this.getSnapshot().selectedModelId;if(r===null)return;let i=await this.client.runtime(r,e);e.aborted||this.generation!==n||this.getSnapshot().selectedModelId===r&&this.dispatch({type:`runtime`,runtime:i,sequence:i.snapshot_sequence,now:t})}reconcilePending(e,t){let n=!1,r=new Set(e.map(e=>e.operation_id));for(let e of this.getSnapshot().pendingReconciliations.values())e.operationId!==null&&r.has(e.operationId)?this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}):t-e.createdAt>=cn&&(this.dispatch({type:`reconciled`,idempotencyKey:e.idempotencyKey}),this.dispatch({type:`connection`,connection:`stale`,error:{code:`unknown_post_unresolved`,message:`A previous control request could not be matched to an operation after reconciliation.`,retryable:!0},now:t}),n=!0);return n}reschedule(e){this.stopped||(this.timer!==null&&this.clock.clearTimeout(this.timer),this.timer=this.clock.setTimeout(()=>{this.timer=null,this.refresh()},e))}pollDelay(){return this.visibility.hidden()?on:an}backoffDelay(){let e=Math.min(sn,500*2**Math.min(6,this.failures));return Math.round(e/2+this.random()*(e/2))}isCurrent(e,t){return this.inflight===e&&this.generation===t&&!e.signal.aborted}};function un(e){let t=nn(e.resourceFences);if(e.serverInstanceId!==null&&t!==null)return{lastEventId:null,serverInstanceId:e.serverInstanceId,afterSequence:t};if(e.lastEventId!==null)return{lastEventId:e.lastEventId,serverInstanceId:null,afterSequence:null}}function dn(e){if(e instanceof pn)return`stale`;let t=typeof e==`object`&&e&&`status`in e&&typeof e.status==`number`?e.status:null;return t===401||e instanceof Error&&/401|unauthorized/i.test(e.message)?`unauthorized`:t===403||e instanceof Error&&/403|forbidden/i.test(e.message)?`forbidden`:e instanceof DOMException&&e.name===`AbortError`?`stale`:`offline`}function fn(e){return e instanceof pn?{code:e.code,message:e.message,retryable:!0}:{code:`sync_error`,message:e instanceof Error?e.message.replace(/Bearer\s+\S+/gi,`Bearer [redacted]`):`Unknown WebUI client error`,retryable:!0}}var pn=class extends Error{code=`snapshot_mismatch`;constructor(e){super(e),this.name=`SnapshotConsistencyError`}};function mn(e,t,n){if(n.server_instance_id!==t.server_instance_id)throw new pn(`${e} pagination crossed a server restart; refresh the authoritative snapshot.`);if(n.snapshot_sequence!==t.snapshot_sequence)throw new pn(`${e} pagination crossed snapshot sequence boundaries; refresh the authoritative snapshot.`)}var hn={setTimeout:(e,t)=>globalThis.setTimeout(e,t),clearTimeout:e=>globalThis.clearTimeout(e),now:()=>Date.now()},gn={hidden:()=>typeof document<`u`&&document.hidden,subscribe:e=>typeof document>`u`?()=>void 0:(document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e))},_n=b.createContext(null),vn=b.createContext(null);function yn({children:e,apiBase:t,fetchImpl:n}){let[r,i]=b.useReducer(Wt,void 0,j),a=b.useRef(r);a.current=r;let o=b.useRef(0),s=b.useRef(null),c=b.useMemo(()=>new St({apiBase:t,fetchImpl:n,onUnauthorized:()=>{o.current+=1,s.current?.stop(),i({type:`logout`,now:Date.now()})}}),[t,n]);b.useEffect(()=>{let e=new ln({client:c,dispatch:i,getSnapshot:()=>a.current});return s.current=e,()=>{e.dispose(),s.current=null,c.abortAll()}},[c]);let l=b.useMemo(()=>({login:async e=>{o.current+=1;let t=o.current;c.abortAll(),c.setBearerToken(e),i({type:`login-start`});try{let e=await c.bootstrap();if(o.current!==t)return;i({type:`login-success`,bootstrap:e,now:Date.now()}),s.current?.start()}catch(e){throw o.current===t&&(c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})),e}},logout:()=>{o.current+=1,s.current?.stop(),c.setBearerToken(null),c.abortAll(),i({type:`logout`,now:Date.now()})},refresh:async()=>{await s.current?.refresh()},selectModel:e=>{c.abortAll(),i({type:`select-model`,modelId:e}),s.current?.refresh()},loadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},unloadModel:async e=>{await u(`model-action`,e.idempotency_key,e.model_id,()=>c.modelAction(e))},downloadModel:async e=>{await u(`download`,e.idempotency_key,void 0,()=>c.download(e))},removeModel:async e=>{await u(`removal`,e.idempotency_key,e.model_id,()=>c.removeModel(e))},refreshRuntime:async e=>{let t=o.current,n=await c.runtime(e);return o.current===t&&i({type:`runtime`,runtime:n,sequence:n.snapshot_sequence,now:Date.now()}),n},refreshCatalog:async e=>{await u(`catalog-refresh`,e,void 0,()=>c.refreshCatalog(e))},cancelOperation:async e=>{await c.cancelOperation(e),await s.current?.refresh()},streamChatCompletions:async(e,t,n,r)=>{await c.chatCompletions(d(e),t,n,r)},streamResponses:async(e,t,n,r)=>{await c.responses(d(e),t,n,r)}}),[c]);async function u(e,t,n,r){try{let i=await r();s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:i.operation_id,modelId:n,createdAt:Date.now()})}catch(r){throw r instanceof bt||s.current?.noteUnknownPost({kind:e,idempotencyKey:t,operationId:null,modelId:n,createdAt:Date.now()}),r}}function d(e){let t=a.current.catalog.find(t=>t.identity.id===e);if(t===void 0)throw Error(`Selected model is not present in the catalog snapshot.`);if(t.identity.inference_id.length===0)throw Error(`Selected model does not expose an inference model id.`);return t.identity.inference_id}return(0,x.jsx)(vn.Provider,{value:l,children:(0,x.jsx)(_n.Provider,{value:r,children:e})})}function bn(){let e=b.useContext(_n);if(e===null)throw Error(`useWebUi must be used within WebUiProvider.`);return e}function xn(){let e=b.useContext(vn);if(e===null)throw Error(`useWebUiActions must be used within WebUiProvider.`);return e}var Sn=[`models`,`chat`,`activity`,`settings`,`gallery`],Cn=[`models`,`chat`,`activity`,`settings`];function wn(){let e=window.location.hash.slice(1).replace(/^\//,``);return Sn.includes(e)?e:`models`}function Tn(){let e=bn(),t=xn(),[n,r]=(0,b.useState)(wn),[i,a]=(0,b.useState)(k),[o,s]=(0,b.useState)(null),[c,l]=(0,b.useState)(null);(0,b.useEffect)(()=>{let e=()=>r(wn());return window.addEventListener(`hashchange`,e),()=>window.removeEventListener(`hashchange`,e)},[]),(0,b.useEffect)(()=>{je(document.documentElement,i),ke(i)},[i]),(0,b.useEffect)(()=>{let e=()=>{document.documentElement.dataset.documentHidden=String(document.hidden)};return e(),document.addEventListener(`visibilitychange`,e),()=>document.removeEventListener(`visibilitychange`,e)},[]),(0,b.useEffect)(()=>{e.auth.status===`authenticated`&&l(null)},[e.auth.status]);let u=e=>{r(e),window.history.replaceState(null,``,`#${e}`)},d=e=>{l(null),t.login(e).catch(e=>l(kt(e)))},f=()=>{l(null),t.logout()},p=En(n,i,a,{snapshot:e,authFailure:c,login:d,logout:f,retry:()=>{l(null),t.refresh()},recoverSchema:()=>{f(),window.location.reload()}});return(0,x.jsxs)(x.Fragment,{children:[(0,x.jsx)(be,{locale:i.locale,route:n,onRouteChange:u,onCommand:()=>s(`command`),onHelp:()=>s(`help`),selectedModel:At(i.locale,e),connectionLabel:jt(i.locale,e),connectionState:e.connection,inspector:null,children:p}),(0,x.jsx)(Mn,{open:o===`command`,locale:i.locale,onClose:()=>s(null),onNavigate:e=>{u(e),s(null)}}),(0,x.jsxs)(le,{open:o===`help`,title:T(i.locale,`help.title`),onClose:()=>s(null),testId:`help-dialog`,closeLabel:T(i.locale,`common.close`),children:[(0,x.jsx)(`p`,{"data-testid":E(`help.body`),children:T(i.locale,`help.body`)}),(0,x.jsxs)(`ul`,{className:`shortcut-list`,children:[(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`⌘/Ctrl`}),` + `,(0,x.jsx)(`kbd`,{children:`K`}),` `,T(i.locale,`command.search`)]}),(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`Esc`}),` `,T(i.locale,`common.cancel`)]}),(0,x.jsxs)(`li`,{children:[(0,x.jsx)(`kbd`,{children:`[`}),` / `,(0,x.jsx)(`kbd`,{children:`]`}),` `,T(i.locale,`nav.models`)]})]})]})]})}function En(e,t,n,r){return e===`models`?(0,x.jsx)(Dn,{locale:t.locale,context:r}):e===`chat`?(0,x.jsx)(On,{locale:t.locale,context:r}):e===`activity`?(0,x.jsx)(kn,{locale:t.locale,context:r}):e===`settings`?(0,x.jsx)(An,{appearance:t,setAppearance:n}):(0,x.jsx)(Fe,{locale:t.locale})}function Dn(e){return(0,x.jsx)(Nt,{locale:e.locale,eyebrow:T(e.locale,`routes.models.eyebrow`),title:T(e.locale,`models.title`),titleTestId:E(`models.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function On(e){return(0,x.jsx)(Nt,{locale:e.locale,eyebrow:T(e.locale,`routes.chat.eyebrow`),title:T(e.locale,`chat.title`),titleTestId:E(`chat.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function kn(e){return(0,x.jsx)(Nt,{locale:e.locale,eyebrow:T(e.locale,`routes.activity.eyebrow`),title:T(e.locale,`activity.title`),titleTestId:E(`activity.title`),snapshot:e.context.snapshot,authFailure:e.context.authFailure,onLogin:e.context.login,onLogout:e.context.logout,onRetry:e.context.retry,onRecoverSchema:e.context.recoverSchema})}function An(e){let t=t=>e.setAppearance({...e.appearance,...t});return(0,x.jsxs)(`div`,{className:`screen-stack`,children:[(0,x.jsxs)(`section`,{className:`screen-heading`,children:[(0,x.jsx)(`p`,{className:`eyebrow`,children:T(e.appearance.locale,`routes.settings.eyebrow`)}),(0,x.jsx)(`h1`,{"data-testid":E(`settings.title`),children:T(e.appearance.locale,`settings.title`)}),(0,x.jsx)(`p`,{"data-testid":E(`settings.appearance`),children:T(e.appearance.locale,`settings.browser_only`)})]}),(0,x.jsxs)(`div`,{className:`settings-grid`,children:[(0,x.jsx)(re,{label:T(e.appearance.locale,`settings.theme`),value:e.appearance.theme,onChange:e=>t({theme:e}),options:[{value:`system`,label:T(e.appearance.locale,`settings.theme.system`)},{value:`light`,label:T(e.appearance.locale,`settings.theme.light`)},{value:`dark`,label:T(e.appearance.locale,`settings.theme.dark`)}],testId:E(`settings.theme`)}),(0,x.jsx)(re,{label:T(e.appearance.locale,`settings.material`),value:e.appearance.material,onChange:e=>t({material:e}),options:[{value:`glass`,label:T(e.appearance.locale,`settings.material.glass`)},{value:`tinted`,label:T(e.appearance.locale,`settings.material.tinted`)},{value:`opaque`,label:T(e.appearance.locale,`settings.material.opaque`)}],testId:E(`settings.material`)}),(0,x.jsx)(re,{label:T(e.appearance.locale,`settings.locale`),value:e.appearance.locale,onChange:e=>t({locale:e}),options:[{value:`en`,label:T(e.appearance.locale,`settings.locale.en`)},{value:`ko`,label:T(e.appearance.locale,`settings.locale.ko`)}],testId:E(`settings.locale`)}),(0,x.jsx)(re,{label:T(e.appearance.locale,`settings.high_contrast`),value:e.appearance.highContrast,onChange:e=>t({highContrast:e}),options:[{value:`system`,label:T(e.appearance.locale,`settings.high_contrast.system`)},{value:`on`,label:T(e.appearance.locale,`settings.high_contrast.on`)},{value:`off`,label:T(e.appearance.locale,`settings.high_contrast.off`)}],testId:E(`settings.high_contrast`)}),(0,x.jsxs)(`label`,{className:`ds-field`,children:[(0,x.jsxs)(`span`,{children:[T(e.appearance.locale,`settings.glass_intensity`),`: `,e.appearance.glassIntensity]}),(0,x.jsx)(`input`,{type:`range`,min:`0`,max:`100`,value:e.appearance.glassIntensity,onChange:e=>t({glassIntensity:Number(e.currentTarget.value)}),"data-testid":E(`settings.glass_intensity`)})]}),(0,x.jsx)(jn,{label:T(e.appearance.locale,`settings.reduce_motion`),checked:e.appearance.reduceMotion,onChange:e=>t({reduceMotion:e}),testId:E(`settings.reduce_motion`)}),(0,x.jsx)(jn,{label:T(e.appearance.locale,`settings.reduce_transparency`),checked:e.appearance.reduceTransparency,onChange:e=>t({reduceTransparency:e}),testId:E(`settings.reduce_transparency`)})]}),(0,x.jsx)(se,{tone:`info`,title:T(e.appearance.locale,`settings.browser_only`),body:T(e.appearance.locale,`settings.browser_only.body`),testId:E(`settings.browser_only`)})]})}function jn(e){return(0,x.jsxs)(`label`,{className:`toggle`,children:[(0,x.jsx)(`input`,{type:`checkbox`,checked:e.checked,onChange:t=>e.onChange(t.currentTarget.checked),"data-testid":e.testId}),(0,x.jsx)(`span`,{children:e.label})]})}function Mn(e){let[t,n]=(0,b.useState)(``),r=(e.locale&&window.location.hash.includes(`gallery`)?Sn:Cn).filter(e=>e.includes(t.toLowerCase()));return(0,x.jsxs)(le,{open:e.open,title:T(e.locale,`command.title`),onClose:e.onClose,testId:`command-dialog`,closeLabel:T(e.locale,`common.close`),children:[(0,x.jsx)(w,{label:T(e.locale,`command.search`),value:t,onChange:n,testId:E(`command.search`)}),r.length===0?(0,x.jsx)(`p`,{"data-testid":E(`command.no_results`),children:T(e.locale,`command.no_results`)}):(0,x.jsx)(`div`,{className:`command-list`,children:r.map(t=>(0,x.jsx)(S,{onClick:()=>e.onNavigate(t),children:T(e.locale,t===`models`?`nav.models`:t===`chat`?`nav.chat`:t===`activity`?`nav.activity`:t===`settings`?`nav.settings`:`nav.gallery`)},t))})]})}var Nn=document.getElementById(`root`);if(Nn===null)throw Error(`Missing #root element for mlxcel WebUI.`);(0,y.createRoot)(Nn).render((0,x.jsx)(b.StrictMode,{children:(0,x.jsx)(yn,{children:(0,x.jsx)(Tn,{})})}));