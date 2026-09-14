// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
export interface LocalImage { name: string; type: string; dataUrl: string }
export interface MediaImageLimits {
  max_images: number; max_image_bytes: number; max_width: number;
  max_height: number; max_decoded_bytes: number; max_body_bytes: number;
}
// Additional browser memory budgets; never substitutes for advertised limits.
export const LOCAL_IMAGE_LIMITS = { count: 4, bytes: 8 * 1024 * 1024, dimension: 4096, pixels: 16_000_000 } as const;
const TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);
function dimensions(bytes: Uint8Array, type: string): [number, number] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const ascii = (offset: number, length: number) => String.fromCharCode(...bytes.subarray(offset, offset + length));
  if (type === 'image/png' && bytes.length >= 33 && bytes[0] === 137 && ascii(1, 7) === 'PNG\r\n\x1a\n' && view.getUint32(8) === 13 && ascii(12, 4) === 'IHDR') return [view.getUint32(16), view.getUint32(20)];
  if (type === 'image/jpeg' && bytes.length >= 4 && bytes[0] === 255 && bytes[1] === 216) {
    let cursor = 2;
    while (cursor + 4 <= bytes.length) {
      if (bytes[cursor++] !== 255) break;
      while (bytes[cursor] === 255) cursor++;
      const marker = bytes[cursor++];
      if (marker === 217 || marker === 218) break;
      if (marker === 1 || (marker >= 208 && marker <= 215)) continue;
      if (cursor + 2 > bytes.length) break;
      const length = view.getUint16(cursor);
      if (length < 2 || cursor + length > bytes.length) break;
      if ([192, 193, 194, 195, 197, 198, 199, 201, 202, 203, 205, 206, 207].includes(marker) && length >= 8) return [view.getUint16(cursor + 5), view.getUint16(cursor + 3)];
      cursor += length;
    }
  }
  if (type === 'image/webp' && bytes.length >= 25 && ascii(0, 4) === 'RIFF' && ascii(8, 4) === 'WEBP' && view.getUint32(4, true) + 8 === bytes.length) {
    const chunk = ascii(12, 4);
    const chunkSize = view.getUint32(16, true);
    if (chunkSize < 5 || 20 + chunkSize > bytes.length) throw new Error('Invalid WebP chunk length.');
    const u24 = (offset: number) => bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
    if (chunk === 'VP8X' && bytes.length >= 30 && view.getUint32(16, true) >= 10 && !(bytes[20] & 2)) return [u24(24) + 1, u24(27) + 1];
    if (chunk === 'VP8 ' && bytes.length >= 30 && bytes[23] === 157 && bytes[24] === 1 && bytes[25] === 42) return [view.getUint16(26, true) & 16383, view.getUint16(28, true) & 16383];
    if (chunk === 'VP8L' && bytes[20] === 47) { const bits = view.getUint32(21, true); return [(bits & 16383) + 1, ((bits >>> 14) & 16383) + 1]; }
  }
  throw new Error('Image content does not match a supported PNG, JPEG or non-animated WebP image.');
}
function readBytes(file: File): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Could not read the selected image.'));
    reader.onabort = () => reject(new Error('Image reading was cancelled.'));
    reader.onload = () => reader.result instanceof ArrayBuffer ? resolve(new Uint8Array(reader.result)) : reject(new Error('Invalid image data.'));
    reader.readAsArrayBuffer(file);
  });
}
function validateLimits(limits: MediaImageLimits): void {
  if (!limits || [limits.max_images, limits.max_image_bytes, limits.max_width, limits.max_height, limits.max_decoded_bytes, limits.max_body_bytes].some(value => !Number.isSafeInteger(value) || value <= 0)) throw new Error('Image limits are unavailable.');
}
function validateDimensions(width: number, height: number, limits: MediaImageLimits): void {
  if (!width || !height || width > Math.min(LOCAL_IMAGE_LIMITS.dimension, limits.max_width) || height > Math.min(LOCAL_IMAGE_LIMITS.dimension, limits.max_height) || width * height > LOCAL_IMAGE_LIMITS.pixels || width * height * 4 > limits.max_decoded_bytes) throw new Error('Image dimensions exceed the server or browser decode budget.');
}
/** Revalidate every image in the FINAL request, including historical turns.
 * The browser's four-image picker budget is per turn; the server count applies
 * across all messages. This does not replace the complete JSON body check.
 */
export function validateRequestImages(images: readonly LocalImage[], limits: MediaImageLimits): void {
  validateLimits(limits);
  if (images.length > limits.max_images) throw new Error('Conversation images exceed the server image count limit.');
  let encodedBytes = 0;
  const byteLimit = Math.min(LOCAL_IMAGE_LIMITS.bytes, limits.max_image_bytes);
  // Bound encoded allocation before decoding any payload.
  for (const image of images) {
    if (!image || !TYPES.has(image.type) || typeof image.dataUrl !== 'string') throw new Error('Invalid local image attachment.');
    encodedBytes += image.dataUrl.length;
    if (encodedBytes > limits.max_body_bytes) throw new Error('Images exceed the server request body limit.');
    const prefix = `data:${image.type};base64,`;
    if (!image.dataUrl.startsWith(prefix)) throw new Error('Invalid image data URL.');
    const payloadLength = image.dataUrl.length - prefix.length;
    if (payloadLength > 4 * Math.ceil(byteLimit / 3)) throw new Error('Image exceeds the current server or browser byte limit.');
  }
  for (const image of images) {
    const payload = image.dataUrl.slice(`data:${image.type};base64,`.length);
    if (!payload || payload.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(payload)) throw new Error('Invalid image base64 encoding.');
    const binary = atob(payload);
    if (btoa(binary) !== payload) throw new Error('Image base64 encoding must be canonical.');
    if (!binary.length || binary.length > byteLimit) throw new Error('Image exceeds the current server or browser byte limit.');
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index++) bytes[index] = binary.charCodeAt(index);
    const [width, height] = dimensions(bytes, image.type);
    validateDimensions(width, height, limits);
  }
}
export async function loadLocalImages(files: File[] | FileList, currentCount: number, limits: MediaImageLimits, existingImages: LocalImage[] = []): Promise<LocalImage[]> {
  validateLimits(limits);
  const countLimit = Math.min(LOCAL_IMAGE_LIMITS.count, limits.max_images);
  const byteLimit = Math.min(LOCAL_IMAGE_LIMITS.bytes, limits.max_image_bytes);
  const selected = Array.from(files);
  if (!Number.isInteger(currentCount) || currentCount < 0 || currentCount + selected.length > countLimit || currentCount !== existingImages.length) throw new Error(`Attach at most ${countLimit} images per message; existing attachment count must match.`);
  for (const file of selected) {
    if (!(file instanceof File) || !TYPES.has(file.type)) throw new Error('Choose local PNG, JPEG or WebP image files.');
    if (!file.size || file.size > byteLimit) throw new Error(`Each image must be nonempty and no larger than ${byteLimit} bytes.`);
  }
  // Base64 text alone must fit; the caller additionally checks the complete
  // serialized request, including text, history and JSON framing overhead.
  const encodedBytes = existingImages.reduce((sum, image) => sum + image.dataUrl.length, 0)
    + selected.reduce((sum, file) => sum + 4 * Math.ceil(file.size / 3) + `data:${file.type};base64,`.length, 0);
  if (encodedBytes > limits.max_body_bytes) throw new Error('Images exceed the server request body limit.');
  const result: LocalImage[] = [];
  for (const file of selected) {
    const bytes = await readBytes(file);
    const [width, height] = dimensions(bytes, file.type);
    validateDimensions(width, height, limits);
    // Check header bounds BEFORE a browser decoder allocates. Where this API is
    // absent, full decoding is left to the server rather than faking a pass.
    if (typeof createImageBitmap === 'function') {
      let bitmap: ImageBitmap;
      try { bitmap = await createImageBitmap(file, { imageOrientation: 'none' }); } catch { throw new Error('The image could not be decoded.'); }
      try { if (bitmap.width !== width || bitmap.height !== height) throw new Error('Decoded image dimensions do not match its header.'); }
      finally { bitmap.close(); }
    }
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
    result.push({ name: file.name, type: file.type, dataUrl: `data:${file.type};base64,${btoa(binary)}` });
  }
  return result;
}
