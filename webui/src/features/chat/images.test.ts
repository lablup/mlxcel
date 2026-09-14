// Copyright 2025-2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadLocalImages, validateRequestImages, type MediaImageLimits } from './images';
const limits: MediaImageLimits = { max_images: 16, max_image_bytes: 64 * 1024 * 1024, max_width: 16384, max_height: 16384, max_decoded_bytes: 512 * 1024 * 1024, max_body_bytes: 2 * 1024 * 1024 };
const pngBytes = () => Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jB1kAAAAASUVORK5CYII='), char => char.charCodeAt(0));
const png = () => new File([pngBytes()], 'local.png', { type: 'image/png' });
afterEach(() => vi.unstubAllGlobals());
describe('local image admission', () => {
  it('encodes a local magic-checked image without any network access', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const [image] = await loadLocalImages([png()], 0, limits);
    expect(image.name).toBe('local.png'); expect(image.type).toBe('image/png');
    expect(image.dataUrl).toMatch(/^data:image\/png;base64,iVBOR/); expect(fetch).not.toHaveBeenCalled();
  });
  it('rejects URLs, SVG, empty content and MIME spoofing', async () => {
    await expect(loadLocalImages(['https://example.test/a.png' as unknown as File], 0, limits)).rejects.toThrow('local');
    await expect(loadLocalImages([new File(['<svg/>'], 'x.svg', { type: 'image/svg+xml' })], 0, limits)).rejects.toThrow('local');
    await expect(loadLocalImages([new File([], 'empty.png', { type: 'image/png' })], 0, limits)).rejects.toThrow('nonempty');
    await expect(loadLocalImages([new File(['<html/>'], 'x.png', { type: 'image/png' })], 0, limits)).rejects.toThrow('content');
    await expect(loadLocalImages([new File([pngBytes()], 'x.jpg', { type: 'image/jpeg' })], 0, limits)).rejects.toThrow('content');
  });
  it('enforces advertised count, byte and body limits before decoding', async () => {
    const decode = vi.fn(); vi.stubGlobal('createImageBitmap', decode);
    await expect(loadLocalImages([png(), png()], 0, { ...limits, max_images: 1 })).rejects.toThrow('at most 1');
    await expect(loadLocalImages([png()], 0, { ...limits, max_image_bytes: 10 })).rejects.toThrow('10 bytes');
    await expect(loadLocalImages([png()], 0, { ...limits, max_body_bytes: 90 })).rejects.toThrow('body');
    await expect(loadLocalImages([png()], 1, limits)).rejects.toThrow('count must match');
    const existing = [{ name: 'old.png', type: 'image/png', dataUrl: 'a'.repeat(100) }];
    await expect(loadLocalImages([png()], 1, { ...limits, max_body_bytes: 150 }, existing)).rejects.toThrow('body');
    expect(decode).not.toHaveBeenCalled();
  });
  it('rejects absent or malformed advertised limits rather than assuming defaults', async () => {
    await expect(loadLocalImages([png()], 0, { ...limits, max_width: NaN })).rejects.toThrow('unavailable');
    await expect(loadLocalImages([png()], 0, { ...limits, max_body_bytes: 0 })).rejects.toThrow('unavailable');
  });
  it('bounds width, height, pixels and decoded bytes before browser allocation', async () => {
    const decode = vi.fn(); vi.stubGlobal('createImageBitmap', decode);
    for (const [width, height, cap] of [[5000, 1, limits], [1, 3, { ...limits, max_height: 2 }], [3, 1, { ...limits, max_width: 2 }], [4096, 4096, limits], [1, 1, { ...limits, max_decoded_bytes: 3 }]] as const) {
      const bytes = pngBytes(); const view = new DataView(bytes.buffer); view.setUint32(16, width); view.setUint32(20, height);
      await expect(loadLocalImages([new File([bytes], 'bomb.png', { type: 'image/png' })], 0, cap)).rejects.toThrow('decode budget');
    }
    expect(decode).not.toHaveBeenCalled();
  });
  it('closes decoded bitmaps on success and mismatch and rejects decoder failure', async () => {
    const close = vi.fn(); const decode = vi.fn().mockResolvedValue({ width: 1, height: 1, close }); vi.stubGlobal('createImageBitmap', decode);
    await loadLocalImages([png()], 0, limits); expect(close).toHaveBeenCalledOnce();
    decode.mockResolvedValue({ width: 2, height: 1, close });
    await expect(loadLocalImages([png()], 0, limits)).rejects.toThrow('do not match'); expect(close).toHaveBeenCalledTimes(2);
    decode.mockRejectedValue(new Error('invalid'));
    await expect(loadLocalImages([png()], 0, limits)).rejects.toThrow('could not be decoded');
  });
});

it('parses JPEG and lossless WebP dimensions before the injected decoder', async () => {
  const close = vi.fn(); vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ width: 1, height: 1, close }));
  // Minimal header probes, NOT represented as real decodable image fixtures.
  const jpeg = new Uint8Array([255, 216, 255, 192, 0, 8, 8, 0, 1, 0, 1, 0]);
  const webp = new Uint8Array(26); const view = new DataView(webp.buffer);
  webp.set(new TextEncoder().encode('RIFF'), 0); view.setUint32(4, 18, true);
  webp.set(new TextEncoder().encode('WEBPVP8L'), 8); view.setUint32(16, 5, true); webp[20] = 47;
  const images = await loadLocalImages([new File([jpeg], 'x.jpg', { type: 'image/jpeg' }), new File([webp], 'x.webp', { type: 'image/webp' })], 0, limits);
  expect(images).toHaveLength(2); expect(close).toHaveBeenCalledTimes(2);
  view.setUint32(16, 999, true);
  await expect(loadLocalImages([new File([webp], 'x.webp', { type: 'image/webp' })], 0, limits)).rejects.toThrow('chunk length');
});
it('rejects truncated headers without out-of-bounds reads', async () => {
  for (const type of ['image/png', 'image/jpeg', 'image/webp']) {
    for (const length of [1, 2, 4, 20, 24]) await expect(loadLocalImages([new File([new Uint8Array(length)], 'bad', { type })], 0, limits)).rejects.toThrow('content');
  }
});

describe('whole-request image revalidation', () => {
  it('counts prior and current images together but does not apply per-turn picker count', async () => {
    const images = await loadLocalImages([png()], 0, limits);
    expect(() => validateRequestImages(Array.from({ length: 5 }, () => images[0]), limits)).not.toThrow();
    expect(() => validateRequestImages([...images, ...images], { ...limits, max_images: 1 })).toThrow('count');
  });
  it('rechecks changed lower byte, decode and body limits after attachment', async () => {
    const images = await loadLocalImages([png()], 0, limits);
    expect(() => validateRequestImages(images, { ...limits, max_image_bytes: png().size - 1 })).toThrow('byte limit');
    expect(() => validateRequestImages(images, { ...limits, max_decoded_bytes: 3 })).toThrow('decode budget');
    expect(() => validateRequestImages([...images, ...images], { ...limits, max_body_bytes: images[0].dataUrl.length + 1 })).toThrow('body limit');
    const bytes = pngBytes(); new DataView(bytes.buffer).setUint32(16, 2);
    const wider = { ...images[0], dataUrl: `data:image/png;base64,${btoa(String.fromCharCode(...bytes))}` };
    expect(() => validateRequestImages([wider], limits)).not.toThrow();
    expect(() => validateRequestImages([wider], { ...limits, max_width: 1 })).toThrow('decode budget');
  });
  it('rejects imported MIME/signature spoofing, remote URLs and noncanonical base64', async () => {
    const [image] = await loadLocalImages([png()], 0, limits);
    for (const dataUrl of ['https://example.test/a.png', 'data:image/png;base64,AAAA\n', 'data:image/png;base64,AB==', 'data:image/png;base64,PHN2Zy8+']) {
      expect(() => validateRequestImages([{ ...image, dataUrl }], limits)).toThrow();
    }
    expect(() => validateRequestImages([{ ...image, type: 'image/jpeg' }], limits)).toThrow('data URL');
    expect(() => validateRequestImages([{ ...image, type: 'image/jpeg', dataUrl: image.dataUrl.replace('image/png', 'image/jpeg') }], limits)).toThrow('content');
  });
});
