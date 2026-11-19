import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { Member } from './types';

const errorLabels: Record<string, string> = {
  account_required: 'Đăng nhập để tiếp tục.',
  invalid_credentials: 'Email hoặc mật khẩu chưa đúng.',
  email_or_session_already_registered: 'Email này đã được đăng ký.',
  listing_unavailable: 'Lá thẻ này hiện không còn mở bán.',
  front_and_back_photos_required: 'Hãy thêm ảnh mặt trước và mặt sau.',
  session_expired: 'Phiên đã hết hạn. Tải lại trang để tiếp tục.',
};

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  const tunnelHeaders = {
    'bypass-tunnel-reminder': 'true',
    'ngrok-skip-browser-warning': 'true',
  };
  try {
    response = await fetch(API_BASE + '/api/v1' + path, {
      ...init,
      headers: init?.body instanceof FormData
        ? { ...tunnelHeaders, ...init.headers }
        : { 'Content-Type': 'application/json', ...tunnelHeaders, ...init?.headers },
    });
  } catch {
    throw new Error('Chưa kết nối được máy chủ backend. Kiểm tra VITE_API_URL hoặc khởi chạy server.');
  }

  const contentType = response.headers.get('content-type') || '';
  if (response.status === 405 || (response.ok && contentType.includes('text/html'))) {
    throw new Error('Chưa kết nối máy chủ API (Vercel tĩnh). Cần cấu hình biến môi trường VITE_API_URL trỏ đến server backend Python.');
  }

  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(errorLabels[body.detail] || (typeof body.detail === 'string' ? body.detail : `Yêu cầu chưa thành công (${response.status}).`));
  }
  return response.json();
}

type Store = {
  me: Member | null;
  ready: boolean;
  error: string;
  backendOnline: boolean;
  refresh: () => Promise<void>;
  saved: string[];
  toggleSaved: (id: string) => void;
  cart: string[];
  addCart: (id: string) => void;
  removeCart: (id: string) => void;
  clearCart: () => void;
};
const Context = createContext<Store | null>(null);

function stored(key: string): string[] {
  try {
    const value: unknown = JSON.parse(localStorage.getItem(key) || '[]');
    return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string').slice(0, 100) : [];
  } catch { return []; }
}

export function Provider({ children }: { children: ReactNode }) {
  const [me, setMe] = useState<Member | null>(null);
  const [ready, setReady] = useState(false);
  const [backendOnline, setBackendOnline] = useState(true);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(() => stored('pkmlens-saved'));
  const [cart, setCart] = useState(() => stored('pkmlens-cart'));

  async function refresh() {
    try {
      setMe(await api<Member>('/me'));
      setBackendOnline(true);
      setError('');
    } catch {
      setBackendOnline(false);
      setError('Chưa kết nối được máy chủ backend.');
    }
  }

  useEffect(() => {
    let active = true;
    api('/session', { method: 'POST' })
      .then(() => api<Member>('/me'))
      .then(member => {
        if (active) {
          setMe(member);
          setBackendOnline(true);
          setReady(true);
        }
      })
      .catch(() => {
        if (active) {
          // Backend is offline or static hosting without proxy (Vercel)
          setBackendOnline(false);
          setReady(true);
        }
      });
    return () => { active = false; };
  }, []);

  useEffect(() => { localStorage.setItem('pkmlens-saved', JSON.stringify(saved)); }, [saved]);
  useEffect(() => { localStorage.setItem('pkmlens-cart', JSON.stringify(cart)); }, [cart]);

  return <Context.Provider value={{
    me, ready, error, backendOnline, refresh, saved,
    toggleSaved: id => setSaved(items => items.includes(id) ? items.filter(item => item !== id) : [...items, id]),
    cart, addCart: id => setCart(items => [...new Set([...items, id])].slice(0, 20)),
    removeCart: id => setCart(items => items.filter(item => item !== id)), clearCart: () => setCart([]),
  }}>{children}</Context.Provider>;
}

export function useStore() {
  const value = useContext(Context);
  if (!value) throw new Error('Provider missing');
  return value;
}

export const money = (value: number | null | undefined) => value == null ? 'Chưa có giá' : new Intl.NumberFormat('vi-VN', { style: 'currency', currency: 'VND', maximumFractionDigits: 0 }).format(value);
export const conditionName: Record<string, string> = { NM: 'Gần như mới', LP: 'Xước nhẹ', MP: 'Đã sử dụng', HP: 'Mòn nhiều', DMG: 'Hư hỏng' };
