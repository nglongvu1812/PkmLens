import { useState, type FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Camera, SignOut, Storefront, UserCircle } from '@phosphor-icons/react';
import { api, useStore } from '../api';

export default function Account() {
  const { me, ready, refresh } = useStore();
  const [register, setRegister] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault(); setBusy(true); setError('');
    try { await api(register ? '/auth/register' : '/auth/login', { method: 'POST', body: JSON.stringify({ email, password, display_name: name || 'Người sưu tập' }) }); await refresh(); setPassword(''); }
    catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }
  async function logout() { await api('/auth/logout', { method: 'POST' }); await api('/session', { method: 'POST' }); await refresh(); }
  if (me?.registered) return <section className="account-page"><div className="account-welcome"><span className="large-avatar">{me.display_name?.[0]?.toUpperCase()}</span><h1>Chào {me.display_name}.</h1><p>Bộ sưu tập nào sẽ có thêm một câu chuyện hôm nay?</p><div className="hero-buttons"><Link className="button primary" to="/sell"><Storefront size={20} />Góc người bán</Link><Link className="button secondary" to="/scan"><Camera size={20} />Quét thẻ</Link></div><button className="text-button logout-button" onClick={logout}><SignOut size={18} />Đăng xuất</button></div></section>;
  return <section className="auth-layout"><div className="auth-intro"><span className="auth-symbol"><CardsIcon /></span><h1>Cùng một<br />niềm đam mê.</h1><p>Thêm những lá thẻ bạn yêu. Kết nối những người sưu tập. Và tìm điều bất ngờ tiếp theo.</p><Link to="/">Tiếp tục khám phá<ArrowRight size={19} /></Link></div><form onSubmit={submit} className="auth-form"><UserCircle size={35} /><h2>{register ? 'Bắt đầu bộ sưu tập của bạn.' : 'Chào mừng trở lại.'}</h2><p>{register ? 'Một tài khoản cho mọi lá thẻ bạn yêu.' : 'Đăng nhập để lưu thẻ và mở gian hàng.'}</p>{register && <label>Tên hiển thị<input value={name} onChange={e => setName(e.target.value)} required maxLength={80} autoComplete="nickname" placeholder="Mọi người sẽ gọi bạn là gì?" /></label>}<label>Email<input type="email" required autoComplete="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="ban@email.com" /></label><label>Mật khẩu<input type="password" required minLength={12} maxLength={128} autoComplete={register ? 'new-password' : 'current-password'} value={password} onChange={e => setPassword(e.target.value)} placeholder="Ít nhất 12 ký tự" /></label>{error && <p className="form-error" role="alert">{error}</p>}<button className="button primary full-width" disabled={busy || !ready}>{busy ? 'Đang xử lý…' : register ? 'Tạo tài khoản' : 'Đăng nhập'}<ArrowRight size={19} /></button><p className="auth-switch">{register ? 'Đã có tài khoản?' : 'Lần đầu ghé PkmLens?'} <button type="button" onClick={() => { setRegister(!register); setError(''); }}>{register ? 'Đăng nhập' : 'Đăng ký ngay'}</button></p><small className="auth-footnote">Phiên bản đồ án · Chỉ giao dịch mô phỏng.</small></form></section>;
}

function CardsIcon() { return <svg width="80" height="90" viewBox="0 0 80 90" fill="none" aria-hidden="true"><rect x="9" y="13" width="47" height="64" rx="7" transform="rotate(-12 9 13)" stroke="currentColor" strokeWidth="3" /><rect x="27" y="10" width="47" height="64" rx="7" transform="rotate(9 27 10)" fill="#fff8eb" stroke="currentColor" strokeWidth="3" /><path d="m41 38 8-6 7 9-8 13-7-16Z" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" /></svg>; }
