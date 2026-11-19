import { useEffect, useState, type FormEvent } from 'react';
import { Link, NavLink, Route, Routes, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowRight, Camera, Cards, Heart, MagnifyingGlass, Plus, ShoppingBag, UserCircle, Storefront, ArrowUpRight, GithubLogo } from '@phosphor-icons/react';
import { useStore } from './api';
import Market from './pages/Market';
import Detail from './pages/Detail';
import Account from './pages/Account';
import Sell from './pages/Sell';
import CameraPage from './pages/Camera';
import Cart from './pages/Cart';

function Brand() {
  return <Link to="/" className="brand" aria-label="PkmLens — Trang chủ"><span className="brand-icon"><Camera size={25} weight="bold" /><i /></span><span><b>pkm<span>lens</span><span className="brand-period">.</span></b><small>Mỗi lá thẻ, một câu chuyện.</small></span></Link>;
}

export default function App() {
  const { me, saved, cart, error } = useStore();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();
  const [query, setQuery] = useState(params.get('q') || '');
  useEffect(() => { setQuery(params.get('q') || ''); }, [params]);
  useEffect(() => { window.scrollTo({ top: 0 }); document.querySelector('.app-scroll')?.scrollTo({ top: 0 }); }, [location.pathname]);
  function search(event: FormEvent) { event.preventDefault(); navigate('/?q=' + encodeURIComponent(query)); }
  return <>
    <div className="app-scroll">
    <a href="#main" className="skip-link">Đến nội dung chính</a>
    <header className="site-header">
      <div className="header-main wrap"><Brand /><form className="global-search" onSubmit={search} role="search"><MagnifyingGlass size={22} aria-hidden="true" /><input aria-label="Tìm thẻ, bộ thẻ hoặc Pokémon" placeholder="Tìm thẻ, bộ thẻ hoặc Pokémon..." value={query} onChange={e => setQuery(e.target.value)} /><button aria-label="Tìm kiếm"><ArrowRight size={20} /></button></form>
        <div className="header-actions"><Link className="header-action saved-link" to="/saved" aria-label={`Đã lưu, ${saved.length} thẻ`}><Heart size={23} /><span>Đã lưu</span>{saved.length > 0 && <small>{saved.length}</small>}</Link><Link className="header-action bag-link" to="/cart" aria-label={`Giỏ hàng, ${cart.length} thẻ`}><ShoppingBag size={24} />{cart.length > 0 && <small>{cart.length}</small>}</Link><Link className="account-link" to="/account" aria-label={me?.display_name ? `Tài khoản ${me.display_name}` : 'Đăng nhập'}><span className="avatar">{me?.display_name?.[0]?.toUpperCase() || <UserCircle size={22} />}</span><span>{me?.display_name || 'Đăng nhập'}</span></Link></div>
      </div>
      <div className="nav-border"><div className="nav-row wrap"><nav aria-label="Điều hướng chính"><NavLink end to="/">Khám phá thẻ</NavLink><NavLink to="/market">Chợ thẻ</NavLink><NavLink to="/saved">Bộ sưu tập</NavLink><NavLink to="/scan" className="scan-nav"><Camera size={18} /> Quét camera AI <span className="new-badge">MỚI</span></NavLink></nav><Link to="/sell" className="sell-link"><Plus size={17} weight="bold" /> Đăng bán thẻ</Link></div></div>
    </header>
    <main id="main" className="main-wrap wrap">{error && <div role="alert" className="error-state">{error} <button onClick={() => window.location.reload()}>Thử lại</button></div>}<Routes>
      <Route path="/" element={<Market />} /><Route path="/market" element={<Market salesOnly />} /><Route path="/saved" element={<Market savedOnly />} />
      <Route path="/cards/:id" element={<Detail />} /><Route path="/listings/:id" element={<Detail listing />} />
      <Route path="/account" element={<Account />} /><Route path="/sell" element={<Sell />} /><Route path="/scan" element={<CameraPage />} /><Route path="/cart" element={<Cart />} />
      <Route path="*" element={<div className="empty-state"><Cards size={48} /><h1>Chưa tìm thấy trang này</h1><Link className="button primary" to="/">Về trang khám phá <ArrowRight /></Link></div>} />
    </Routes></main>
    <footer className="site-footer"><div className="wrap footer-content"><div><Brand /><p>Khám phá, kết nối và viết tiếp bộ sưu tập của bạn.</p></div><div className="footer-links"><Link to="/market">Chợ thẻ <ArrowUpRight /></Link><Link to="/scan">Camera AI <ArrowUpRight /></Link><Link to="/sell">Góc người bán <ArrowUpRight /></Link></div><div className="footer-note"><p>Đồ án môn Học Máy · Giao dịch mô phỏng</p><small>Dữ liệu: <a href="https://github.com/cardcorpus/pokemon-card-dataset" target="_blank" rel="noreferrer">CardCorpus <GithubLogo size={13} /></a> & TCGdex.<br />Pokémon và hình ảnh thẻ thuộc các chủ sở hữu tương ứng.</small></div></div></footer>
    </div>
    <nav className="mobile-nav" aria-label="Điều hướng di động"><NavLink end to="/"><Storefront size={23} /><span>Khám phá</span></NavLink><NavLink to="/saved"><Heart size={23} /><span>Đã lưu</span></NavLink><NavLink to="/scan" className="mobile-scan"><span><Camera size={25} weight="bold" /></span><b>Quét AI</b></NavLink><NavLink to="/sell"><Plus size={24} /><span>Đăng bán</span></NavLink><NavLink to="/account"><UserCircle size={24} /><span>Tài khoản</span></NavLink></nav>
  </>;
}
