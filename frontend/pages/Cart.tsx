import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle, ShoppingBag, Trash } from '@phosphor-icons/react';
import { api, money, useStore } from '../api';
import type { Listing } from '../types';
import { CardArt } from '../components/CardTile';

export default function Cart() {
  const { cart, removeCart, clearCart, ready, me } = useStore();
  const [items, setItems] = useState<Listing[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  useEffect(() => { if (!ready) return; Promise.all(cart.map(id => api<Listing>('/shop/listings/' + id))).then(setItems).catch(reason => setError(reason.message)); }, [cart, ready]);
  async function checkout() {
    setBusy(true); setError('');
    try {
      await api('/wallet/demo-credit', { method: 'POST' });
      const order = await api<{ id: string }>('/orders', { method: 'POST', body: JSON.stringify({ listing_ids: cart, request_key: crypto.randomUUID() }) });
      await api('/orders/' + order.id + '/pay', { method: 'POST' });
      clearCart(); setDone(true);
    } catch (reason) { setError((reason as Error).message); } finally { setBusy(false); }
  }
  return <><div className="page-heading"><h1>Những lá thẻ sắp về nhà.</h1><p>Giỏ hàng của bạn · Giao dịch mô phỏng cho đồ án</p></div>{error && <p className="error-state" role="alert">{error}</p>}{done ? <div className="empty-state"><CheckCircle size={48} /><h2>Đã tạo đơn hàng mô phỏng.</h2><p>Không có tiền thật được chuyển. Giao dịch đã được lưu trong hệ thống demo.</p><Link className="button primary" to="/">Tiếp tục khám phá<ArrowRight /></Link></div> : !cart.length ? <div className="empty-state"><ShoppingBag size={46} /><h2>Chỗ trống cho một điều đặc biệt.</h2><p>Khám phá chợ thẻ và tìm lá thẻ tiếp theo của bạn.</p><Link className="button primary" to="/market">Ghé chợ thẻ<ArrowRight /></Link></div> : <div className="cart-layout"><div>{items.map(item => <div className="cart-row panel" key={item.id}><div className="cart-art"><CardArt src={item.image} name={item.card.name} /></div><div><h2>{item.card.name}</h2><p>{item.card.set_name}</p><small>{item.seller}</small></div><strong>{money(item.asking_vnd)}</strong><button className="icon-button" aria-label={'Bỏ ' + item.card.name + ' khỏi giỏ'} onClick={() => removeCart(item.id)}><Trash size={20} /></button></div>)}</div><aside className="panel checkout-summary"><h2>Đơn hàng của bạn</h2><div><span>{items.length} lá thẻ</span><strong>{money(items.reduce((sum, item) => sum + (item.asking_vnd || 0), 0))}</strong></div><p>Thanh toán bằng số dư demo. Hệ thống không thu tiền thật.</p>{me?.registered ? <button className="button primary full-width" onClick={checkout} disabled={busy}>{busy ? 'Đang xử lý…' : 'Đặt hàng mô phỏng'}<ArrowRight /></button> : <Link className="button primary full-width" to="/account">Đăng nhập để đặt hàng<ArrowRight /></Link>}</aside></div>}</>;
}
