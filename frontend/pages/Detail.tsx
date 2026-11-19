import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Camera, Heart, ShoppingBag, Storefront } from '@phosphor-icons/react';
import { api, conditionName, money, useStore } from '../api';
import type { Card, Listing, Price } from '../types';
import { CardArt } from '../components/CardTile';
import { languageLabel } from '../components/LanguageLabel';
import { FALLBACK_CARDS } from '../fallbackData';

export default function Detail({ listing = false }: { listing?: boolean }) {
  const { id } = useParams();
  const { ready, saved, toggleSaved, addCart, cart } = useStore();
  const [card, setCard] = useState<Card | null>(null);
  const [sale, setSale] = useState<Listing | null>(null);
  const [listings, setListings] = useState<Listing[]>([]);
  const [variant, setVariant] = useState('');
  const [condition, setCondition] = useState('NM');
  const [price, setPrice] = useState<Price | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    if (!ready || !id) return;
    setError(''); setCard(null);
    if (listing) api<Listing & { reference_price: Price }>('/shop/listings/' + encodeURIComponent(id)).then(value => { setSale(value); setCard(value.card); setVariant(value.variant); setCondition(value.condition); setPrice(value.reference_price); }).catch(reason => setError(reason.message));
    else api<{ card: Card; listings: Listing[] }>('/shop/cards/' + encodeURIComponent(id)).then(value => { setCard(value.card); setListings(value.listings); setVariant(value.card.variants[0]?.id || ''); }).catch(reason => {
      const decoded = decodeURIComponent(id);
      const fallback = FALLBACK_CARDS.find(item => item.id === id || item.id === decoded || item.id.replace(/:/g, '-') === id || item.id.replace(/-/g, ':') === id);
      if (fallback) {
        setCard(fallback);
        setListings([]);
        setVariant(fallback.variants[0]?.id || '');
        setPrice(fallback.reference_price);
      } else {
        setError(reason.message);
      }
    });
  }, [id, ready, listing]);
  useEffect(() => {
    if (!card || !variant) return;
    let active = true;
    api<Price[]>('/prices/batch', { method: 'POST', body: JSON.stringify({ items: [{ card_id: card.id, variant, condition }] }) }).then(value => { if (active && value?.[0]) setPrice(value[0]); }).catch(() => {});
    return () => { active = false; };
  }, [card, variant, condition]);
  return <><Link className="back-link" to={listing ? '/market' : '/'}><ArrowLeft size={17} />Về {listing ? 'chợ thẻ' : 'trang khám phá'}</Link>{error && <div role="alert" className="error-state">{error}</div>}{!card ? <div className="detail-loading">Đang mở lá thẻ…</div> : <><div className="detail-layout"><div className="detail-art"><CardArt src={sale?.image || card.image} name={card.name} eager /><span>{sale ? 'Ảnh từ người bán' : 'Ảnh tham chiếu · ' + card.source}</span></div><div className="detail-copy"><p className="detail-set">{card.set_name} · #{card.number}{card.total ? '/' + card.total : ''}</p><h1>{card.name}</h1><div className="detail-tags"><span className="tag">{card.rarity || 'Pokémon TCG'}</span><span className="tag neutral">{languageLabel(card.language)}</span><span className="tag neutral">Thẻ đơn</span></div><div className="detail-price-section"><span>{sale ? 'Giá người bán' : 'Giá tham khảo'}</span><strong>{money(sale?.asking_vnd ?? price?.amount_vnd)}</strong>{price?.amount_vnd != null ? <p>{price.source} · {price.source_at ? new Date(price.source_at).toLocaleDateString('vi-VN') : ''}{price.price_stale || price.fx_stale ? ' · Dữ liệu bản lưu' : ''}</p> : <p>Chưa có giá cho bản in này. Bạn vẫn có thể tự đặt giá khi đăng bán.</p>}</div><div className="form-grid"><label>Biến thể<select value={variant} onChange={e => setVariant(e.target.value)} disabled={listing}>{card.variants.map(value => <option key={value.id} value={value.id}>{value.label}</option>)}</select></label><label>Tình trạng<select value={condition} onChange={e => setCondition(e.target.value)} disabled={listing}>{Object.entries(conditionName).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label></div><div className="detail-actions">{sale && !sale.is_owner ? <button className="button primary" disabled={cart.includes(sale.id) || sale.status !== 'active'} onClick={() => addCart(sale.id)}><ShoppingBag size={20} />{cart.includes(sale.id) ? 'Đã thêm vào giỏ' : 'Thêm vào giỏ hàng'}</button> : <Link className="button primary" to={'/sell?card=' + encodeURIComponent(card.id)}>Đăng bán lá thẻ này<ArrowRight size={19} /></Link>}<button className="button secondary" aria-pressed={saved.includes(card.id)} onClick={() => toggleSaved(card.id)}><Heart size={21} weight={saved.includes(card.id) ? 'fill' : 'regular'} />{saved.includes(card.id) ? 'Đã lưu' : 'Lưu thẻ'}</button></div>{sale && <div className="seller-info"><Storefront size={24} /><div><strong>{sale.seller}</strong><small>Người bán trên PkmLens</small></div></div>}<div className="detail-note"><Camera size={20} /><p>Đang cầm lá thẻ này? <Link to="/scan">Thử nhận diện bằng camera AI.</Link></p></div><dl className="card-facts"><div><dt>Mã thẻ</dt><dd>{card.id}</dd></div><div><dt>Nguồn dữ liệu</dt><dd>{card.source}</dd></div><div><dt>Giá tham khảo</dt><dd>Theo biến thể, tình trạng giả định</dd></div></dl></div></div>{!listing && <section className="card-sellers"><h2>Đang bán lá thẻ này</h2>{listings.length ? listings.map(value => <Link key={value.id} className="sale-row" to={'/listings/' + value.id}><span><Storefront size={20} />{value.seller}</span><span>{conditionName[value.condition]}</span><strong>{money(value.asking_vnd)}</strong><ArrowRight size={20} /></Link>) : <div className="inline-empty"><p>Chưa có ai đăng bán bản in này.</p><Link to={'/sell?card=' + encodeURIComponent(card.id)}>Trở thành người bán đầu tiên <ArrowRight size={16} /></Link></div>}</section>}</>}</>;
}
