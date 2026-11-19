import { useEffect, useRef, useState, type FormEvent } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowRight, Camera, CheckCircle, Image, Plus, Storefront, UploadSimple } from '@phosphor-icons/react';
import { api, conditionName, money, useStore } from '../api';
import type { Card, Listing } from '../types';
import { CardArt } from '../components/CardTile';
import { languageLabel } from '../components/LanguageLabel';

export default function Sell() {
  const { me, ready } = useStore();
  const [params] = useSearchParams();
  const [cards, setCards] = useState<Card[]>([]);
  const [mine, setMine] = useState<Listing[]>([]);
  const [selected, setSelected] = useState(params.get('card') || '');
  const [selectedCard, setSelectedCard] = useState<Card | null>(null);
  const [variant, setVariant] = useState('');
  const [condition, setCondition] = useState('NM');
  const [asking, setAsking] = useState('');
  const [front, setFront] = useState<File | null>(null);
  const [back, setBack] = useState<File | null>(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [busy, setBusy] = useState(false);
  const [query, setQuery] = useState('');
  const draft = useRef<{ id: string; signature: string } | null>(null);
  const card = selectedCard;

  useEffect(() => {
    if (!ready) return;
    const controller = new AbortController();
    api<{ items: Card[] }>('/shop/catalog?limit=60&q=' + encodeURIComponent(query), { signal: controller.signal })
      .then(value => { if (!controller.signal.aborted) setCards(value.items); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [ready, query]);

  useEffect(() => {
    if (!ready || !selected) { setSelectedCard(null); return; }
    const controller = new AbortController();
    api<{ card: Card }>('/shop/cards/' + encodeURIComponent(selected), { signal: controller.signal })
      .then(value => {
        if (controller.signal.aborted) return;
        setSelectedCard(value.card);
        const requestedVariant = params.get('variant');
        const cardMatchesRequest = params.get('card') === value.card.id;
        setVariant(cardMatchesRequest && value.card.variants.some(option => option.id === requestedVariant) ? requestedVariant || '' : '');
      }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); });
    return () => controller.abort();
  }, [selected, ready, params]);

  useEffect(() => {
    if (ready && me?.registered) api<Listing[]>('/shop/mine').then(setMine).catch(reason => setError(reason.message));
  }, [ready, me, success]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!card || !card.variants.some(option => option.id === variant) || !front || !back) return;
    if (front.size > 8 * 1024 * 1024 || back.size > 8 * 1024 * 1024) {
      setError('Mỗi ảnh tối đa 8 MB. Chọn ảnh nhỏ hơn rồi thử lại.'); return;
    }
    setBusy(true); setError(''); setSuccess('');
    try {
      const body = { card_id: card.id, variant, condition, asking_vnd: Number(asking) };
      const signature = JSON.stringify(body);
      if (!draft.current || draft.current.signature !== signature) {
        const result = await api<{ id: string }>('/listings/manual', { method: 'POST', body: signature });
        draft.current = { id: result.id, signature };
      }
      const photos = new FormData(); photos.append('front', front); photos.append('back', back);
      await api('/listings/' + draft.current.id + '/photos', { method: 'POST', body: photos });
      const result = await api<{ status: string }>('/listings/' + draft.current.id + '/publish', {
        method: 'POST', body: JSON.stringify({ asking_vnd: Number(asking) }),
      });
      draft.current = null;
      setSuccess(result.status === 'active'
        ? 'Lá thẻ của bạn đã lên chợ. Chúc bạn tìm được người sưu tập tiếp theo!'
        : 'Đã gửi tin đăng. Ảnh cần được kiểm tra thêm trước khi xuất hiện trên chợ.');
    } catch (reason) { setError((reason as Error).message); }
    finally { setBusy(false); }
  }

  const options = card && !cards.some(value => value.id === card.id) ? [card, ...cards] : cards;
  return <>
    <div className="page-heading"><h1>Một lá thẻ. Một người bạn mới.</h1><p>Trao lá thẻ của bạn cho người sưu tập tiếp theo.</p></div>
    {!me?.registered ? <div className="empty-state"><Storefront size={44} /><h2>Mở góc sưu tập của bạn.</h2>
      <p>Đăng nhập để tạo và quản lý các tin bán thẻ.</p><Link className="button primary" to="/account">Đăng nhập hoặc đăng ký<ArrowRight /></Link>
    </div> : <div className="seller-layout">
      <form className="seller-form panel" onSubmit={submit}>
        <h2>Đăng bán thẻ</h2>
        <Link className="scan-assist" to="/scan"><Camera size={23} /><span>Không nhớ chính xác bản in?<small>Để camera AI tìm giúp bạn.</small></span><ArrowRight size={20} /></Link>
        <label>Tìm thẻ trong catalog<input placeholder="Nhập tên Pokémon hoặc số thẻ…" value={query} onChange={e => setQuery(e.target.value)} /></label>
        <label htmlFor="sell-card">Chọn lá thẻ</label>
        <select id="sell-card" aria-label="Chọn lá thẻ" required value={selected} onChange={e => { setVariant(''); setSelected(e.target.value); }}>
          <option value="">Chọn thẻ để đăng bán</option>
          {options.map(value => <option key={value.id} value={value.id}>{value.name} · {value.set_name} #{value.number} · {languageLabel(value.language)}</option>)}
        </select>
        <div className="form-grid seller-variants">
          <div><label htmlFor="sell-variant">Biến thể</label><select id="sell-variant" aria-label="Biến thể" required value={variant} onChange={e => setVariant(e.target.value)}>
            <option value="">Chọn biến thể</option>{card?.variants.map(value => <option key={value.id} value={value.id}>{value.label}</option>)}
          </select></div>
          <div><label htmlFor="sell-condition">Tình trạng</label><select id="sell-condition" aria-label="Tình trạng" value={condition} onChange={e => setCondition(e.target.value)}>
            {Object.entries(conditionName).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select></div>
        </div>
        <label>Giá mong muốn (VND)<input required type="number" min={1} max={2000000000} value={asking} onChange={e => setAsking(e.target.value)} placeholder="Ví dụ: 250000" /></label>
        <div className="photo-grid">{(['front', 'back'] as const).map(side => <label className="photo-picker" key={side}>
          <UploadSimple size={27} /><strong>{side === 'front' ? 'Ảnh mặt trước' : 'Ảnh mặt sau'}</strong>
          <span>{(side === 'front' ? front : back)?.name || 'JPG, PNG hoặc WebP · Tối đa 8 MB'}</span>
          <input type="file" accept="image/jpeg,image/png,image/webp" required onChange={e => (side === 'front' ? setFront : setBack)(e.target.files?.[0] || null)} />
        </label>)}</div>
        <p className="form-hint">Ảnh tin đăng không được dùng để train model nếu chưa có sự đồng ý riêng.</p>
        {error && <p role="alert" className="form-error">{error}</p>}
        {success && <p role="status" className="form-success"><CheckCircle size={21} />{success}</p>}
        <button className="button primary full-width" disabled={busy || !card || !variant}>{busy ? 'Đang đăng thẻ…' : 'Đăng bán lá thẻ'}<ArrowRight size={19} /></button>
      </form>
      <aside className="seller-preview panel"><h2>Xem trước lá thẻ</h2>{card ? <>
        <div className="preview-art"><CardArt src={card.image} name={card.name} /></div><h3>{card.name}</h3><p>{card.set_name} · #{card.number} · {languageLabel(card.language)}</p>
        <strong className="preview-price">{asking ? money(Number(asking)) : 'Chưa đặt giá'}</strong>
      </> : <div className="preview-empty"><Image size={42} /><p>Chọn một lá thẻ để xem trước.</p></div>}</aside>
    </div>}
    <section className="my-listings"><h2>Những lá thẻ của bạn</h2>{mine.length ? <div className="mine-list">{mine.map(value => <div className="mine-row" key={value.id}>
      <div className="mine-art"><CardArt src={value.card.image} name={value.card.name} /></div>
      <div><strong>{value.card.name}</strong><p>{conditionName[value.condition]} · {money(value.asking_vnd)}</p></div>
      <span className="tag">{{ draft: 'Bản nháp', active: 'Đang bán', pending_review: 'Chờ kiểm tra', sold: 'Đã bán', reserved: 'Đã giữ' }[value.status] || value.status}</span>
      {value.status === 'active' && <Link className="icon-button" aria-label={'Xem ' + value.card.name} to={'/listings/' + value.id}><ArrowRight size={20} /></Link>}
    </div>)}</div> : <div className="inline-empty"><Plus size={22} /><p>Tin đăng của bạn sẽ xuất hiện ở đây.</p></div>}</section>
  </>;
}
