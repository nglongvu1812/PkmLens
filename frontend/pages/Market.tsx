import { useEffect, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { ArrowRight, Camera, CaretDown, Cards, Check, GlobeHemisphereWest, Heart, List, MagnifyingGlass, SlidersHorizontal, SquaresFour, X } from '@phosphor-icons/react';
import { api, useStore } from '../api';
import type { Card, Listing, SetInfo } from '../types';
import CardTile, { CardArt } from '../components/CardTile';
import { languageLabel } from '../components/LanguageLabel';

export default function Market({ salesOnly = false, savedOnly = false }: { salesOnly?: boolean; savedOnly?: boolean }) {
  const { ready, saved } = useStore();
  const [params, setParams] = useSearchParams();
  const q = params.get('q') || '';
  const setId = params.get('set') || '';
  const language = params.get('lang') || '';
  const [cards, setCards] = useState<Card[]>([]);
  const [sales, setSales] = useState<Listing[]>([]);
  const [sets, setSets] = useState<SetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState({ scope: '', offset: 0 });
  const [sort, setSort] = useState('featured');
  const [rare, setRare] = useState(false);
  const [condition, setCondition] = useState('');
  const [view, setView] = useState('grid');
  const [filterOpen, setFilterOpen] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const pageSize = 24;
  const scope = [q, setId, language, sort, rare, condition, salesOnly, savedOnly, savedOnly ? saved.join(',') : ''].join('\0');
  const offset = page.scope === scope ? page.offset : 0;
  function setOffset(value: number) { setPage({ scope, offset: value }); }
  useEffect(() => {
    if (!ready) return;
    api<SetInfo[]>('/shop/sets').then(setSets).catch(() => {});
  }, [ready]);
  useEffect(() => {
    if (!ready) return;
    const controller = new AbortController();
    setLoading(true); setError('');
    const query = new URLSearchParams({ q, limit: String(pageSize), offset: String(offset) });
    query.set('sort', sort);
    if (setId) query.set('set_id', setId);
    if (language && !salesOnly) query.set('language', language);
    if (savedOnly) query.set('ids', saved.join(','));
    if (rare && !salesOnly) query.set('rare', 'true');
    const path = salesOnly ? '/shop/listings?' + query : '/shop/catalog?' + query;
    api<{ items: Card[]; total: number } | Listing[]>(path, { signal: controller.signal }).then(result => {
      if (controller.signal.aborted) return;
      if (Array.isArray(result)) { setSales(result); setTotal(result.length); }
      else { setCards(result.items); setTotal(result.total); }
    }).catch(reason => { if (!controller.signal.aborted) setError(reason.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [q, setId, language, offset, ready, salesOnly, savedOnly, saved, rare, sort]);
  useEffect(() => { if (filterOpen) dialog.current?.showModal(); else dialog.current?.close(); }, [filterOpen]);
  function updateSet(value: string) { const next = new URLSearchParams(params); if (value) next.set('set', value); else next.delete('set'); setOffset(0); setParams(next); }
  function updateLanguage(value: string) { const next = new URLSearchParams(params); if (value) next.set('lang', value); else next.delete('lang'); setOffset(0); setParams(next); }
  function clear() { setParams({}); setRare(false); setCondition(''); setOffset(0); }
  const languages = [...new Set(['en', 'ja', 'zh-Hans', ...(language ? [language] : []), ...sets.map(set => set.language)])];
  const items = (salesOnly ? sales.map(sale => ({ card: sale.card, sale })) : cards.map(card => ({ card, sale: undefined as Listing | undefined })))
    .filter(({ card, sale }) => (!savedOnly || saved.includes(card.id)) && (!rare || /rare|illustration|ultra|hyper/i.test(card.rarity)) && (!condition || sale?.condition === condition));
  const sorted = salesOnly ? [...items].sort((a, b) => {
    const aa = a.sale?.asking_vnd ?? a.card.reference_price?.amount_vnd;
    const bb = b.sale?.asking_vnd ?? b.card.reference_price?.amount_vnd;
    if (sort === 'price-up') return (aa ?? Infinity) - (bb ?? Infinity);
    if (sort === 'price-down') return (bb ?? -1) - (aa ?? -1);
    if (sort === 'name') return a.card.name.localeCompare(b.card.name);
    const rank = (card: Card) => (card.source === 'CardCorpus' ? 100 : 0) + (/Charizard|Pikachu|Blastoise|Venusaur|Mew ex/.test(card.name) ? 10 : 0) + (/Illustration/.test(card.rarity) ? 5 : 0);
    return rank(b.card) - rank(a.card);
  }) : items;
  function filters() { return <>
    <div className="filter-heading"><h2>Bộ lọc</h2><button className="text-button" onClick={clear}>Xóa tất cả</button></div>
    <fieldset><legend>Loại thẻ</legend><label className="check-row"><span><input type="checkbox" checked disabled /> Thẻ đơn Pokémon</span><span>{total}</span></label>
      <label className="check-row"><span><input type="checkbox" checked={rare} onChange={e => { setRare(e.target.checked); setOffset(0); }} /> Rare & Illustration</span></label>
    </fieldset>
    <label className="filter-label">Bộ thẻ<select aria-label="Bộ thẻ" value={setId} onChange={e => updateSet(e.target.value)}><option value="">Tất cả bộ thẻ</option>{sets.map(set => <option key={set.id} value={set.id}>{set.name} ({set.count}){set.language && <> · {languageLabel(set.language)}</>}</option>)}</select></label>
    {!salesOnly && <label className="filter-label">Ngôn ngữ<select aria-label="Ngôn ngữ" value={language} onChange={e => updateLanguage(e.target.value)}><option value="">Tất cả ngôn ngữ</option>{languages.map(value => <option key={value} value={value}>{languageLabel(value)}</option>)}</select></label>}
    {salesOnly && <label className="filter-label">Tình trạng<select aria-label="Tình trạng" value={condition} onChange={e => setCondition(e.target.value)}><option value="">Tất cả tình trạng</option><option value="NM">Gần như mới</option><option value="LP">Xước nhẹ</option><option value="MP">Đã sử dụng</option><option value="HP">Mòn nhiều</option><option value="DMG">Hư hỏng</option></select></label>}
    <div className="filter-note"><Cards size={26} /><strong>Tìm đúng lá thẻ của bạn.</strong><p>Thử camera AI để nhận diện thẻ trong bộ sưu tập.</p><Link to="/scan">Mở máy quét <ArrowRight size={16} /></Link></div>
  </>; }
  return <>
    {!savedOnly && !salesOnly ? <section className="market-hero"><div className="hero-text"><h1>Lá thẻ tiếp theo.<br /><span>Câu chuyện của bạn.</span></h1><p>Một góc nhỏ cho niềm đam mê lớn.<br className="desktop-break" /> Khám phá thẻ Pokémon, tìm đúng bản in và kết nối người sưu tập.</p><div className="hero-buttons"><Link className="button primary" to="/scan"><Camera size={20} />Quét thẻ bằng AI<ArrowUpRightIcon /></Link><Link className="hero-secondary" to="/market">Khám phá chợ thẻ <ArrowRight size={18} /></Link></div></div><div className="hero-art" aria-label="Thẻ Pokémon Base Set"><div className="hero-orbit" /><Link to="/cards/base1%3A3%3Aen" className="hero-card card-back"><CardArt src="/api/v1/cards/base1:3:en/image" name="Chansey Base Set" eager /></Link><Link to="/cards/base1%3A4%3Aen" className="hero-card card-front"><CardArt src="/api/v1/cards/base1:4:en/image" name="Charizard Base Set" eager /></Link><div className="hero-caption"><span><Check size={14} weight="bold" /></span> Sưu tập điều bạn yêu.</div><span className="hero-small-star star-one">✦</span><span className="hero-small-star star-two">✧</span></div></section> : <div className="page-heading"><h1>{savedOnly ? 'Những lá thẻ bạn yêu.' : 'Từ người sưu tập, đến người sưu tập.'}</h1><p>{savedOnly ? 'Lưu lại cảm hứng cho bộ sưu tập tiếp theo.' : 'Những tin đăng đang mở từ cộng đồng PkmLens.'}</p></div>}
    <div className="discovery-tabs"><div><Link className={!salesOnly && !savedOnly ? 'selected' : ''} to="/"><Cards size={19} />Khám phá catalog</Link><Link className={salesOnly ? 'selected' : ''} to="/market">Đang mở bán{sales.length > 0 && <small>{sales.length}</small>}</Link></div><span className="source-note"><GlobeHemisphereWest size={16} /> Dữ liệu CardCorpus & TCGdex</span></div>
    <div className="market-layout"><aside className="filter-sidebar" aria-label="Lọc thẻ">{filters()}</aside><section className="results-area" aria-label="Kết quả thẻ"><div className="results-toolbar"><div><h2>{savedOnly ? `${sorted.length} thẻ đã lưu` : `${total.toLocaleString('vi-VN')} ${salesOnly ? 'tin đăng' : 'lá thẻ để khám phá'}`}</h2><p>{q ? `Kết quả cho “${q}”` : salesOnly ? 'Giá và tình trạng do người bán cung cấp.' : 'Từ những bản in đầu tiên đến những nét vẽ bạn yêu.'}</p></div><div className="result-controls"><button className="icon-button mobile-filter" aria-label="Mở bộ lọc" onClick={() => setFilterOpen(true)}><SlidersHorizontal size={21} /><span>Lọc</span></button><label className="sort-control"><span className="sr-only">Sắp xếp thẻ</span><select value={sort} onChange={e => setSort(e.target.value)}><option value="featured">Nổi bật trước</option><option value="name">Tên A → Z</option><option value="price-up">Giá thấp → cao</option><option value="price-down">Giá cao → thấp</option></select><CaretDown size={14} /></label><div className="view-controls"><button aria-label="Xem dạng lưới" aria-pressed={view === 'grid'} className={view === 'grid' ? 'active' : ''} onClick={() => setView('grid')}><SquaresFour size={20} /></button><button aria-label="Xem dạng danh sách" aria-pressed={view === 'list'} className={view === 'list' ? 'active' : ''} onClick={() => setView('list')}><List size={21} /></button></div></div></div>
    {(q || setId || language || rare) && <div className="active-filters"><span>Đang lọc:</span>{q && <button onClick={() => { const next = new URLSearchParams(params); next.delete('q'); setParams(next); }}>{q}<X size={13} /></button>}{setId && <button onClick={() => updateSet('')}>{sets.find(set => set.id === setId)?.name}<X size={13} /></button>}{language && <button onClick={() => updateLanguage('')}>{languageLabel(language)}<X size={13} /></button>}{rare && <button onClick={() => setRare(false)}>Rare & Illustration<X size={13} /></button>}</div>}
    {error ? <div className="error-state" role="alert">{error}<button onClick={() => window.location.reload()}>Thử lại</button></div> : loading || !ready ? <div className="product-grid" aria-label="Đang tải thẻ" aria-busy="true">{Array.from({ length: 8 }, (_, i) => <div className="tile-skeleton" key={i}><div /><span /><span /></div>)}</div> : sorted.length ? <div className={'product-grid' + (view === 'list' ? ' list-view' : '')}>{sorted.map(item => <CardTile key={item.sale?.id || item.card.id} {...item} />)}</div> : <div className="empty-state">{savedOnly ? <Heart size={44} /> : <MagnifyingGlass size={44} />}<h3>{salesOnly ? 'Lá thẻ đầu tiên đang chờ bạn.' : savedOnly ? 'Bộ sưu tập bắt đầu từ một cái chạm.' : 'Chưa tìm thấy lá thẻ phù hợp.'}</h3><p>{salesOnly ? 'Chưa có tin đang mở bán. Đăng lá thẻ của bạn để bắt đầu.' : savedOnly ? 'Chạm biểu tượng trái tim trên lá thẻ để lưu vào đây.' : 'Thử một từ khóa khác hoặc bỏ bớt bộ lọc.'}</p><Link className="button primary" to={salesOnly ? '/sell' : '/'}>{salesOnly ? 'Đăng bán thẻ' : 'Khám phá thẻ'}<ArrowRight /></Link></div>}
     {!salesOnly && total > pageSize && <div className="pagination"><button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - pageSize))}>Trang trước</button><span>{Math.floor(offset / pageSize) + 1} / {Math.ceil(total / pageSize)}</span><button disabled={offset + pageSize >= total} onClick={() => setOffset(offset + pageSize)}>Trang tiếp</button></div>}</section></div>
    <section className="camera-promo"><div className="promo-camera"><Camera size={30} /></div><div><h2>Không nhớ tên thẻ? Cứ đưa lên camera.</h2><p>Nhận diện lá thẻ, đối chiếu bản in và xem giá tham khảo.</p></div><Link className="button secondary" to="/scan">Thử quét thẻ <ArrowRight size={18} /></Link></section>
    <dialog ref={dialog} className="filter-dialog" onClose={() => setFilterOpen(false)}><div className="dialog-top"><h2>Lọc thẻ</h2><button className="icon-button" onClick={() => setFilterOpen(false)} aria-label="Đóng bộ lọc"><X size={22} /></button></div>{filters()}<button className="button primary full-width" onClick={() => setFilterOpen(false)}>Xem kết quả<ArrowRight /></button></dialog>
  </>;
}

function ArrowUpRightIcon() { return <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M6 18 18 6M6 6h12v12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>; }
