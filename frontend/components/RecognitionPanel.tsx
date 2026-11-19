import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Check, MagnifyingGlass, Plus, Scan } from '@phosphor-icons/react';
import { api, money } from '../api';
import type { Candidate, Card, Price, SessionCard } from '../types';
import { CardArt } from './CardTile';

export type ScanTrack = {
  track_id: string;
  generation: string;
  box: number[];
  quadrilateral?: number[][] | null;
  confidence: number;
  status: string;
  candidates: Candidate[];
  ocr?: { number?: string; name?: string; text?: string[] };
  quality: { usable: boolean; reasons: string[] };
  votes: number;
  consensus_required?: number;
};
const statuses: Record<string, string> = {
  reading: 'Đang tìm bản in', stabilizing: 'Đang đối chiếu',
  uncertain: 'Chưa đủ bằng chứng', reject: 'Không đủ điều kiện nhận diện', ineligible: 'Không đủ điều kiện nhận diện',
  needs_confirmation: 'Chờ bạn xác nhận', unknown: 'Chưa xác định', quality: 'Cần ảnh rõ hơn',
};

export default function RecognitionPanel({ tracks, selected, onSelect, onSave, stillImage = false }: {
  tracks: ScanTrack[];
  selected?: ScanTrack;
  onSelect: (id: string) => void;
  onSave: (card: SessionCard) => void;
  stillImage?: boolean;
}) {
  const trackKey = selected ? selected.track_id + ':' + selected.generation : '';
  const [correction, setCorrection] = useState<{ trackKey: string; candidate: Candidate } | null>(null);
  const [query, setQuery] = useState('');
  const [matches, setMatches] = useState<Card[]>([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');
  const [variantChoice, setVariantChoice] = useState<{ trackKey: string; cardId: string; id: string } | null>(null);
  const [price, setPrice] = useState<Price | null>(null);
  const manual = correction?.trackKey === trackKey ? correction.candidate : null;
  const ineligible = !!selected && (!selected.quality.usable || ['reject', 'ineligible', 'quality'].includes(selected.status));
  const best = manual || (ineligible || selected?.status === 'unknown' ? undefined : selected?.candidates[0]);
  const required = selected?.consensus_required;
  const consensusRequired = required && Number.isInteger(required) && required > 0 ? required : 3;
  const printingReady = !!manual || (!ineligible && selected?.status === 'needs_confirmation');
  const cardId = best?.card_id;
  const variantId = variantChoice?.trackKey === trackKey && variantChoice.cardId === cardId ? variantChoice.id : '';
  const selectedVariant = variantId ? best?.card?.variants.find(value => value.id === variantId) : undefined;
  const priceVariant = printingReady ? selectedVariant?.id : undefined;
  const quote = priceVariant && price?.variant === priceVariant && price.condition === 'NM' ? price : null;

  useEffect(() => { setCorrection(null); setQuery(''); setMatches([]); setError(''); }, [trackKey]);
  useEffect(() => { setVariantChoice(null); }, [trackKey, cardId]);
  useEffect(() => {
    setPrice(null);
    if (!cardId || !priceVariant) return;
    const controller = new AbortController();
    api<Price[]>('/prices/batch', { signal: controller.signal, method: 'POST', body: JSON.stringify({
      items: [{ card_id: cardId, variant: priceVariant, condition: 'NM' }],
    }) }).then(result => { if (!controller.signal.aborted) setPrice(result[0] ?? null); })
      .catch(() => { if (!controller.signal.aborted) setPrice(null); });
    return () => controller.abort();
  }, [cardId, priceVariant]);
  useEffect(() => {
    if (query.trim().length < 2) { setMatches([]); setSearching(false); return; }
    const controller = new AbortController();
    setSearching(true); setError('');
    api<{ items: Card[] }>('/shop/catalog?limit=6&q=' + encodeURIComponent(query.trim()), { signal: controller.signal })
      .then(result => { if (!controller.signal.aborted) setMatches(result.items); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason.message); })
      .finally(() => { if (!controller.signal.aborted) setSearching(false); });
    return () => controller.abort();
  }, [query]);

  function choose(candidate: Candidate) {
    setCorrection({ trackKey, candidate }); setVariantChoice(null); setQuery(''); setMatches([]);
  }
  return <aside className="recognition-panel panel">
    <div className="recognition-title"><h2>Kết quả nhận diện</h2><span>{tracks.length} thẻ</span></div>
    {selected ? <>
      <div className="recognized-art">{best?.card ? <CardArt src={best.card.image} name={best.card.name} /> : <Scan size={50} />}</div>
      <span className={'recognition-status ' + (manual ? 'needs_confirmation' : selected.status)}>
        {manual ? 'Bản in do bạn chọn' : statuses[selected.status] || selected.status}
      </span>
      <h3>{best?.card?.name || selected.ocr?.name || 'Chưa xác định lá thẻ'}</h3>
      <p className="recognized-set">{best?.card ? `${best.card.set_name} · #${best.card.number}` : 'Văn bản OCR chưa xác minh với catalog'}</p>
      <div className="recognition-evidence">
        <div><span>Độ tương đồng ảnh</span><strong>{manual ? 'Chọn thủ công' : best ? best.similarity.toFixed(3) : 'Chưa xác định'}</strong></div>
        <div><span>Số thẻ OCR</span><strong>{selected.ocr?.number || 'Chưa đọc được'}</strong></div>
        <div><span>Đối chiếu khung hình</span><strong>{selected.votes}/{consensusRequired}</strong></div>
      </div>
      <p className="recognition-measure-note">Điểm tương đồng và điểm detector không phải xác suất nhận diện đúng.</p>
      {stillImage && <p className="recognition-measure-note">Consensus từ cùng một ảnh tĩnh gửi lặp; không phải kiểm chứng camera thật độc lập.</p>}
      {ineligible && <p className="recognition-measure-note">Khung này chưa đủ điều kiện cho định danh AI. Cải thiện ảnh hoặc tự đối chiếu bản in trong catalog.</p>}
      {!!selected.candidates.length && <fieldset className="candidate-options">
        <legend>Đối chiếu và chọn lại bản in</legend>
        {selected.candidates.filter(candidate => candidate.card).map(candidate => <button type="button"
          key={candidate.card_id} aria-pressed={best?.card_id === candidate.card_id}
          onClick={() => choose(candidate)}>
          <span><b>{candidate.card!.name}</b><small>{candidate.card!.set_name} · #{candidate.card!.number}</small></span>
          {best?.card_id === candidate.card_id && <Check size={17} />}
        </button>)}
      </fieldset>}
      <div className="manual-identification">
        <label htmlFor="scanner-catalog-search">Tìm bản in khác trong catalog</label>
        <div className="manual-search"><MagnifyingGlass size={18} /><input id="scanner-catalog-search"
          value={query} onChange={event => setQuery(event.target.value)} placeholder="Tên hoặc số thẻ…" /></div>
        {searching && <p role="status">Đang tìm thẻ…</p>}
        {error && <p role="alert">{error}</p>}
        {!searching && query.trim().length >= 2 && !matches.length && <p>Chưa có trong catalog. Bạn có thể lưu văn bản OCR ở dạng chưa định danh.</p>}
        {matches.length > 0 && <ul className="manual-matches">{matches.map(card => <li key={card.id}>
          <button type="button" onClick={() => choose({ card_id: card.id, card, similarity: 0 })}>
            <b>{card.name}</b><small>{card.set_name} · #{card.number}</small>
          </button>
        </li>)}</ul>}
      </div>
      {best?.card && <>
        <label>Biến thể bản in<select aria-label="Biến thể bản in" value={selectedVariant?.id ?? ''} onChange={event => {
          const id = event.currentTarget.value;
          setVariantChoice(id ? { trackKey, cardId: best.card_id, id } : null);
        }}>
          <option value="">Chọn biến thể / chưa xác minh</option>
          {best.card.variants.map(value => <option key={value.id} value={value.id}>{value.label}</option>)}
        </select></label>
        <div className="recognition-price"><small>Giá tham khảo · tình trạng NM giả định</small>
          <strong>{!printingReady ? 'Cần xác nhận bản in' : best.card.variants.length === 0 || selectedVariant ? money(quote?.amount_vnd) : 'Chọn biến thể để xem giá'}</strong>
          {quote?.source && <small>{quote.source} · {quote.source_at ? new Date(quote.source_at).toLocaleDateString('vi-VN') : 'Chưa có ngày'}
            {quote.price_stale || quote.fx_stale ? ' · Giá/tỷ giá bản lưu' : ''}</small>}
        </div>
        <button className="button primary full-width" disabled={!printingReady} onClick={() => onSave({
          card_id: best.card_id, card: best.card, similarity: best.similarity, evidence: best.evidence,
          variant: selectedVariant?.id ?? null, variantLabel: selectedVariant?.label || 'Chưa xác minh',
          variantSource: selectedVariant ? 'user_selected' : undefined,
        })}><Plus size={19} />Xác nhận và lưu vào phiên</button>
        <Link className="button secondary full-width" to={'/cards/' + encodeURIComponent(best.card_id)}>Xem & đối chiếu bản in<ArrowRight size={18} /></Link>
        <Link className="text-button scanner-sell" to={'/sell?card=' + encodeURIComponent(best.card_id)}>Đăng bán lá thẻ này<ArrowRight size={15} /></Link>
      </>}
      {!best && (selected.ocr?.name || selected.ocr?.number) && <button className="button secondary full-width"
        onClick={() => onSave({ card_id: '', similarity: 0, ocrName: selected.ocr?.name, ocrNumber: selected.ocr?.number })}>
        <Plus size={18} />Lưu OCR chưa định danh
      </button>}
    </> : <div className="recognition-empty"><Scan size={58} weight="light" />
      <h3>Lá thẻ nào đang chờ được khám phá?</h3>
      <p>Khi nguồn quét tìm thấy thẻ, tên, bộ thẻ và các bằng chứng nhận diện sẽ xuất hiện ở đây.</p>
      <span><i />Đang chờ khung hình</span>
    </div>}
    {tracks.length > 1 && <div className="track-selector" aria-label="Chọn thẻ trong khung">{tracks.map(track =>
      <button key={track.track_id} className={(track.track_id === selected?.track_id ? 'selected ' : '') + (track.status === 'needs_confirmation' ? 'recognized' : '')}
        onClick={() => onSelect(track.track_id)}>#{track.track_id} {track.candidates[0]?.card?.name || 'Đang đọc'}</button>)}</div>}
  </aside>;
}
