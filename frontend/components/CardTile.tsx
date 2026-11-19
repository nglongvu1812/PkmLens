import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowUpRight, Heart, ImageBroken, Storefront } from '@phosphor-icons/react';
import { conditionName, mediaUrl, money, useStore } from '../api';
import type { Card, Listing } from '../types';

export function CardArt({ src, name, eager = false }: { src: string; name: string; eager?: boolean }) {
  const [broken, setBroken] = useState(false);
  return broken ? <div className="missing-art"><ImageBroken size={32} /><span>Chưa tải được ảnh</span></div> : <img src={mediaUrl(src)} alt={name} loading={eager ? 'eager' : 'lazy'} onError={() => setBroken(true)} />;
}

export default function CardTile({ card, sale }: { card: Card; sale?: Listing }) {
  const { saved, toggleSaved } = useStore();
  const isSaved = saved.includes(card.id);
  const href = sale ? '/listings/' + sale.id : '/cards/' + encodeURIComponent(card.id);
  const amount = sale?.asking_vnd ?? card.reference_price?.amount_vnd;
  return <article className="card-tile">
    <div className="tile-art"><Link to={href} tabIndex={-1} aria-hidden="true"><CardArt src={sale?.image || card.image} name={card.name} /></Link><button className={'save-button' + (isSaved ? ' is-saved' : '')} aria-label={`${isSaved ? 'Bỏ lưu' : 'Lưu'} ${card.name}`} aria-pressed={isSaved} onClick={() => toggleSaved(card.id)}><Heart size={21} weight={isSaved ? 'fill' : 'regular'} /></button>{card.rarity && /illustration|ultra|hyper/i.test(card.rarity) && <span className="art-rarity">{card.rarity.replace('Special ', '')}</span>}</div>
    <div className="tile-info"><p className="tile-set">{card.set_name}</p><h2><Link to={href}>{card.name}<ArrowUpRight size={17} /></Link></h2><div className="tile-meta"><span>#{card.number}{card.total ? `/${card.total}` : ''}</span><span className="tag">{sale ? conditionName[sale.condition] : card.rarity || 'Thẻ đơn'}</span></div>
      <div className="tile-price"><div><small>{sale ? 'Giá người bán' : 'Giá tham khảo'}</small><strong className={amount == null ? 'unavailable-price' : ''}>{money(amount)}</strong></div>{!sale && card.reference_price?.price_stale && <span className="price-age">Bản lưu</span>}</div>
      <div className="tile-footer">{sale ? <span><Storefront size={14} />{sale.seller}</span> : <span>{card.source}</span>}<span className="language-label">{card.language === 'en' ? 'EN' : card.language.toUpperCase()}</span></div>
    </div>
  </article>;
}
