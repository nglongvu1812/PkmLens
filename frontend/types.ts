export type Price = {
  amount_vnd: number | null;
  variant: string;
  condition: string;
  reason: string | null;
  source: string | null;
  source_at: string | null;
  price_stale?: boolean;
  fx_stale?: boolean;
};

export type Card = {
  id: string;
  name: string;
  number: string;
  total?: number;
  set_id: string;
  set_name: string;
  language: string;
  rarity: string;
  image: string;
  source: string;
  variants: { id: string; label: string }[];
  reference_price: Price | null;
};

export type Listing = {
  id: string;
  card: Card;
  variant: string;
  variant_label: string;
  condition: string;
  asking_vnd: number | null;
  status: string;
  seller: string;
  image: string;
  is_owner: boolean;
};

export type Member = { registered: boolean; display_name: string | null; role: string };
export type SetInfo = { id: string; name: string; count: number; language: string };
export type Candidate = { card_id: string; similarity: number; card?: Card; evidence?: Record<string, unknown> };
export type SessionCard = Candidate & {
  ocrName?: string; ocrNumber?: string; source?: 'camera' | 'image' | 'video'; stillImageRepeated?: boolean;
  variant?: string | null; variantLabel?: string; variantSource?: 'user_selected';
  trackKey?: string; autoAdded?: boolean; priceVnd?: number | null; priceSource?: string | null;
};
