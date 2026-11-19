const nativeNames: Record<string, string> = {
  en: 'EN \u00b7 English',
  ja: 'JA \u00b7 \u65e5\u672c\u8a9e',
  'zh-Hans': 'zh-Hans \u00b7 \u7b80\u4f53\u4e2d\u6587',
};

export function languageLabel(language: string): string {
  return nativeNames[language] || language;
}
