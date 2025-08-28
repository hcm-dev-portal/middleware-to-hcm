class TranslationManager {
  constructor() {
    this.currentLang = 'en-US';
    this.translations = {};
    this.elements = new Map();
    this.initialized = false;
  }

  async init(defaultLang = 'en-US') {
    try {
      // Load both language files
      const [enResponse, zhResponse] = await Promise.all([
        fetch('./lang/en-US.json'),
        fetch('./lang/zh-TW.json')
      ]);

      this.translations['en-US'] = await enResponse.json();
      this.translations['zh-TW'] = await zhResponse.json();

      // Get saved language preference or use default
      const savedLang = localStorage.getItem('app-language') || defaultLang;
      this.currentLang = this.isValidLang(savedLang) ? savedLang : defaultLang;
      
      this.initialized = true;
      this.scanAndRegisterElements();
      this.applyTranslations();
      
      console.log(`[i18n] Initialized with language: ${this.currentLang}`);
    } catch (error) {
      console.error('[i18n] Failed to load translation files:', error);
    }
  }

  isValidLang(lang) {
    return ['en-US', 'zh-TW'].includes(lang);
  }

  scanAndRegisterElements() {
    // Clear existing registrations
    this.elements.clear();

    // Find all elements with data-i18n attributes
    const elementsWithI18n = document.querySelectorAll('[data-i18n]');
    elementsWithI18n.forEach(el => {
      const key = el.getAttribute('data-i18n');
      const type = el.getAttribute('data-i18n-type') || 'text';
      
      if (!this.elements.has(key)) {
        this.elements.set(key, []);
      }
      
      this.elements.get(key).push({ element: el, type });
    });
  }

  registerElement(key, element, type = 'text') {
    if (!this.elements.has(key)) {
      this.elements.set(key, []);
    }
    
    this.elements.get(key).push({ element, type });
    
    // Apply translation immediately if initialized
    if (this.initialized) {
      this.translateElement(key, element, type);
    }
  }

  translateElement(key, element, type) {
    const translation = this.getTranslation(key);
    if (!translation) return;

    switch (type) {
      case 'text':
        element.textContent = translation;
        break;
      case 'html':
        element.innerHTML = translation;
        break;
      case 'placeholder':
        element.placeholder = translation;
        break;
      case 'title':
        element.title = translation;
        break;
      case 'aria-label':
        element.setAttribute('aria-label', translation);
        break;
    }
  }

  getTranslation(key, params = {}) {
    const keys = key.split('.');
    let value = this.translations[this.currentLang];
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        console.warn(`[i18n] Translation key not found: ${key} for language: ${this.currentLang}`);
        return key; // Return key as fallback
      }
    }
    
    // Handle parameter substitution
    if (typeof value === 'string' && Object.keys(params).length > 0) {
      return value.replace(/\{(\w+)\}/g, (match, param) => {
        return params[param] !== undefined ? params[param] : match;
      });
    }
    
    return value;
  }

  applyTranslations() {
    if (!this.initialized) return;

    this.elements.forEach((elementInfos, key) => {
      elementInfos.forEach(({ element, type }) => {
        if (element && element.isConnected) {
          this.translateElement(key, element, type);
        }
      });
    });
  }

  switchLanguage(newLang) {
    if (!this.isValidLang(newLang) || newLang === this.currentLang) return;
    
    this.currentLang = newLang;
    localStorage.setItem('app-language', newLang);
    
    // Re-scan for new elements
    this.scanAndRegisterElements();
    this.applyTranslations();
    
    // Dispatch language change event
    window.dispatchEvent(new CustomEvent('languageChanged', { 
      detail: { language: newLang } 
    }));
    
    console.log(`[i18n] Switched to language: ${newLang}`);
  }

  getCurrentLanguage() {
    return this.currentLang;
  }

  t(key, params = {}) {
    return this.getTranslation(key, params);
  }

  // Helper method for dynamic content updates
  updateDynamicContent(containerId, contentData) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Re-scan for any new i18n elements in the updated content
    setTimeout(() => {
      this.scanAndRegisterElements();
      this.applyTranslations();
    }, 0);
  }

  // Format numbers according to locale
  formatNumber(number, options = {}) {
    const locale = this.currentLang === 'zh-TW' ? 'zh-TW' : 'en-US';
    return new Intl.NumberFormat(locale, options).format(number);
  }

  // Format dates according to locale
  formatDate(date, options = {}) {
    const locale = this.currentLang === 'zh-TW' ? 'zh-TW' : 'en-US';
    return new Intl.DateTimeFormat(locale, options).format(date);
  }

  // Format time according to locale
  formatTime(date, options = { hour: '2-digit', minute: '2-digit' }) {
    const locale = this.currentLang === 'zh-TW' ? 'zh-TW' : 'en-US';
    return new Intl.DateTimeFormat(locale, options).format(date);
  }
}

// Export for use in HTML
window.TranslationManager = TranslationManager;