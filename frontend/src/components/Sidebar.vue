<script setup>
defineEmits(['new-session', 'clear-history'])
</script>

<template>
  <div class="sidebar-actions">
    <button class="sidebar-btn sidebar-btn--new" @click="$emit('new-session')" aria-label="开始新对话">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <line x1="12" y1="5" x2="12" y2="19"/>
        <line x1="5" y1="12" x2="19" y2="12"/>
      </svg>
      <span>新对话</span>
    </button>
    <button class="sidebar-btn sidebar-btn--clear" @click="$emit('clear-history')" aria-label="清除聊天记录">
      <span>清除记录</span>
    </button>
  </div>
</template>

<style scoped>
.sidebar-actions {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
}

.sidebar-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  font-weight: 500;
  line-height: 1.29;
  padding: 8px 16px;
  border-radius: var(--rounded-sm);
  white-space: nowrap;
  transition: background 0.15s ease, color 0.15s ease;
}

/* Primary — Airbnb button-primary adapted */
.sidebar-btn--new {
  background: var(--color-primary);
  color: var(--color-on-primary);
  height: 36px;
}

.sidebar-btn--new:hover {
  background: var(--color-primary-active);
}

/* Secondary — text-only, underlined on hover */
.sidebar-btn--clear {
  color: var(--color-muted);
  font-size: 13px;
  padding: 6px 10px;
}

.sidebar-btn--clear:hover {
  color: var(--color-ink);
  text-decoration: underline;
}
</style>
