<script setup>
defineEmits(['new-session', 'clear-history'])
</script>

<template>
  <div class="sidebar-actions">
    <button class="sidebar-btn sidebar-btn--primary" @click="$emit('new-session')" aria-label="开始新对话">
      <span class="btn-icon" aria-hidden="true">🐾</span>
      <span>新对话</span>
    </button>
    <button class="sidebar-btn sidebar-btn--subtle" @click="$emit('clear-history')" aria-label="清除聊天记录">
      <span>清除记录</span>
    </button>
  </div>
</template>

<style scoped>
.sidebar-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sidebar-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 14px;
  padding: 6px 14px;
  border-radius: var(--radius-btn);
  white-space: nowrap;
  transition: all 0.2s ease;
}

.sidebar-btn--primary {
  background: var(--color-primary);
  color: #fff;
  font-weight: 500;
}

.sidebar-btn--primary:hover {
  background: #7A4E2E;
}

.sidebar-btn--subtle {
  color: var(--color-text-muted);
  font-size: 13px;
  padding: 6px 10px;
}

.sidebar-btn--subtle:hover {
  color: var(--color-text);
  background: var(--color-bg);
}

.btn-icon {
  font-size: 16px;
}
</style>
