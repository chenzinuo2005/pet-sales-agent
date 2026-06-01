<script setup>
defineProps({
  url: { type: String, required: true },
})

const emit = defineEmits(['remove'])
</script>

<template>
  <div class="image-preview">
    <div class="preview-wrapper">
      <img :src="url" alt="预览图片" class="preview-img" />
      <button class="remove-btn" @click="emit('remove')" title="移除图片">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
          <line x1="18" y1="6" x2="6" y2="18"/>
          <line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.image-preview {
  display: flex;
  justify-content: center;
  margin-bottom: var(--space-sm);
}

.preview-wrapper {
  position: relative;
  display: inline-block;
}

.preview-img {
  max-height: 108px;
  max-width: 200px;
  border-radius: var(--rounded-md);
  border: 1px solid var(--color-hairline);
  object-fit: cover;
  display: block;
}

.remove-btn {
  position: absolute;
  top: -6px;
  right: -6px;
  width: 22px;
  height: 22px;
  border-radius: var(--rounded-full);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-muted);
  box-shadow: var(--shadow-float);
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}

.remove-btn:hover {
  background: var(--color-ink);
  border-color: var(--color-ink);
  color: var(--color-on-primary);
}
</style>
