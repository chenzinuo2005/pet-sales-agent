<script setup>
defineProps({
  url: {
    type: String,
    required: true,
  },
})

const emit = defineEmits(['remove'])
</script>

<template>
  <div class="image-preview">
    <div class="preview-wrapper">
      <img :src="url" alt="预览图片" class="preview-img" />
      <button class="remove-btn" @click="emit('remove')" title="移除图片">
        <span class="remove-icon">&#10005;</span>
      </button>
    </div>
  </div>
</template>

<style scoped>
.image-preview {
  display: flex;
  justify-content: center;
  margin-bottom: 8px;
}

.preview-wrapper {
  position: relative;
  display: inline-block;
}

.preview-img {
  max-height: 120px;
  max-width: 200px;
  border-radius: var(--radius-card);
  border: 2px solid var(--color-border);
  object-fit: cover;
  display: block;
}

.remove-btn {
  position: absolute;
  top: -8px;
  right: -8px;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-soft);
  transition: all 0.2s ease;
}

.remove-btn:hover {
  background: var(--color-error);
  border-color: var(--color-error);
  color: #fff;
}

.remove-icon {
  font-size: 12px;
  line-height: 1;
  color: var(--color-text-muted);
}

.remove-btn:hover .remove-icon {
  color: #fff;
}
</style>
