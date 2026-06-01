<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  disabled: { type: Boolean, default: false },
  hasImage:  { type: Boolean, default: false },
})

const emit = defineEmits(['send', 'image-selected'])

const text = ref('')
const textareaRef = ref(null)

function adjustHeight() {
  const el = textareaRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 72) + 'px'
}

watch(text, adjustHeight)

function handleSend() {
  const trimmed = text.value.trim()
  if ((!trimmed && !props.hasImage) || props.disabled) return
  emit('send', trimmed)
  text.value = ''
  if (textareaRef.value) {
    textareaRef.value.style.height = 'auto'
  }
}

function handleKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    if ((text.value.trim() || props.hasImage) && !props.disabled) {
      handleSend()
    }
  }
}

function handleFileChange(e) {
  const file = e.target.files?.[0]
  if (file) emit('image-selected', file)
  e.target.value = ''
}

function onDrop(e) {
  const file = e.dataTransfer?.files?.[0]
  if (file && file.type.startsWith('image/')) {
    emit('image-selected', file)
  }
}

function onDragOver(e) { e.preventDefault() }
</script>

<template>
  <div class="chat-input" @drop.prevent="onDrop" @dragover="onDragOver">
    <div class="input-wrapper" :class="{ 'input-wrapper--focused': false }">
      <!-- Photo upload -->
      <label class="attach-btn" for="pet-image-upload" title="上传宠物图片" aria-label="上传宠物图片">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
          <rect x="3" y="3" width="18" height="18" rx="3"/>
          <circle cx="8.5" cy="8.5" r="1.5" fill="currentColor" stroke="none"/>
          <path d="M21 15l-5-5L5 21"/>
        </svg>
        <input
          id="pet-image-upload"
          name="pet-image-upload"
          type="file"
          accept="image/jpeg,image/png"
          class="file-input"
          @change="handleFileChange"
        />
      </label>

      <!-- Text area -->
      <textarea
        id="chat-message-input"
        name="message"
        ref="textareaRef"
        v-model="text"
        class="input-field"
        :disabled="disabled"
        placeholder="输入消息，或拖拽上传宠物图片..."
        rows="1"
        autocomplete="off"
        @keydown="handleKeydown"
      ></textarea>

      <!-- Send orb — Airbnb search-orb inspired -->
      <button
        class="send-orb"
        :disabled="disabled || (!text.trim() && !hasImage)"
        @click="handleSend"
        aria-label="发送消息"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"/>
          <polygon points="22 2 15 22 11 13 2 9 22 2"/>
        </svg>
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-input {
  max-width: 768px;
  margin: 0 auto;
  width: 100%;
}

/* Pill wrapper — Airbnb search-bar-pill inspired */
.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: var(--space-sm);
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-radius: var(--rounded-full);
  padding: 6px 6px 6px 16px;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
  box-shadow: rgba(0,0,0,0.02) 0 0 0 1px, rgba(0,0,0,0.04) 0 2px 6px;
}

.input-wrapper:focus-within {
  border-color: var(--color-ink);
  box-shadow: rgba(0,0,0,0.02) 0 0 0 1px,
              rgba(0,0,0,0.04) 0 2px 6px,
              rgba(0,0,0,0.1) 0 4px 8px;
}

/* Photo button */
.attach-btn {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--rounded-full);
  color: var(--color-muted);
  cursor: pointer;
  transition: color 0.15s ease, background 0.15s ease;
}

.attach-btn:hover {
  color: var(--color-primary);
  background: var(--color-surface-soft);
}

.file-input { display: none; }

/* Text field */
.input-field {
  flex: 1;
  font-size: 15px;
  font-weight: 400;
  line-height: 1.5;
  padding: 7px 4px;
  color: var(--color-ink);
  min-height: 24px;
  max-height: 72px;
}

.input-field::placeholder {
  color: var(--color-muted);
  opacity: 0.7;
  font-weight: 400;
}

.input-field:disabled {
  opacity: 0.5;
}

/* Send orb — the single Rausch moment */
.send-orb {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--rounded-full);
  background: var(--color-primary);
  color: var(--color-on-primary);
  transition: background 0.15s ease, opacity 0.15s ease, transform 0.15s ease;
}

.send-orb:hover:not(:disabled) {
  background: var(--color-primary-active);
}

.send-orb:active:not(:disabled) {
  transform: scale(0.93);
}

.send-orb:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}
</style>
