<script setup>
import { ref, watch } from 'vue'

const props = defineProps({
  disabled: {
    type: Boolean,
    default: false,
  },
  hasImage: {
    type: Boolean,
    default: false,
  },
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
  if (file) {
    emit('image-selected', file)
  }
  e.target.value = ''
}

function onDrop(e) {
  const file = e.dataTransfer?.files?.[0]
  if (file && file.type.startsWith('image/')) {
    emit('image-selected', file)
  }
}

function onDragOver(e) {
  e.preventDefault()
}
</script>

<template>
  <div class="chat-input" @drop.prevent="onDrop" @dragover="onDragOver">
    <div class="input-wrapper">
      <label class="image-btn" title="上传宠物图片" aria-label="上传宠物图片">
        📷
        <input
          type="file"
          accept="image/jpeg,image/png"
          class="file-input"
          @change="handleFileChange"
        />
      </label>
      <textarea
        ref="textareaRef"
        v-model="text"
        class="input-field"
        :disabled="disabled"
        placeholder="输入消息，或拖拽/上传宠物图片..."
        rows="1"
        @keydown="handleKeydown"
      ></textarea>
      <button
        class="send-btn"
        :disabled="disabled || (!text.trim() && !hasImage)"
        @click="handleSend"
      >
        🐾
      </button>
    </div>
  </div>
</template>

<style scoped>
.chat-input {
  max-width: 800px;
  margin: 0 auto;
  width: 100%;
}

.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 8px;
  background: var(--color-bg);
  border: 1.5px solid var(--color-border);
  border-radius: var(--radius-input);
  padding: 6px 8px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.input-wrapper:focus-within {
  border-color: var(--color-accent);
  box-shadow: 0 0 0 3px rgba(244, 164, 96, 0.1);
}

.image-btn {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  border-radius: 50%;
  transition: background 0.2s ease;
  color: var(--color-text-muted);
  cursor: pointer;
}

.image-btn:hover {
  background: var(--color-border);
  color: var(--color-accent);
}

.file-input {
  display: none;
}

.input-field {
  flex: 1;
  font-size: 15px;
  line-height: 1.5;
  padding: 4px 4px;
  color: var(--color-text);
  min-height: 24px;
  max-height: 72px;
}

.input-field::placeholder {
  color: var(--color-text-muted);
  opacity: 0.6;
}

.input-field:disabled {
  opacity: 0.5;
}

.send-btn {
  flex-shrink: 0;
  width: 38px;
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  border-radius: 50%;
  background: var(--color-primary);
  transition: background 0.2s ease, opacity 0.2s ease;
}

.send-btn:hover:not(:disabled) {
  background: #7A4E2E;
}

.send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
</style>
