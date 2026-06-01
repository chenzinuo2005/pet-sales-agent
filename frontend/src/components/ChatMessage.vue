<script setup>
import StreamingToken from './StreamingToken.vue'

defineProps({
  role: {
    type: String,
    required: true,
    validator: (v) => ['user', 'assistant'].includes(v),
  },
  content: {
    type: String,
    default: '',
  },
  isStreaming: {
    type: Boolean,
    default: false,
  },
  isLast: {
    type: Boolean,
    default: false,
  },
})
</script>

<template>
  <div
    class="chat-message"
    :class="`message--${role}`"
  >
    <div v-if="role === 'assistant'" class="message-avatar" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <circle cx="9" cy="10" r="2.5" fill="var(--color-primary)" opacity="0.5"/>
        <circle cx="15" cy="10" r="2.5" fill="var(--color-primary)" opacity="0.5"/>
        <ellipse cx="12" cy="16" rx="4.5" ry="3.5" fill="var(--color-primary)" opacity="0.3"/>
        <path d="M5 18c1.5 3 4 5.5 7 5.5s5.5-2.5 7-5.5" stroke="var(--color-primary)" stroke-width="1.2" stroke-linecap="round" opacity="0.4"/>
      </svg>
    </div>

    <div class="message-bubble">
      <template v-if="isStreaming">
        <StreamingToken :content="content" />
      </template>
      <template v-else>
        <div class="message-text">{{ content }}</div>
      </template>
    </div>

    <div v-if="role === 'user'" class="message-avatar message-avatar--user" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="9" r="4" fill="var(--color-ink)" opacity="0.3"/>
        <ellipse cx="12" cy="20" rx="7" ry="5" fill="var(--color-ink)" opacity="0.2"/>
      </svg>
    </div>
  </div>
</template>

<style scoped>
.chat-message {
  display: flex;
  align-items: flex-start;
  gap: var(--space-sm);
  animation: messageIn 0.3s ease;
}

.message--user {
  flex-direction: row-reverse;
}

/* Avatar */
.message-avatar {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--rounded-full);
  background: var(--color-surface-soft);
}

.message-avatar--user {
  background: var(--color-surface-strong);
}

/* Bubble */
.message-bubble {
  max-width: 68%;
  padding: 12px 16px;
  border-radius: var(--rounded-md);
  font-size: 15px;
  font-weight: 400;
  line-height: 1.6;
}

.message--assistant .message-bubble {
  background: var(--color-surface-soft);
  border-top-left-radius: var(--rounded-xs);
  color: var(--color-ink);
}

.message--user .message-bubble {
  background: var(--color-canvas);
  border: 1px solid var(--color-hairline);
  border-top-right-radius: var(--rounded-xs);
  color: var(--color-ink);
}

.message-text {
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 744px) {
  .message-bubble {
    max-width: 82%;
    font-size: 14px;
  }
}
</style>
