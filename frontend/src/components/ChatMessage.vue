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
    :class="[
      `message--${role}`,
      { 'message--last': isLast && isStreaming },
    ]"
  >
    <div v-if="role === 'assistant'" class="message-avatar" aria-hidden="true">🐾</div>

    <div class="message-bubble">
      <template v-if="isStreaming">
        <StreamingToken :content="content" />
      </template>
      <template v-else>
        <div class="message-text">{{ content }}</div>
      </template>
    </div>

    <div v-if="role === 'user'" class="message-avatar message-avatar--user" aria-hidden="true">
      👤
    </div>
  </div>
</template>

<style scoped>
.chat-message {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  animation: fadeInUp 0.35s ease;
}

.message--user {
  flex-direction: row-reverse;
}

.message-avatar {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  font-size: 18px;
  background: var(--color-bg);
  border: 1px solid var(--color-border);
}

.message-avatar--user {
  background: var(--color-secondary);
  border-color: var(--color-secondary);
}

.message-bubble {
  max-width: 70%;
  padding: 10px 16px;
  border-radius: var(--radius-card);
  font-size: 15px;
  line-height: 1.7;
}

.message--assistant .message-bubble {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-top-left-radius: 4px;
  color: var(--color-text);
}

.message--user .message-bubble {
  background: #FFF0E0;
  border-top-right-radius: 4px;
  color: var(--color-text);
}

.message-text {
  white-space: pre-wrap;
  word-break: break-word;
}

@media (max-width: 600px) {
  .message-bubble {
    max-width: 85%;
    font-size: 14px;
  }
}
</style>
