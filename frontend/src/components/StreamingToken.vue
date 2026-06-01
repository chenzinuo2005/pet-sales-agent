<script setup>
defineProps({
  content: { type: String, default: '' },
})
</script>

<template>
  <span class="streaming-token">
    <span class="token-text">{{ content }}</span>
    <span class="token-cursor" aria-hidden="true">|</span>
  </span>
</template>

<style scoped>
.streaming-token {
  display: inline;
}

.token-text {
  white-space: pre-wrap;
  word-break: break-word;
}

.token-cursor {
  display: inline;
  color: var(--color-primary);
  font-weight: 700;
  animation: blink 0.7s infinite;
}
</style>
