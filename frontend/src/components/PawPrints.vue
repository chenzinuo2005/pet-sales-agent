<script setup>
// Decorative background — subtle paw prints floating across the canvas
</script>

<template>
  <div class="paw-prints" aria-hidden="true">
    <span class="paw paw--1" />
    <span class="paw paw--2" />
    <span class="paw paw--3" />
    <span class="paw paw--4" />
    <span class="paw paw--5" />
    <span class="paw paw--6" />
    <span class="paw paw--7" />
    <span class="paw paw--8" />
  </div>
</template>

<style scoped>
.paw-prints {
  position: absolute;
  inset: 0;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}

.paw {
  position: absolute;
  width: 40px;
  height: 40px;
  opacity: 0.03;
  background: var(--color-primary);
  animation: float 10s ease-in-out infinite;
}

.paw::before,
.paw::after {
  content: '';
  position: absolute;
  background: var(--color-primary);
  border-radius: 50%;
}

/* Main pad */
.paw::before {
  width: 20px;
  height: 22px;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
}

/* Toe pads */
.paw::after {
  width: 35px;
  height: 18px;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  background: none;
  box-shadow:
    5px 2px 0 1px var(--color-primary),
    15px 0px 0 1px var(--color-primary),
    25px 2px 0 1px var(--color-primary);
  border-radius: 0;
}

.paw--1 { top: 8%;  left: 8%;  animation-delay: 0s; }
.paw--2 { top: 22%; left: 88%; animation-delay: 1.2s; transform: rotate(30deg); }
.paw--3 { top: 48%; left: 4%;  animation-delay: 2.4s; transform: rotate(-20deg); }
.paw--4 { top: 72%; left: 92%; animation-delay: 3.6s; transform: rotate(45deg); }
.paw--5 { top: 12%; left: 52%; animation-delay: 4.8s; transform: rotate(-35deg); }
.paw--6 { top: 58%; left: 38%; animation-delay: 6s;   transform: rotate(15deg); }
.paw--7 { top: 88%; left: 18%; animation-delay: 7.2s; transform: rotate(-10deg); }
.paw--8 { top: 36%; left: 72%; animation-delay: 8.4s; transform: rotate(25deg); }

@media (max-width: 744px) {
  .paw-prints { display: none; }
}
</style>
