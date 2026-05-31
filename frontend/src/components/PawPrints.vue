<script setup>
// Pure decorative component — no logic needed
</script>

<template>
  <div class="paw-prints" aria-hidden="true">
    <span class="paw paw--1"></span>
    <span class="paw paw--2"></span>
    <span class="paw paw--3"></span>
    <span class="paw paw--4"></span>
    <span class="paw paw--5"></span>
    <span class="paw paw--6"></span>
    <span class="paw paw--7"></span>
    <span class="paw paw--8"></span>
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
  opacity: 0.04;
  background: var(--color-primary);
  animation: float 8s ease-in-out infinite;
}

.paw::before,
.paw::after {
  content: '';
  position: absolute;
  background: var(--color-primary);
  border-radius: 50%;
}

/* main pad */
.paw::before {
  width: 20px;
  height: 22px;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
}

/* toe pads */
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

.paw--1 { top: 10%; left: 10%; animation-delay: 0s; }
.paw--2 { top: 25%; left: 85%; animation-delay: 1s; transform: rotate(30deg); }
.paw--3 { top: 50%; left: 5%; animation-delay: 2s; transform: rotate(-20deg); }
.paw--4 { top: 70%; left: 90%; animation-delay: 3s; transform: rotate(45deg); }
.paw--5 { top: 15%; left: 50%; animation-delay: 4s; transform: rotate(-35deg); }
.paw--6 { top: 60%; left: 40%; animation-delay: 5s; transform: rotate(15deg); }
.paw--7 { top: 85%; left: 20%; animation-delay: 6s; transform: rotate(-10deg); }
.paw--8 { top: 40%; left: 70%; animation-delay: 7s; transform: rotate(25deg); }

@media (max-width: 768px) {
  .paw-prints {
    display: none;
  }
}
</style>
