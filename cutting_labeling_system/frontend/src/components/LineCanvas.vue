<template>
  <canvas 
    ref="cv" 
    style="
      border:1px solid #ccc;
      cursor: crosshair;
      background:#fff;
      display:block;
    "
  ></canvas>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted } from 'vue'

const props = defineProps({
  imageUrl: String,
  lines: Array,
  selectedIndexes: { type: Array, default: () => [] },
  readonly: { type: Boolean, default: false },
  lineColor: { type: String, default: '#ff0000' },
  zoom: { type: Number, default: 1.0 }
})

const emit = defineEmits(['update:lines', 'update:selectedIndexes'])
const cv = ref(null)
let img = new Image()
img.crossOrigin = 'anonymous'

const isSelecting = ref(false)
const startX = ref(0)
const endX = ref(0)

function draw() {
  const canvas = cv.value
  if (!canvas || !img.complete || !img.width) return
  const ctx = canvas.getContext('2d')

  const zoom = props.zoom || 1.0
  canvas.width = img.width * zoom
  canvas.height = img.height * zoom

  ctx.clearRect(0,0,canvas.width,canvas.height)
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

  if(props.lines){
    props.lines.forEach((line, idx)=>{
      const isSelected = props.selectedIndexes?.includes(idx)
      if (isSelected) {
        ctx.strokeStyle = "#FFEC00"
        ctx.lineWidth = 2
      } else {
        ctx.strokeStyle = line.color || '#ff0000'
        ctx.lineWidth = 1
      }
      
      const pos = line.pos * zoom
      ctx.beginPath()
      ctx.moveTo(pos, 0)
      ctx.lineTo(pos, canvas.height)
      ctx.stroke()
    })
  }

  if (isSelecting.value) {
    const minX = Math.min(startX.value, endX.value) * zoom
    const maxX = Math.max(startX.value, endX.value) * zoom
    ctx.fillStyle = 'rgba(24, 144, 255, 0.2)'
    ctx.fillRect(minX, 0, maxX - minX, canvas.height)
    ctx.strokeStyle = '#1890ff'
    ctx.strokeRect(minX, 0, maxX - minX, canvas.height)
  }
}

function onMouseDown(e) {
  if (e.ctrlKey && e.button === 0) {
    isSelecting.value = true
    const rect = cv.value.getBoundingClientRect()
    const scale = img.width / rect.width
    startX.value = (e.clientX - rect.left) * scale
    endX.value = startX.value
    draw()
  }
}

function onMouseMove(e) {
  if (!isSelecting.value) return
  const rect = cv.value.getBoundingClientRect()
  const scale = img.width / rect.width
  endX.value = (e.clientX - rect.left) * scale
  draw()
}

function onMouseUp() {
  if (!isSelecting.value) return
  isSelecting.value = false

  const minX = Math.min(startX.value, endX.value)
  const maxX = Math.max(startX.value, endX.value)

  const selected = []
  props.lines.forEach((line, idx) => {
    if (line.pos >= minX && line.pos <= maxX) {
      selected.push(idx)
    }
  })

  emit('update:selectedIndexes', selected)
  draw()
}

function onCanvasClick(e) {
  if (isSelecting.value) return
  if (e.ctrlKey) return

  const rect = cv.value.getBoundingClientRect()
  const scale = img.width / rect.width
  const x = Math.round((e.clientX - rect.left) * scale)
  
  const idx = props.lines.findIndex(line => Math.abs(line.pos - x) < 8)
  
  if (idx !== -1) {
    emit('update:selectedIndexes', [idx])
  } else if (!props.readonly && props.lineColor) {
    const newLines = [...props.lines, { pos: x, color: props.lineColor }]
    newLines.sort((a, b) => a.pos - b.pos)
    emit('update:lines', newLines)
    emit('update:selectedIndexes', [newLines.length - 1])
    draw()
  }
}

function onKeyDown(e) {
  if (props.readonly) return
  
  if (props.selectedIndexes?.length === 0) return

  if (e.key === 'Delete') {
    const newLines = [...props.lines]
    props.selectedIndexes.sort((a, b) => b - a).forEach(i => newLines.splice(i, 1))
    emit('update:lines', newLines)
    emit('update:selectedIndexes', [])
    draw()
    return
  }

  if (e.key === 'ArrowLeft') {
    const newLines = JSON.parse(JSON.stringify(props.lines))
    props.selectedIndexes.forEach(index => {
      if (newLines[index].pos > 0) {
        newLines[index].pos -= 1
      }
    })
    emit('update:lines', newLines)
    draw()
  }
  if (e.key === 'ArrowRight') {
    const newLines = JSON.parse(JSON.stringify(props.lines))
    props.selectedIndexes.forEach(index => {
      if (newLines[index].pos < (img.width || Infinity)) {
        newLines[index].pos += 1
      }
    })
    emit('update:lines', newLines)
    draw()
  }
}

watch(() => props.imageUrl, (url) => { img.src = url; img.onload = draw }, { immediate: true })
watch(() => props.lines, draw, { deep: true })
watch(() => props.selectedIndexes, draw, { deep: true })
watch(() => props.zoom, draw)

onMounted(() => {
  const canvas = cv.value
  canvas.addEventListener('mousedown', onMouseDown)
  canvas.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', onMouseUp)
  canvas.addEventListener('click', onCanvasClick)
  window.addEventListener('keydown', onKeyDown)
  draw()
})

onUnmounted(() => {
  const canvas = cv.value
  if (canvas) {
    canvas.removeEventListener('mousedown', onMouseDown)
    canvas.removeEventListener('mousemove', onMouseMove)
    canvas.removeEventListener('click', onCanvasClick)
  }
  window.removeEventListener('mouseup', onMouseUp)
  window.removeEventListener('keydown', onKeyDown)
})
</script>