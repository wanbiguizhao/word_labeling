import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'
import SegmentLabelPage from '../views/SegmentLabelPage.vue'
import LabelPage from '../views/LabelPage.vue'

const routes = [
  { path: '/', name: 'home', component: HomePage },
  { path: '/segment', name: 'segment', component: SegmentLabelPage },
  { path: '/label/:id', name: 'label', component: LabelPage },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
