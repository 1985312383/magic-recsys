import DefaultTheme from 'vitepress/theme'
import './custom.css'
import type { Theme } from 'vitepress'
import 'viewerjs/dist/viewer.min.css';
import imageViewer from 'vitepress-plugin-image-viewer';
import { useRoute } from 'vitepress';
import { h } from 'vue'
import ReadingProgress from './components/ReadingProgress.vue' // 阅读进度圈组件(vuepress同款)


export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        // 注册全局组件（可选）
        app.component('imageViewer', imageViewer);
    },
    setup() {
        const route = useRoute();
        // 启用插件
        imageViewer(route);
    },
     // 布局扩展
    Layout: () => {
        return h(DefaultTheme.Layout, null, {
        // 添加阅读进度圈组件
        'layout-bottom': () => h(ReadingProgress),
        })
    },
} satisfies Theme
