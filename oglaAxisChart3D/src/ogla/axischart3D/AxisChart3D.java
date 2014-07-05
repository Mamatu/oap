package ogla.axischart3D;

import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Point;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;
import javax.swing.JPanel;
import ogla.axischart.AxisChart;
import ogla.axischart.Displayer;
import ogla.axischart.lists.BaseList;
import ogla.axischart.lists.ListImageComponentBundle;
import ogla.axischart.lists.ListInstalledImageComponent;
import ogla.axischart.lists.ListInterface;
import ogla.axischart.lists.ListListener;
import ogla.axischart.plugin.image.ImageComponent;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.util.DefaultDisplayingStackImpl;
import ogla.axischart3D.gui.CameraPanel;
import ogla.axischart3D.gui.ChartCanvas;
import ogla.axischart3D.gui.ComponentsPanel;
import ogla.axischart3D.gui.DataPanel;
import ogla.axischart3D.lists.ListAxisChart3DInstalledComponent;
import ogla.axischart3D.lists.ListAxisChart3DPlotManagerBundle;
import ogla.axischart3D.lists.ListDataRepositoryWithPrimitives;
import ogla.axischart3D.plugin.chartcomponent.AxisChart3DComponent;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManager;
import ogla.axischart3D.plugin.plotmanager.AxisChart3DPlotManagerBundle;
import ogla.axischart3D.util.DisplayerImpl;
import ogla.chart.DrawableOnChart;
import ogla.chart.DrawingInfo;
import ogla.chart.gui.InternalFrame;
import ogla.math.Vector3;

public class AxisChart3D extends AxisChart {

    public enum Displaying {

        GRID, ALL
    }

    public void setTypeOfDisplaying(Displaying displaying) {
        if (displaying == Displaying.ALL) {
            this.renderManager.defaultRenderImpl.setDrawingType(DefaultRenderImpl.PAINT_ALL);
        }
        if (displaying == Displaying.GRID) {
            this.renderManager.defaultRenderImpl.setDrawingType(DefaultRenderImpl.PAINT_GRID);
        }
    }

    public String getDescription() {
        return "";
    }

    public String getLabel() {
        return "Axis Chart 3D";
    }

    @Override
    public int getWidth() {
        return this.chartSurface.getWidth();
    }

    @Override
    public int getHeight() {
        return this.chartSurface.getHeight();
    }

    @Override
    public Point getLocation() {
        return this.chartSurface.getLocation();
    }
    private DesktopPanel desktopPanel;
    private ChartCanvas chartCanvas = null;
    private DataPanel dataPanel = null;
    private ComponentsPanel componentsPanel = null;
    private CameraPanel cameraPanel = null;
    private ListAxisChart3DInstalledComponent listAxisChart3DInstalledComponent = new ListAxisChart3DInstalledComponent();
    private ListDataRepositoryWithPrimitives listDataRepositoryWithPrimitives = new ListDataRepositoryWithPrimitives();

    public AxisChart3D(DesktopPanel desktopPanel, List<BaseList> baseLists) {
        super(desktopPanel.getChartSurface(), baseLists);
        this.desktopPanel = desktopPanel;
        listsContainer.add(listAxisChart3DInstalledComponent);
        listsContainer.add(listDataRepositoryWithPrimitives);
        this.chartCanvas = new ChartCanvas(this, listsContainer);
        this.dataPanel = new DataPanel(this, listsContainer);
        this.componentsPanel = new ComponentsPanel(this, listsContainer, defaultDisplayStackImpl);
        this.cameraPanel = new CameraPanel(this, listsContainer, defaultDisplayStackImpl);
        listDisplayer.addListListener(new ListListener<DisplayerImpl>() {

            public void isAdded(DisplayerImpl t) {
                defaultDisplayStackImpl.add(t);
                //AxisChart3D.this.repaintChartSurface();
            }

            public void isRemoved(DisplayerImpl t) {
                defaultDisplayStackImpl.remove(t);
                //AxisChart3D.this.repaintChartSurface();
            }
        });

        listAxisChart3DInstalledComponent.addListListener(new ListListener<AxisChart3DComponent>() {

            public void isAdded(AxisChart3DComponent t) {
                defaultDisplayStackImpl.add(t);
                //  AxisChart3D.this.repaintChartSurface();
            }

            public void isRemoved(AxisChart3DComponent t) {
                defaultDisplayStackImpl.remove(t);
                //  AxisChart3D.this.repaintChartSurface();
            }
        });
        ListImageComponentBundle listImageComponentBundle = listsContainer.get(ListImageComponentBundle.class);
        listImageComponentBundle.addListListener(new ListListener<ImageComponentBundle>() {

            public void isAdded(ImageComponentBundle t) {

                ImageComponent imageComponent = t.newImageComponent();
                if (put(t, t.newImageComponent())) {
                    ListInstalledImageComponent list = listsContainer.get(ListInstalledImageComponent.class);
                    list.add(imageComponent);
                }

            }

            public void isRemoved(ImageComponentBundle t) {
                for (ImageComponent imageComponent : imageBundleImageComponents.get(t)) {
                    ListInstalledImageComponent list = listsContainer.get(ListInstalledImageComponent.class);
                    list.remove(imageComponent);
                }
            }
        });

        listDataRepositoryWithPrimitives.addListListener(new ListListener<DataRepository>() {

            public void isAdded(DataRepository t) {
                dataRepositoryBlocks.put(t, new ArrayList<DataBlock>());
            }

            public void isRemoved(DataRepository t) {
                dataRepositoryBlocks.remove(t);
            }
        });
        initializeGUI();
    }

    private boolean put(ImageComponentBundle bundleImageComponent, ImageComponent imageComponent) {
        if (imageBundleImageComponents.containsKey(bundleImageComponent)) {
            imageBundleImageComponents.get(bundleImageComponent).add(imageComponent);
            return false;
        } else {
            List<ImageComponent> list = new ArrayList<ImageComponent>();
            list.add(imageComponent);
            imageBundleImageComponents.put(bundleImageComponent, list);
            return true;
        }
    }
    private Map<ImageComponentBundle, List<ImageComponent>> imageBundleImageComponents = new HashMap<ImageComponentBundle, List<ImageComponent>>();

    public Projection getProjection() {
        return this.renderManager.projection;
    }

    public void setProjection(Projection projection) {
        this.renderManager.projection = projection;
    }

    public void setAxisChart3DPlotManagerBundle(AxisChart3DPlotManagerBundle axisChart3DPlotManagerBundle) {
        this.renderManager.axisChart3DPlotManager = axisChart3DPlotManagerBundle.newPlotManager();
    }

    public AxisChart3DPlotManager getAxisChart3DPlotManager() {
        return this.renderManager.axisChart3DPlotManager;
    }

    public void shouldBeRendered() {
        this.renderManager.shouldBeRendered();
    }
    private Map<DataBlock, List<Primitive>> dataBlockPrimitives = new HashMap<DataBlock, List<Primitive>>();
    private Map<DataRepository, List<DataBlock>> dataRepositoryBlocks = new HashMap<DataRepository, List<DataBlock>>();
    private Map<DataRepository, AxisChart3DPlotManager> dataRepositoryPlotManager = new HashMap<DataRepository, AxisChart3DPlotManager>();

    public AxisChart3DPlotManager getAxisChart3DPlotManagerOf(DataRepository dataRepository) {
        return dataRepositoryPlotManager.get(dataRepository);
    }

    public void putNewPrimitives(DataRepository dataRepository, AxisChart3DPlotManager axisChart3DPlotManager) {
        dataRepositoryBlocks.remove(dataRepository);
        for (int fa = 0; fa < dataRepository.size(); fa++) {
            DataBlock dataBlock = dataRepository.get(fa);
            dataBlockPrimitives.remove(dataBlock);
        }
        dataRepositoryPlotManager.put(dataRepository, axisChart3DPlotManager);
    }

    public void disattachPrimitives(DataBlock dataBlock) {
        dataBlockPrimitives.remove(dataBlock);
    }

    private class RenderThreadImpl implements Runnable {

        private RenderManager renderManager = null;

        public RenderThreadImpl(RenderManager renderManager) {
            this.renderManager = renderManager;
        }
        private Color tempColor = new Color();

        private void addPrimitivesToRenderManager(Primitive[] primitives) {
            renderManager.add(primitives);
        }

        public void run() {
            chartCanvas.setLabelRendering();
            for (int fa = 0; fa < listDisplayer.size(); fa++) {
                Displayer displayer = listDisplayer.get(fa);
                DisplayerImpl displayerImpl = (DisplayerImpl) displayer;
                tempColor.set(displayerImpl.getColor());
                for (int faa = 0; faa < displayer.getNumberOfDataRepositories(); faa++) {
                    DataRepository dataRepository = displayer.getDataRepository(faa);

                    if (!dataRepositoryBlocks.containsKey(dataRepository)) {
                        listDataRepositoryWithPrimitives.add(dataRepository);
                        AxisChart3DPlotManager axisChart3DPlotManager = this.renderManager.axisChart3DPlotManager;
                        if (dataRepositoryPlotManager.containsKey(dataRepository)) {
                            axisChart3DPlotManager = dataRepositoryPlotManager.get(dataRepository);
                        }
                        for (int fb = 0; fb < dataRepository.size(); fb++) {
                            DataBlock dataBlock = dataRepository.get(fb);
                            if (dataBlock != null && dataBlock.columns() >= 3) {
                                Primitive[] primitives = axisChart3DPlotManager.getPrimitives(dataBlock);
                                addPrimitivesToRenderManager(primitives);
                                List<Primitive> primitivesList = new ArrayList<Primitive>();
                                primitivesList.addAll(Arrays.asList(primitives));
                                dataBlockPrimitives.put(dataBlock, primitivesList);
                                dataRepositoryBlocks.get(dataRepository).add(dataBlock);
                            }
                        }

                    } else {
                        for (int fb = 0; fb < dataRepository.size(); fb++) {
                            DataBlock dataBlock = dataRepository.get(fb);
                            List<Primitive> primitivesList = dataBlockPrimitives.get(dataBlock);
                            if (primitivesList != null) {
                                Primitive[] primitives = new Primitive[primitivesList.size()];
                                primitives = primitivesList.toArray(primitives);
                                addPrimitivesToRenderManager(primitives);
                            }
                        }
                    }
                }
            }
            final int width = AxisChart3D.this.chartSurface.getWidth();
            final int height = AxisChart3D.this.chartSurface.getHeight();
            Image scaledImage = null;
            Image renderedImage = this.renderManager.defaultRenderImpl.paint(this.renderManager.projection,
                    new Color(AxisChart3D.this.getChartSurface().getBackground()), 0, 0);
            if (renderedImage == null) {
                renderManager.finishThread();
                renderManager.scaledImage = renderManager.tempScaledImage;
                renderManager.renderedImage = renderManager.tempRenderedImage;
                return;
            }
            final int xlocation = AxisChart3D.this.getLocation().x;
            final int ylocation = AxisChart3D.this.getLocation().y;

            if (renderedImage != null) {
                scaledImage = renderedImage;
                if (renderedImage.getWidth(null) > width || renderedImage.getHeight(null) > height) {
                    scaledImage = renderedImage.getScaledInstance(width, height, Image.SCALE_SMOOTH);
                    renderManager.drawImage(scaledImage, xlocation, ylocation);
                } else {
                    renderManager.drawImage(scaledImage, width / 2 - scaledImage.getWidth(null) / 2,
                            height / 2 - scaledImage.getHeight(null) / 2);
                }

            }
            renderManager.setRenderedImage(renderedImage);
            renderManager.setScaledImage(scaledImage);
            // renderManager.drawPicture();
            AxisChart3D.this.repaintChartSurface();

            renderManager.finishThread();
        }
    }

    private class RenderManager {

        private Projection projection = null;
        private ListAxisChart3DPlotManagerBundle listProjectionBundle = null;
        private AxisChart3DPlotManager axisChart3DPlotManager = null;
        private ReentrantLock mutex = new ReentrantLock();

        public RenderManager() {
            defaultDisplayStackImpl.add(new DrawableOnChartImpl());
            listProjectionBundle = listsContainer.get(ListAxisChart3DPlotManagerBundle.class);
            listProjectionBundle.addListListener(new ListListener<AxisChart3DPlotManagerBundle>() {

                public void isAdded(AxisChart3DPlotManagerBundle t) {
                    if (axisChart3DPlotManager == null) {
                        axisChart3DPlotManager = t.newPlotManager();
                    }

                }

                public void isRemoved(AxisChart3DPlotManagerBundle t) {
                }
            });

        }

        public void shouldBeRendered() {
            shouldBeRendered = true;
        }
        private Vector3 cameraPosition = new Vector3();
        private Vector3 lookAt = new Vector3();
        private boolean shouldBeRendered = false;

        public void render(Vector3 cameraPosition, Vector3 lookAt) {
            mutex.lock();
            if (defaultRenderImpl == null) {
                defaultRenderImpl = new DefaultRenderImpl(cameraPosition, lookAt);
                defaultRenderImpl.clear();
                defaultRenderImpl.defineCameraPlane(cameraPosition, lookAt);
                this.projection.set(cameraPosition, lookAt);
                this.cameraPosition.set(cameraPosition);
                this.lookAt.set(lookAt);
                shouldBeRendered = true;
                mutex.unlock();
                return;
            }
            //if (!(this.cameraPosition.equals(cameraPosition)) || !(this.lookAt.equals(lookAt))) {
            shouldBeRendered = true;
            defaultRenderImpl.defineCameraPlane(cameraPosition, lookAt);
            this.projection.set(cameraPosition, lookAt);
            this.lookAt.set(lookAt);
            this.cameraPosition.set(cameraPosition);
            //}
            mutex.unlock();
        }
        private DefaultRenderImpl defaultRenderImpl = null;
        private Image renderedImage = null;
        private Image tempRenderedImage = null;
        private Image scaledImage = null;
        private Image tempScaledImage = null;
        private Graphics2D gd = null;

        public void setRenderedImage(Image renderedImage) {
            mutex.lock();
            this.renderedImage = renderedImage;
            mutex.unlock();
        }

        public void setScaledImage(Image scaledImage) {
            mutex.lock();
            this.scaledImage = scaledImage;
            mutex.unlock();
        }

        public void finishThread() {
            thread = null;
            chartCanvas.clearLabel();
        }

        private void setGraphics2D(Graphics2D gd) {
            mutex.lock();
            this.gd = gd;
            mutex.unlock();
        }

        public void drawImage(Image image, int x, int y) {
            mutex.lock();
            if (gd == null) {
                mutex.unlock();
                return;
            }
            gd.drawImage(image, x, y, null);
            mutex.unlock();
        }
        private RenderThreadImpl renderThreadImpl = new RenderThreadImpl(this);
        private Thread thread = null;

        private void add(Primitive[] primitives) {
            if (primitives != null) {
                for (Primitive primitive : primitives) {
                    defaultRenderImpl.add(primitive);
                }
            }
        }

        private void remove(List<Primitive> primitives) {
            if (primitives != null) {
                for (Primitive primitive : primitives) {
                    defaultRenderImpl.remove(primitive);
                }
            }

        }
        private int width = -1;
        private int height = -1;

        private void drawImage() {
            mutex.lock();
            final int xlocation = AxisChart3D.this.getLocation().x;
            final int ylocation = AxisChart3D.this.getLocation().y;
            final int width = AxisChart3D.this.chartSurface.getWidth();
            final int height = AxisChart3D.this.chartSurface.getHeight();

            if (this.width != width || this.height != height) {
                scaledImage = renderedImage;
                if (renderedImage.getWidth(null) > width || renderedImage.getHeight(null) > height) {
                    scaledImage = renderedImage.getScaledInstance(width, height, Image.SCALE_SMOOTH);
                    drawImage(scaledImage, xlocation, ylocation);
                } else {
                    drawImage(scaledImage, width / 2 - scaledImage.getWidth(null) / 2,
                            height / 2 - scaledImage.getHeight(null) / 2);
                }

                this.width = width;
                this.height = height;
            } else {
                if (renderedImage.getWidth(null) > width || renderedImage.getHeight(null) > height) {
                    drawImage(scaledImage, xlocation, ylocation);
                } else {
                    drawImage(scaledImage, width / 2 - scaledImage.getWidth(null) / 2,
                            height / 2 - scaledImage.getHeight(null) / 2);
                }
            }
            mutex.unlock();
        }

        private class DrawableOnChartImpl implements DrawableOnChart {

            public String getLabel() {
                return "";
            }

            public String getDescription() {
                return "";
            }
            private ogla.g3d.Color tempColor = new ogla.g3d.Color();
            private Map<AxisChart3DComponent, List<Primitive>> componentPrimitives = new HashMap<AxisChart3DComponent, List<Primitive>>();

            public void drawOnChart(Graphics2D gd, DrawingInfo drawingInfo) {
                setGraphics2D(gd);
                if (shouldBeRendered) {
                    if (defaultRenderImpl == null) {
                        defaultRenderImpl = new DefaultRenderImpl(AxisChart3D.this.cameraPanel.getCameraPosition(),
                                AxisChart3D.this.cameraPanel.getLookAt());
                        defaultRenderImpl.clear();
                        defaultRenderImpl.defineCameraPlane(AxisChart3D.this.cameraPanel.getCameraPosition(),
                                AxisChart3D.this.cameraPanel.getLookAt());
                        RenderManager.this.projection.set(AxisChart3D.this.cameraPanel.getCameraPosition(),
                                AxisChart3D.this.cameraPanel.getLookAt());
                        RenderManager.this.cameraPosition.set(AxisChart3D.this.cameraPanel.getCameraPosition());
                        RenderManager.this.lookAt.set(AxisChart3D.this.cameraPanel.getLookAt());
                    }
                    if (thread != null) {
                        defaultRenderImpl.stop();
                    }
                    defaultRenderImpl.clear();
                    for (int fa = 0; fa < listAxisChart3DInstalledComponent.size(); fa++) {
                        Primitive[] primitives = listAxisChart3DInstalledComponent.get(fa).getPrimitives();
                        if (componentPrimitives.containsKey(listAxisChart3DInstalledComponent.get(fa))) {
                            List<Primitive> primitivesList = componentPrimitives.get(listAxisChart3DInstalledComponent.get(fa));
                            remove(primitivesList);
                            primitivesList.clear();
                            primitivesList.addAll(Arrays.asList(primitives));
                            add(primitives);

                        } else {
                            List<Primitive> primitivesList = new ArrayList<Primitive>();
                            primitivesList.addAll(Arrays.asList(primitives));
                            componentPrimitives.put(listAxisChart3DInstalledComponent.get(fa), primitivesList);
                            add(primitives);
                        }
                    }

                    gd.clearRect(0, 0, AxisChart3D.this.getWidth(), AxisChart3D.this.getHeight());
                    mutex.lock();
                    shouldBeRendered = false;
                    tempRenderedImage = renderedImage;
                    tempScaledImage = scaledImage;
                    scaledImage = null;
                    renderedImage = null;
                    thread = new Thread(renderThreadImpl);
                    thread.setDaemon(true);
                    thread.start();
                    thread = null;
                    mutex.unlock();

                } else if (shouldBeRendered == false) {
                    if (renderedImage == null) {
                        return;
                    }
                    drawImage();
                }
            }
        }
    }

    public void render(Vector3 cameraPosition, Vector3 lookAt) {
        renderManager.render(cameraPosition, lookAt);

    }

    private InternalFrame[] initializeGUI() {
        InternalFrame[] frames = new InternalFrame[4];

        frames[0] = newInternalFrame(chartCanvas, "Chart");
        frames[1] = newInternalFrame(dataPanel, "Data");
        frames[2] = newInternalFrame(componentsPanel, "Components");
        frames[3] = newInternalFrame(cameraPanel, "Camera");

        desktopPanel.getDesktop().revalidate();
        desktopPanel.getDesktop().repaint();
        return frames;
    }

    private InternalFrame newInternalFrame(JPanel panel, String title) {
        return new InternalFrame(title, desktopPanel, panel);
    }
    private DefaultDisplayingStackImpl defaultDisplayStackImpl = new DefaultDisplayingStackImpl();
    private RenderManager renderManager = new RenderManager();

    @Override
    public DisplayingStack getDisplayStack() {
        return defaultDisplayStackImpl;
    }

    public ListInterface<Displayer> getListDisplayerInfo() {
        return listDisplayer;
    }

    @Override
    public void repaintChartSurface() {
        desktopPanel.repaint();
        super.repaintChartSurface();
    }
}
