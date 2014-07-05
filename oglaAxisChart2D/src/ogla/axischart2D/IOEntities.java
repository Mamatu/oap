package ogla.axischart2D;

import ogla.core.util.Reader;
import ogla.core.util.Writer;
import ogla.core.data.DataBundle;
import ogla.core.data.DataRepository;
import ogla.axischart2D.plugin.chartcomponent.AxisChart2DComponent;
import ogla.axischart.lists.ListDataBundle;
import ogla.axischart.lists.ListDisplayer;
import ogla.axischart.lists.ListListener;
import ogla.axischart.lists.ListsContainer;
import ogla.axischart2D.util.DisplayerTools;
import ogla.axischart2D.lists.ListAxisChart2DInstalledComponent;
import ogla.axischart2D.util.AxisIndices;
import ogla.chart.IOEntity;
import ogla.chart.gui.InternalFrame;
import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class IOEntities {

    private class RepositoryInfo {

        public String label;
        public String symbolicName;
    }
    private AxisChart2D axisChart = null;
    private ListDisplayer listDisplayers = null;
    private ListDataBundle listDataBundle = null;
    private ListAxisChart2DInstalledComponent listInstalledChartComponent = null;

    public IOEntities(AxisChart2D axisChart, ListsContainer listsContainer) {
        this.axisChart = axisChart;
        this.listDisplayers = listsContainer.get(ListDisplayer.class);
        this.listDataBundle = listsContainer.get(ListDataBundle.class);
        this.listInstalledChartComponent = listsContainer.get(ListAxisChart2DInstalledComponent.class);
        this.axisChart.addIOEntityProxy(new WorkspaceEntity());
        this.axisChart.addIOEntityProxy(new PropertiesEntity(this.axisChart.xTicks));
        this.axisChart.addIOEntityProxy(new PropertiesEntity(this.axisChart.yTicks));
        this.axisChart.addIOEntityProxy(new DisplayersEntity());
        this.axisChart.addIOEntityProxy(new ComponentEntity());
        this.axisChart.addIOEntityProxy(new InternalFrameEntity());
    }

    public class WorkspaceEntity implements IOEntity {

        public void save(Writer writer) throws IOException {
            writer.write(axisChart.getChartSurface().getBackground().getRed());
            writer.write(axisChart.getChartSurface().getBackground().getGreen());
            writer.write(axisChart.getChartSurface().getBackground().getBlue());
            writer.write(axisChart.automaticalRefresh);
            writer.write(axisChart.automaticalAdjusting);
            writer.write(axisChart.automaticalShiftingToEnd);
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {
            int r = reader.readInt();
            int g = reader.readInt();
            int b = reader.readInt();
            axisChart.getChartSurface().setBackground(new Color(r, g, b));
            axisChart.automaticalRefresh = reader.readBoolean();
            axisChart.automaticalAdjusting = reader.readBoolean();
            axisChart.automaticalShiftingToEnd = reader.readBoolean();
        }
    }

    public class PropertiesEntity implements IOEntity {

        private Ticks ticks = null;

        public PropertiesEntity(Ticks ticks) {
            this.ticks = ticks;
        }

        public void save(Writer writer) throws IOException {
            writer.write(ticks.displayedTicks.size());
            for (int fa = 0; fa < ticks.displayedTicks.size(); fa++) {
                writer.write(ticks.displayedTicks.get(fa).getNumber().floatValue());
            }
            writer.write(ticks.lowTick.floatValue());
            writer.write(ticks.greatTick.floatValue());
            writer.write(ticks.labelsLength);
            int index = axisChart.propertiesPanel.indexOfValuesColumn();
            String[] text = {
                axisChart.propertiesPanel.getText(index, 0),
                axisChart.propertiesPanel.getText(index, 1),
                axisChart.propertiesPanel.getText(index, 2),
                axisChart.propertiesPanel.getText(index, 3),
                axisChart.propertiesPanel.getText(index, 4),
                axisChart.propertiesPanel.getText(index, 5),
                axisChart.propertiesPanel.getText(index, 6),
                axisChart.propertiesPanel.getText(index, 7)
            };
            writer.write(index);
            writer.write(text.length);
            for (int fa = 0; fa < text.length; fa++) {
                writer.write(text[fa].length());
                writer.write(text[fa]);
            }
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {

            int size = reader.readInt();
            float[] fticks = new float[size];
            for (int fa = 0; fa < size; fa++) {
                fticks[fa] = reader.readFloat();
            }
            float lowTick = reader.readFloat();
            float greatTick = reader.readFloat();
            int length = reader.readInt();
            ticks.setDisplayedTicks(fticks);
            ticks.setLowestTick(lowTick);
            ticks.setGreatestTick(greatTick);
            ticks.setLengthOfLabels(length);
            int index = reader.readInt();
            int n = reader.readInt();
            for (int fa = 0; fa < n; fa++) {
                int strLength = reader.readInt();
                String text = reader.readString(strLength);
                axisChart.propertiesPanel.setText(text, index, fa);
            }

        }
    }

    public class DisplayersEntity implements IOEntity {

        private void tryDisplay(DisplayerInfo displayerInfo, DataBundle bundleData) {
            List<DataRepository> dataRepositorys = new ArrayList<DataRepository>();
            for (RepositoryInfo info : displayerInfo.repositories) {
                if (bundleData.getSymbolicName().equals(info.symbolicName)) {
                    for (int fa = 0; fa < bundleData.size(); fa++) {
                        DataRepository dataRepositoriesGroup = bundleData.get(fa);
                        if (dataRepositoriesGroup.getBundle().getSymbolicName().equals(info.symbolicName)
                                && dataRepositoriesGroup.getLabel().equals(info.label)) {
                            dataRepositorys.add(dataRepositoriesGroup);
                        }

                    }
                }
            }
            if (dataRepositorys.size() == 0) {
                return;
            }
            DisplayerImpl[] displayer = new DisplayerImpl[1];
            DataRepository[] array = new DataRepository[dataRepositorys.size()];
            array = dataRepositorys.toArray(array);
            AxisIndices axisIndices = new AxisIndices(displayerInfo.rawIndexX,
                    displayerInfo.sideX, displayerInfo.rawIndexY,
                    displayerInfo.sideY);
            DisplayerTools.create(axisChart, array, displayerInfo.symbolicNameOfPlotFactory,
                    displayerInfo.savedContentOfPlotManager, displayer, displayerInfo.name,
                    axisIndices);
            if (displayer[0] != null) {
                listDisplayers.add(displayer[0]);
            }
        }

        private class ListListenerimpl implements ListListener<DisplayerImpl> {

            public void isAdded(DisplayerImpl t) {
                for (int fa = 0; fa < listDataBundle.size(); fa++) {
                    DataBundle dataBundle = listDataBundle.get(fa);
                    for (DisplayerInfo displayerInfo : displayed) {
                        tryDisplay(displayerInfo, dataBundle);
                    }
                }
            }

            public void isRemoved(DisplayerImpl t) {
            }
        }

        public DisplayersEntity() {
            listDisplayers.addListListener(new ListListenerimpl());
        }

        private class DisplayerInfo {
            //

            int rawIndexX;
            //
            int rawIndexY;
            // type for x from which end should be calculated x (From Begin or From End)
            int sideX;
            // type for y from which end should be calculated y (From Begin or From End)
            int sideY;
            public String name;
            public byte[] savedContentOfPlotManager;
            String symbolicNameOfPlotFactory;
            List<RepositoryInfo> repositories = new ArrayList<RepositoryInfo>();
        }
        private List<DisplayerInfo> displayed = new ArrayList<DisplayerInfo>();

        public void save(Writer writer) throws IOException {
            int numberOfRepositories = 0;
            int numberOfDisplayers = 0;
            Writer localWriter = new Writer();
            for (int x = 0; x < listDisplayers.size(); x++) {
                ogla.axischart.Displayer displayer = listDisplayers.get(x);
                DisplayerImpl displayerImpl = (DisplayerImpl) displayer;
                numberOfDisplayers++;
                for (int fb = 0; fb < listDataBundle.size(); fb++) {
                    DataBundle bundleData = listDataBundle.get(fb);
                    for (int fa = 0; fa < displayerImpl.getNumberOfDataRepositories(); fa++) {
                        DataRepository dataRepository = displayerImpl.getDataRepository(fa);
                        if (dataRepository.getBundle() == bundleData) {
                            numberOfRepositories++;
                            localWriter.write(bundleData.getSymbolicName().length());
                            localWriter.write(bundleData.getSymbolicName());
                            localWriter.write(dataRepository.getLabel().length());
                            localWriter.write(dataRepository.getLabel());
                        }
                    }
                }
            }
            writer.write(numberOfDisplayers);
            int i = 0;
            for (int fa = 0; fa < listDisplayers.size(); fa++) {
                ogla.axischart.Displayer displayerInfo = listDisplayers.get(fa);
                DisplayerImpl displayer = (DisplayerImpl) displayerInfo;
                writer.write(displayer.getLabel().length());
                writer.write(displayer.getLabel());
                writer.write(displayer.getAxisIndices().getRawXIndex());
                writer.write(displayer.getAxisIndices().getSideX());
                writer.write(displayer.getAxisIndices().getRawYIndex());
                writer.write(displayer.getAxisIndices().getSideY());
                writer.write(displayer.getPlotManager().getPlotManagerFactory().getSymbolicName().length());
                writer.write(displayer.getPlotManager().getPlotManagerFactory().getSymbolicName());
                if (displayer.getPlotManager().getPlotManagerFactory().canBeSaved()) {
                    byte[] bytes = displayer.getPlotManager().save();
                    writer.write(bytes.length);
                    writer.write(bytes);
                } else {
                    writer.write(0);
                }
                writer.write(numberOfRepositories);
                writer.write(localWriter.getBytes());
            }
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {
            int size = reader.readInt();
            for (int fa = 0; fa < size; fa++) {
                String nameOfDisplayer = reader.readString(reader.readInt());
                int rawX = reader.readInt();
                int sideX = reader.readInt();
                int rawY = reader.readInt();
                int sideY = reader.readInt();
                String symbolicNameOfPlotFactory = reader.readString(reader.readInt());
                int length = reader.readInt();
                byte[] bytes = null;
                if (length != 0) {
                    bytes = new byte[length];
                    reader.readBytes(bytes);
                }
                int numberofRepositories = reader.readInt();
                DisplayerInfo displayerInfo = new DisplayerInfo();
                displayerInfo.name = nameOfDisplayer;
                displayerInfo.rawIndexX = rawX;
                displayerInfo.sideX = sideX;
                displayerInfo.rawIndexY = rawY;
                displayerInfo.sideY = sideX;
                displayerInfo.savedContentOfPlotManager = bytes;
                displayerInfo.symbolicNameOfPlotFactory = symbolicNameOfPlotFactory;
                for (int fb = 0; fb < numberofRepositories; fb++) {
                    length = reader.readInt();
                    String symbolicName = reader.readString(length);
                    length = reader.readInt();
                    String label = reader.readString(length);

                    RepositoryInfo repositoryInfo = new RepositoryInfo();
                    repositoryInfo.label = label;
                    repositoryInfo.symbolicName = symbolicName;
                    displayerInfo.repositories.add(repositoryInfo);
                }
                displayed.add(displayerInfo);
                for (int fb = 0; fb < listDataBundle.size(); fb++) {
                    DataBundle dataBundle = listDataBundle.get(fb);
                    tryDisplay(displayerInfo, dataBundle);
                }
            }

        }
    }

    public class ComponentEntity implements IOEntity {

        private class ChartComponentInfo {

            public String symbolicName;
            public byte[] representation;
        }
        private List<ChartComponentInfo> chartComponentInfos = new ArrayList<ChartComponentInfo>();

        public void save(Writer writer) throws IOException {
            int size = 0;
            Writer localWriter = new Writer();
            for (int fa = 0; fa < listInstalledChartComponent.size(); fa++) {
                AxisChart2DComponent chartComponent = listInstalledChartComponent.get(fa);
                if (chartComponent.getBundle().canBeSaved()) {
                    byte[] representation = chartComponent.save();
                    if (representation != null && representation.length > 0) {
                        localWriter.write(chartComponent.getBundle().getSymbolicName().length());
                        localWriter.write(chartComponent.getBundle().getSymbolicName());

                        localWriter.write(representation.length);
                        localWriter.write(representation);
                        size++;
                    }
                }
            }
            writer.write(size);
            writer.write(localWriter.getBytes());
            localWriter.clear();
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {

            int size = reader.readInt();
            for (int fa = 0; fa < size; fa++) {
                int length = reader.readInt();
                String symbolicName = reader.readString(length);
                length = reader.readInt();
                byte[] representation = new byte[length];
                reader.readBytes(representation);
                ChartComponentInfo componentInfo = new ChartComponentInfo();
                componentInfo.symbolicName = symbolicName;
                componentInfo.representation = representation;
                chartComponentInfos.add(componentInfo);
            }
        }
    }

    public class InternalFrameEntity implements IOEntity {

        public void save(Writer writer) throws IOException {
            int size = axisChart.frames.length;
            writer.write(size);
            for (InternalFrame frame : axisChart.frames) {
                writer.write(frame.isVisible());
                writer.write(frame.getLocation().x);
                writer.write(frame.getLocation().y);
                writer.write(frame.getSize().width);
                writer.write(frame.getSize().height);
            }

            int x = axisChart.getChartSurface().getLocation().x;
            int y = axisChart.getChartSurface().getLocation().y;
            int width = axisChart.getChartSurface().getSize().width;
            int height = axisChart.getChartSurface().getSize().height;
            axisChart.repaintChartSurface();
            writer.write(x);
            writer.write(y);
            writer.write(width);
            writer.write(height);
            x = axisChart.chartCanvas.getLocation().x;
            y = axisChart.chartCanvas.getLocation().y;
            width = axisChart.chartCanvas.getSize().width;
            height = axisChart.chartCanvas.getSize().height;
            axisChart.getChartSurface().setSize(axisChart.chartCanvas.getSizeOfMainPanel());
            axisChart.getChartSurface().setPreferredSize(axisChart.chartCanvas.getSizeOfMainPanel());
            writer.write(x);
            writer.write(y);
            writer.write(width);
            writer.write(height);
        }

        public void load(Reader reader) throws IOException, Reader.EndOfBufferException {
            int length = reader.readInt();
            for (int fa = 0; fa < length; fa++) {
                boolean bool = reader.readBoolean();
                int x = reader.readInt();
                int y = reader.readInt();
                int width = reader.readInt();
                int height = reader.readInt();
                axisChart.frames[fa].setVisible(bool);
                axisChart.frames[fa].setLocation(x, y);
                axisChart.frames[fa].setSize(width, height);
                if (bool == true) {
                    axisChart.chartPanel.setComponentZOrder(axisChart.frames[fa], 0);
                }
            }

            int dx = reader.readInt();
            int dy = reader.readInt();
            int dwidth = reader.readInt();
            int dheight = reader.readInt();
            int x = reader.readInt();
            int y = reader.readInt();
            int width = reader.readInt();
            int height = reader.readInt();
            axisChart.chartCanvas.setLocation(x, y);
            axisChart.chartCanvas.setSize(width, height);
            axisChart.chartCanvas.getParent().repaint();
            axisChart.getChartSurface().setLocation(dx, dy);
            axisChart.getChartSurface().setSize(dwidth, dheight);
            axisChart.repaintChartSurface();
            axisChart.chartPanel.revalidate();
            axisChart.chartPanel.repaint();
        }
    }
}
