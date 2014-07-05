package ogla.legend;

import ogla.axischart.AxisChart;
import ogla.axischart.Displayer;
import ogla.axischart.plugin.image.ImageComponentBundle;
import ogla.axischart.plugin.image.ImageComponent;
import ogla.chart.Chart;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;

public class Legend implements ImageComponent {

    private ImageComponentBundle bundleImageComponent;
    private ConfigurationFrame confFrame = null;
    private BufferedImage currentBufferedImage = null;
    private AxisChart axisChart = null;
    private Chart chart = null;

    public Legend(ImageComponentBundle bundleImageComponent) {
        this.bundleImageComponent = bundleImageComponent;
    }

    @Override
    public void setBufferedImage(BufferedImage bufferedImage) {
        currentBufferedImage = bufferedImage;
    }

    @Override
    public void attachChart(AxisChart axisChart) {
        if (axisChart == null) {
            return;
        }
        this.axisChart = axisChart;
        confFrame = new ConfigurationFrame(chart, axisChart);
    }

    public void process() {
    }

    @Override
    public byte[] save() {
        return null;
    }

    @Override
    public ImageComponentBundle getBundle() {
        return bundleImageComponent;
    }

    @Override
    public JFrame getFrame() {
        confFrame.configurationPanel.updateList();
        if (confFrame.configurationPanel.map.size() <= 0) {
            return null;
        }
        return confFrame;
    }

    private int getLengthOfText(String text, Graphics2D gd) {
        int length = 0;
        for (int fa = 0; fa < text.length(); fa++) {
            length += gd.getFontMetrics().charWidth(text.charAt(fa));
        }
        return length;
    }

    private int getLengthOfLongest(String[] labels, Graphics2D gd) {
        if (labels.length <= 0) {
            return 0;
        }
        int max = getLengthOfText(labels[0], gd);
        for (int fa = 1; fa < labels.length; fa++) {
            int l = getLengthOfText(labels[fa], gd);
            if (l > max) {
                max = l;
            }
        }
        return max;
    }

    private int getHeightOfLegendPart(String[] labels, Graphics2D gd) {
        int n = labels.length;
        int height = 2 * n * gd.getFont().getSize();
        return height;
    }
    Rectangle2D.Float rect = new Rectangle2D.Float();

    @Override
    public BufferedImage getBufferedImage() {
        confFrame.configurationPanel.updateList();
        String[] labels = new String[axisChart.getListDisplayerInfo().size()];
        for (int fa = 0; fa < axisChart.getListDisplayerInfo().size(); fa++) {
            Displayer displayer = axisChart.getListDisplayerInfo().get(fa);
            labels[fa] = confFrame.configurationPanel.map.get(displayer);
        }
        Graphics2D currentGraphics2D = (Graphics2D) currentBufferedImage.getGraphics();

        int height = currentBufferedImage.getHeight();
        int partOfHeight = getHeightOfLegendPart(labels, currentGraphics2D);

        height += partOfHeight;
        int width = getLengthOfLongest(labels, currentGraphics2D);
        if (width < currentBufferedImage.getWidth()) {
            width = currentBufferedImage.getWidth();
        }

        BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics2D = (Graphics2D) bufferedImage.getGraphics();
        graphics2D.setBackground(chart.getChartSurface().getBackground());
        graphics2D.clearRect(0, 0, bufferedImage.getWidth(), bufferedImage.getHeight());

        graphics2D.drawImage(currentBufferedImage, 0, 0, null);

        for (int fa = 0; fa < labels.length; fa++) {
            String label = labels[fa];
            Displayer displayer = (Displayer) axisChart.getListDisplayerInfo().get(fa);
            displayer.getPlotManager().getGraphicSymbol(0, 0, 100, graphics2D.getFont().getSize(), graphics2D);
            graphics2D.drawString(label, 100, partOfHeight + fa * graphics2D.getFont().getSize());
        }

        graphics2D.drawImage(currentBufferedImage, null, 0, 0);
        Color defaultColor = graphics2D.getColor();

        graphics2D.setColor(defaultColor);
        return bufferedImage;
    }
}
