/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.chart;

import java.nio.ByteBuffer;
import ogla.core.ui.OglaMethod;
import ogla.core.ui.OglaObject;
import ogla.core.util.ArgumentType;
import ogla.core.util.ArgumentsUtils;

/**
 *
 * @author mmatula
 */
public abstract class ChartObject {

    public static abstract class CreatorObject {

        private OglaObject groovyObject = null;

        public CreatorObject(String name) {
            this.groovyObject = new OglaObject(name, null, null);
            this.groovyObject.addMethod(this.createMetod);
        }

        protected abstract ChartObject createChartObject(String name);

        private final OglaMethod createMetod = new OglaMethod() {

            @Override
            public Object invoke(Object args, OglaObject oglaObject, Object userData) {
                ByteBuffer byteBuffer = null;
                ArgumentType[] argsTypes = {ArgumentType.ARGUMENT_TYPE_STRING};
                byteBuffer = ArgumentsUtils.convertObject(args, argsTypes);
                if (byteBuffer == null) {
                    int length = byteBuffer.getInt();
                    StringBuilder builder = new StringBuilder();
                    for (int fa = 0; fa < length; fa++) {
                        builder.append(byteBuffer.getChar());
                    }
                    String name = builder.toString();
                    ChartObject chartObject = createChartObject(name);
                    if (chartObject == null) {
                        throw new NullPointerException("Instance of chart object can't be null.");
                    }
                }
                return null;
            }
        };
    }

    private OglaObject oglaObject = null;

    public ChartObject(String name) {
        this.oglaObject = new OglaObject(name, null, null);
        this.oglaObject.addMethod(this.plotMetod);
    }

    protected abstract Chart createChart(ChartSurface chartSurface);

    private Chart chart = null;
    private ChartFrame chartFrame = null;
    private ChartPanel chartPanel = null;

    private final OglaMethod plotMetod = new OglaMethod() {

        @Override
        public Object invoke(Object args, OglaObject oglaObject, Object userData) {
            if (ChartObject.this.chartPanel == null) {
                ChartObject.this.chartPanel = new ChartPanel();
                ChartObject.this.chartFrame = new ChartFrame(ChartObject.this.chartPanel);
                chart = ChartObject.this.createChart(ChartObject.this.chartPanel.getChartSurface());
            }
            ChartObject.this.chart.repaintChartSurface();
            return null;
        }
    };
}
