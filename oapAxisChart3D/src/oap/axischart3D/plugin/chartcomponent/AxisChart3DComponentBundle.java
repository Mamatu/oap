package ogla.axischart3D.plugin.chartcomponent;

import ogla.core.AdvancedInformation;

public interface AxisChart3DComponentBundle extends AdvancedInformation {

    public String getLabel();

    public AxisChart3DComponent newAxisChart3DComponent();
}
