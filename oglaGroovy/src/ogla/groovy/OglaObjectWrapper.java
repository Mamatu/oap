/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.groovy;

import groovy.lang.GroovyObjectSupport;
import java.util.List;
import ogla.core.ui.OglaObject;

public class OglaObjectWrapper extends GroovyObjectSupport {

    private OglaObject oglaObject = null;

    public OglaObject getOglaObject() {
        return oglaObject;
    }

    public OglaObjectWrapper(OglaObject oglaObject) {
        this.oglaObject = oglaObject;
    }

    public OglaObjectWrapper(String name, OglaObject root, List<OglaObject> roots) {
        this.oglaObject = new OglaObject(name, root, roots);
    }

    @Override
    public Object invokeMethod(String arg0, Object arg1) {
        Object object = this.oglaObject.invokeMethod(arg0, arg1);
        return object;
    }
}
