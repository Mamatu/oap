/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ogla.g3d;

import java.util.logging.Level;
import java.util.logging.Logger;
import ogla.math.Matrix;
import ogla.math.Quaternion;
import ogla.math.Vector3;

/**
 *
 * @author mmatula
 */
public class Camera {

    Vector3 lookAt = new Vector3(0, 0, 0);
    Vector3 position = new Vector3(0, 0, 20);
    Vector3 orientation = new Vector3(0, 1, 0);

    private Vector3 temp = new Vector3();
    private Vector3 temp1 = new Vector3();
    private Vector3 tempCross = new Vector3();
    private Vector3 upVector = new Vector3(0, 1, 0);
    private Quaternion quaternion = new Quaternion();
    private Matrix matrix = new Matrix(3, 3);

    public void setCameraCoord(Vector3 position, Vector3 lookAt) {
        temp1.set(lookAt);
        temp1.substract(position);
        temp = temp1.projectOnPlane(upVector, temp);
        tempCross.set(temp);
        tempCross.crossProduct(temp1);
        tempCross.normalize();
        temp1.normalize();
        temp.normalize();
        float angle = temp1.dotProduct(temp);
        quaternion.set(angle, tempCross);
        matrix = quaternion.toMatrix(matrix);
        this.position.set(position);
        this.lookAt.set(lookAt);
        try {
            this.orientation.rotate(matrix);
            this.orientation.normalize();
        } catch (Vector3.InvalidMatrixDimensionException ex) {
            Logger.getLogger(Camera.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
