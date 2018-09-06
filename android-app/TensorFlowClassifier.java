package germanmalsagov.tfliteandroid.models;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.os.SystemClock;
import android.util.Log;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;

import static android.content.ContentValues.TAG;

//lets create this classifer
public class TensorFlowClassifier {

    static {
        System.loadLibrary("tensorflow_inference");
    }

//Initialize files and variables needed for inference
    private static final String MODEL_FILE = "model2.tflite";
    private static final long[][] OUTPUT_SIZE = new long[1][1];


//Instantiate Interpreter class
    protected Interpreter tflite;

//    Constructor
    public TensorFlowClassifier(final Activity activity) throws IOException {
        tflite = new Interpreter(loadTfLiteModel(activity, MODEL_FILE));
    }

    public Object[] predictEmotion(int[] data, int[] inf_delay) {

        if (tflite == null) {
            Log.e(TAG, "Classifier has not been initialized.");
        }

        long startTime = SystemClock.uptimeMillis();
        tflite.run(data, OUTPUT_SIZE);
        long endTime = SystemClock.uptimeMillis();

        int delay = (int) (endTime - startTime);
        for (int i = 0; i < inf_delay.length; i++) {
            if (inf_delay[i] == 0) {
                inf_delay[i] = delay;
                break;
            }
        }

        Object[] results = {OUTPUT_SIZE, inf_delay};
        Log.d(TAG, "Time cost to run model inference: " + Integer.toString(delay));

        return results;
    }

    private MappedByteBuffer loadTfLiteModel(Activity activity, String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

}
