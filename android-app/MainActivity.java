package germanmalsagov.tfliteandroid;

import android.app.Activity;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.PointF;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import germanmalsagov.tfliteandroid.models.TensorFlowClassifier;

import static android.content.ContentValues.TAG;
import static android.os.Parcelable.PARCELABLE_WRITE_RETURN_VALUE;

public class MainActivity extends Activity implements View.OnClickListener {

    private static final int SENTENCE_LENGTH = 40;

    protected Context context;

    // ui elements
    private Button clearBtn, classBtn;
    private TextView resText;
    private List<Classifier> mClassifiers = new ArrayList<>();

    // views
    private TextView typeHere;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        typeHere = (TextView) findViewById(R.id.typeHere);

        // give it a touch listener to activate when the user taps
        typeHere.setOnClickListener(clickListener);

        //clear button
        //clear the drawing when the user taps
        clearBtn = (Button) findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        // performs classification on the drawn image
        classBtn = (Button) findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        // text that shows the output of the classification
        resText = (TextView) findViewById(R.id.Recognize);

    }

    private View.OnClickListener clickListener= new View.OnClickListener() {
        public void onClick(View v) {
            typeHere.setText("");
        }
    };

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    protected void onResume() {
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    protected void onPause() {
        super.onPause();
    }

    @Override
    public void onClick(View view) {
        //if user clicks something
        if (view.getId() == R.id.btn_clear) {
            //if its the clear button
            //clear the drawing
            typeHere.setText("");
            typeHere.invalidate();
            //empty the text view
            resText.setText("");
        }
        else if (view.getId() == R.id.btn_class) {
            //if the user clicks the classify button

            String line = null;
            String[] sentences = null;
            String[] labels_true = null;
            String[] labels = new String[7];
            HashMap<String, Integer> map = new HashMap();

            try {
                BufferedReader br1 = new BufferedReader(new InputStreamReader(
                        getAssets().open("vocab.csv")));

                BufferedReader br2 = new BufferedReader(new InputStreamReader(
                        getAssets().open("isear_test.csv")));


                BufferedReader br3 = new BufferedReader(new InputStreamReader(
                        getAssets().open("labels.txt")));

                int num = 0;

//               Read labels file and store in array
                while((line=br3.readLine())!=null) {
                    labels[num] = line;
                    num++;
                }

//                Read dictionary into a hashmap
                while((line=br1.readLine())!=null) {
                    String str[] = line.split(",");
                    map.put(str[0], Integer.valueOf(str[1]));

                }

                int i = 0;
                line = null;
                String[] splitted = null;
                labels_true = new String[250];
                sentences = new String[250];

//                Skip through lines before n
                while((line=br2.readLine())!=null && num < 250) {
                    // Regular Expression to split sentences correctly
                    for(int k = 0; k < 10; k++)
                        splitted = line.split(",(?=([^\"]*\"[^\"]*\")*[^\"]*$)");
                    labels_true[num] = splitted[0];
                    sentences[num] = splitted[1];
                    num++;
                }

//                Close buffer readers
                br1.close();
                br2.close();
                br3.close();

            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            int size = sentences.length;

//            Split sentence into words and store in array
            String[] words = null;
            int[][] encoded_words = new int[sentences.length][40];

            for (int j = 0; j < sentences.length;j++) {

                if (sentences[j] != null) {
                    words = sentences[j].split(" ");
                } else {
                    continue;
                }

//              Encode words into int ids from dictionary
                for (int i = 0; i < words.length; i++) {

                    if (i < encoded_words[j].length) {
                        Integer value = map.get(words[i]);

                        int id = 0;
                        if (value != null) {
                            id = Integer.parseInt(value.toString());
                        }

                        encoded_words[j][i] = id;

                    } else {
                        break;
                    }
                }

            }

            Context context = this;
            Activity activity = (Activity) context;
            long[][] score = new long[1][1];
            int[] inf_delay = new int[size];

//            Initialize TensorflowClassifier class
            TensorFlowClassifier classifier = null;
            Object[] results;
            int correct = 0;
            for (int i = 0; i < encoded_words.length; i++){

                try {
//                Pass activity to classifier object
                    classifier = new TensorFlowClassifier(activity);
                    results = new Object[2];
//                Run inference
                    results = classifier.predictEmotion(encoded_words[i], inf_delay);

                    score = (long[][]) results[0];
                    inf_delay = (int[]) (results[1]);

                    int index = (int) score[0][0];
                    String predicted = labels[index];

                    Log.d(TAG,"True: " + labels_true[i] + "\n" +" Predicted: " + predicted);

                    if (predicted.equals(labels_true[i])) {

                        correct += 1;
                    }

                } catch (IOException e) {
                    e.printStackTrace();
                }

            }

            double accuracy = (double) correct / 500;

            Log.d(TAG, "Accuracy: " + Double.toString(accuracy));


        }
    }

    public void writeFile(int[] num_words, int[] inf_delay) {

    }

}