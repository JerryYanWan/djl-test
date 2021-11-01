package com.xx.examples;

import com.xx.examples.MyLoadedModel;

import java.io.IOException;
import java.nio.file.*;

import ai.djl.*;
import ai.djl.inference.*;
import ai.djl.modality.*;
import ai.djl.modality.cv.*;
import ai.djl.modality.cv.util.*;
import ai.djl.modality.cv.transform.*;
import ai.djl.modality.cv.translator.*;
import ai.djl.repository.zoo.*;
import ai.djl.translate.*;
import ai.djl.training.util.*;
import ai.djl.ndarray.*;

import java.util.Arrays;

public class Test2 {

    public static void main(String[] args) throws Exception {

        Predictor<Float[], Float[]> predictor = MyLoadedModel.predictor;

        System.out.println("First input:");
        Float[] sampleInput = new Float[]{100.0f, 100.0f, 200.0f, 250.0f, 300.0f};
        Float[] ret;

        ret = predictor.predict(sampleInput);
        System.out.println("output:");
        for (int i = 0, L = ret.length; i<L; i++) {
            System.out.println(ret[i]);
        }

        System.out.println("Second input:");
        Float[] sampleInput2 = new Float[]{10000.0f, 20000.0f, 20000.0f, 25000.0f, 30000.0f};
        Float[] ret2;

        ret2 = predictor.predict(sampleInput2);
        System.out.println("output:");
        for (int i = 0, L = ret2.length; i<L; i++) {
            System.out.println(ret2[i]);
        }

        System.out.println(Arrays.toString(ret2));

    }
}
