# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Computes metrics for Google Landmarks Retrieval dataset predictions.

Metrics are written to stdout.
"""
import argparse
import sys, os
sys.path.append(os.getcwd())
sys.path.append("./torchImageRecognition/eval/gldv2")
import metrics
import dataset_file_io

cmd_args = None

def metric_gldv2(predictions_path, solution_path = '/workspace/mnt/storage/zhangjunkang/zjk1/GLDv2/test_labels/retrieval_solution_v2.1.csv'):
  # Read solution.
  print('Reading solution...')
  public_solution, private_solution, ignored_ids = dataset_file_io.ReadSolution(
      solution_path, dataset_file_io.RETRIEVAL_TASK_ID)
  print('done!')

  # Read predictions.
  print('Reading predictions...')
  public_predictions, private_predictions = dataset_file_io.ReadPredictions(
      predictions_path, set(public_solution.keys()),
      set(private_solution.keys()), set(ignored_ids),
      dataset_file_io.RETRIEVAL_TASK_ID)
  print('done!')

  # Mean average precision.
  print('**********************************************')
  print('(Public)  Mean Average Precision: %f' %
        metrics.MeanAveragePrecision(public_predictions, public_solution))
  print('(Private) Mean Average Precision: %f' %
        metrics.MeanAveragePrecision(private_predictions, private_solution))

  # Mean precision@k.
  print('**********************************************')
  public_precisions = 100.0 * metrics.MeanPrecisions(public_predictions,
                                                     public_solution)
  private_precisions = 100.0 * metrics.MeanPrecisions(private_predictions,
                                                      private_solution)
  print('(Public)  Mean precisions: P@1: %.2f, P@5: %.2f, P@10: %.2f, '
        'P@50: %.2f, P@100: %.2f' %
        (public_precisions[0], public_precisions[4], public_precisions[9],
         public_precisions[49], public_precisions[99]))
  print('(Private) Mean precisions: P@1: %.2f, P@5: %.2f, P@10: %.2f, '
        'P@50: %.2f, P@100: %.2f' %
        (private_precisions[0], private_precisions[4], private_precisions[9],
         private_precisions[49], private_precisions[99]))

  # Mean/median position of first correct.
  print('**********************************************')
  public_mean_position, public_median_position = metrics.MeanMedianPosition(
      public_predictions, public_solution)
  private_mean_position, private_median_position = metrics.MeanMedianPosition(
      private_predictions, private_solution)
  print('(Public)  Mean position: %.2f, median position: %.2f' %
        (public_mean_position, public_median_position))
  print('(Private) Mean position: %.2f, median position: %.2f' %
        (private_mean_position, private_median_position))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--predictions_path',
      type=str,
      default='/workspace/mnt/storage/zhangjunkang/zjk1/GLDv2/test_labels/retrieval_solution_v2.1.csv',
      help="""
      Path to CSV predictions file, formatted with columns 'id,images' (the
      file should include a header).
      """)
  parser.add_argument(
      '--solution_path',
      type=str,
      default='/workspace/mnt/storage/zhangjunkang/zjk1/GLDv2/test_labels/retrieval_solution_v2.1.csv',
      help="""
      Path to CSV solution file, formatted with columns 'id,images,Usage'
      (the file should include a header).
      """)
  args = parser.parse_args()
  metric_gldv2(args.predictions_path, args.solution_path)
