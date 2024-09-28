import time
import logging
import subprocess
from os.path import join
from pprint import pprint
import argparse
from upaths_wx import preprocessing_script_dpath

# Constants
MODES = ['from_genvrt', 'from_gentiles', 'from_genfeatures']
SCRIPTS = {
  'from_genvrt': ["a_genvrts.py", "b_gentiles.py", "c_genfeatures.py"],
  'from_gentiles': ["b_gentiles.py", "c_genfeatures.py"],
  'from_genfeatures': ["c_genfeatures.py"]
}

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_preprocessing_steps(modeid):
  try:
      mode = MODES[modeid]
      scripts = SCRIPTS[mode]
      print('='*100)
      print('mode', mode)
      print('scripts', scripts)
      return scripts
  except IndexError:
      logging.error(f"Invalid modeid: {modeid}. Must be between 0 and {len(MODES) - 1}.")
      return []
  except KeyError:
      logging.error(f"No scripts defined for mode: {mode}.")
      return []

def run_scripts_in_sequence(script_list):
  for script in script_list:
      try:
          ti = time.process_time()
          logging.info(f"Running {script}...")
          result = subprocess.run(['python', script], check=True, text=True, capture_output=True)
          logging.info(f"Output of {script}:\n{result.stdout}")
          logging.info(f"Errors of {script}:\n{result.stderr}")
          tf = time.process_time()
          logging.info(f'Run time for {script}: {(tf - ti) / 60:.2f} mins')
      except subprocess.CalledProcessError as e:
          logging.error(f"An error occurred while running {script}: {e}")
          logging.error(f"Output:\n{e.output}")
          logging.error(f"Errors:\n{e.stderr}")
          break

  logging.info("All scripts processed.")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run preprocessing scripts based on modeid.')
  parser.add_argument('modeid', type=int, help='Mode ID to select preprocessing steps (0, 1, or 2)')
  args = parser.parse_args()

  modeid = args.modeid

  preprocessing_steps = get_preprocessing_steps(modeid)
  if not preprocessing_steps:
      logging.error("No preprocessing steps found. Exiting.")
  else:
      script_list = [join(preprocessing_script_dpath, x) for x in preprocessing_steps]

      ti = time.perf_counter()
      pprint(script_list)
      run_scripts_in_sequence(script_list)
      tf = time.perf_counter()
      logging.info(f'Total run time: {(tf - ti) / 60:.2f} mins')