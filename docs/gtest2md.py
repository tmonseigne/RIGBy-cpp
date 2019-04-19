# coding: utf8

import sys
import os
import shutil
import glob
import math

import xml.etree.ElementTree as ET

def usage():
	print('Usage:')
	print('  python gtest2md.py <REPORT_FILE> <OUTPUT_FILE>')
	print('  Args:')
	print('    REPORT_FILE: Gtest xml report.')
	print('    OUTPUT_FILE: Path to the output file.')

###################################
def convert_timestamp(timestamp):
	[date, time] = timestamp.split("T")
	[Y,M,D] = date.split("-")
	return [D + "-" + M + "-" + Y, time]


###################################
def get_xml_attribute(convert_func, xml_node, attribute_name, default_value, print_warning=True):
	if not attribute_name in xml_node.attrib and print_warning:
		print('Warning: Attribute {!r} was not found inside xml node {!s}[{!r}]. Set it to its default value {!r}.'.format(
			attribute_name, xml_node.tag, xml_node.attrib, default_value))
		return default_value
	return convert_func(xml_node.attrib[attribute_name])


###################################
def generate_summary(xml_testsuites_node):
	# Parse the testsuites node attributes.
	total_test = get_xml_attribute(int, xml_testsuites_node, 'tests', 0)
	total_fail = get_xml_attribute(int, xml_testsuites_node, 'failures', 0)
	total_disabled = get_xml_attribute(int, xml_testsuites_node, 'disabled', 0)
	total_success = total_test - total_fail - total_disabled
	total_execution_time = get_xml_attribute(str, xml_testsuites_node, 'time', 0)
	timestamp = get_xml_attribute(str, xml_testsuites_node, 'timestamp', 0)
	[date, time] = convert_timestamp(timestamp)

    # Generate the progress bar for the test summary.

    # Generate the table for the test summary.
	res =	"| Date | Time | Tests | Success | Fails | Disabled | Execution time |\n"+\
			"|:----:|:----:|:-----:|:-------:|:-----:|:--------:|---------------:|\n"+\
			"| " + date + " | " + time + \
			" | <span style=\"color:DarkBlue\">" + str(total_test) + \
			"</span> | <span style=\"color:DarkGreen\">" + str(total_success) + \
			"</span> | <span style=\"color:DarkRed\">" + str(total_fail) + \
			"</span> | <span style=\"color:DarkOrange\">" + str(total_disabled) + \
			"</span> | " + str(total_execution_time) + " sec |\n\n"

	return res

###################################
def generate_tests_result(xml_testsuites_node):
	# Generate HTML for the single test result listings.
	xml_testsuite_nodes = xml_testsuites_node.findall('./testsuite')
	if len(xml_testsuite_nodes) == 0:
		print('Warning: No nodes {!r} found in {!r}. Nothing is listed inisde the single test_result listing.'.format('testsuite', report_file))
	
	res = ""

	for xml_testsuite_node in xml_testsuite_nodes:
		# Parse XML.
		testsuite_name = get_xml_attribute(str, xml_testsuite_node, 'name', '-undefined-')
		testsuite_test = get_xml_attribute(int, xml_testsuite_node, 'tests', 0)
		testsuite_fails = get_xml_attribute(int, xml_testsuite_node, 'failures', 0)
		testsuite_disabled = get_xml_attribute(int, xml_testsuite_node, 'disabled', 0)
		testsuite_success = testsuite_test - testsuite_fails - testsuite_disabled
		testsuite_execution_time = get_xml_attribute(str, xml_testsuite_node, 'time', '-undefined-')

		# Generate the progress bar for the testsuite summary.

		res +=	"## Testsuite: " + testsuite_name + "\n### Summary\n\n"+\
				"| Tests | Success | Fails | Disabled | Execution time |\n"+\
				"|:-----:|:-------:|:-----:|:--------:|---------------:|\n"+\
				" | <span style=\"color:DarkBlue\">" + str(testsuite_test) + \
				"</span> | <span style=\"color:DarkGreen\">" + str(testsuite_success) + \
				"</span> | <span style=\"color:DarkRed\">" + str(testsuite_fails) + \
				"</span> | <span style=\"color:DarkOrange\">" + str(testsuite_disabled) + \
				"</span> | " + str(testsuite_execution_time) + " sec |\n\n"
		
		res += generate_testcases(xml_testsuite_node)

	return res

def generate_testcases(xml_testsuite_node):
	xml_testcase_nodes = xml_testsuite_node.findall('./testcase')
	if len(xml_testcase_nodes) == 0:
		print("Warning: No nodes {!r} found in testsuite element with name {!r}.".format('testcase', test_name))

	# Generate the table for the testcases.
	res =	"### Testcases: \n\n"+\
			"| # | Name | Execution time | Status |\n"+\
			"|:-:|:-----|---------------:|:------:|\n"

	for idx, xml_testcase_node in enumerate(xml_testcase_nodes):
		test_number = idx + 1
		test_name = get_xml_attribute(str, xml_testcase_node, 'name', '-undefined-')
		test_execution_time = get_xml_attribute(str, xml_testcase_node, 'time', '-undefined-')
		test_status = get_xml_attribute(str, xml_testcase_node, 'status', '-undefined-')

		test_icon = ''
		# Select icon name and html class considering the number of failure-children and the test status.
		xml_failure_nodes = xml_testcase_node.findall('./failure')
		
		if len(xml_failure_nodes) == 0 and test_status == 'run':
			test_icon = '✔️'
		elif test_status == 'notrun':
			test_icon = '⚠️'
		else:
			test_icon = '❌'

        # If failures occurs, generate the listing with error messages.
		fail_msg = ""
		if len(xml_failure_nodes) > 0:
			fail_msg = "<br>"
			for xml_failure_node in xml_failure_nodes:
				error_message = get_xml_attribute(str, xml_failure_node, 'message', '-undefined-')
				fail_msg+=error_message.split("\\")[-1].replace("\n"," <br>")

        # generate Row
		res += "| " + str(test_number) + " | " + test_name + fail_msg + " | " + test_execution_time + " sec | " + test_icon + " |\n"

	return res + "\n"

###################################
def generate_md(report_file, destination_file):
	# Parse XML.
	xml_tree = ET.parse(report_file)
	xml_root = xml_tree.getroot()

	# Check if the root element 'testsuites' exists.
	xml_testsuites_nodes = xml_root.findall('.')
	if len(xml_testsuites_nodes) == 0:
		print('Error: The xml file {!r} has no root node.'.format(report_file))
		exit(-1)
	xml_testsuites_node = xml_testsuites_nodes[0]
	if xml_testsuites_node.tag != 'testsuites':
		print('Error: The xml file {!r} has an invalid root node tag (found: {!r}, expected: {!r})'.format(
			report_file, xml_testsuites_node.tag, 'testsuites'))
		exit(-1)

	# Generation
	Title = "# Google Test Report \n## Summary\n\n"
	Summary = generate_summary(xml_testsuites_node)
	TestsResults = generate_tests_result(xml_testsuites_node)

	md = Title + Summary + TestsResults

	with open(destination_file, 'w',encoding="utf-8") as fout:
		fout.write(md)

	return True

##### Script #####
if __name__ == '__main__':
	if len(sys.argv) < 2:
		usage()
		exit(0)

	# Get the source and destination directories.
	source_directory = os.path.dirname(os.path.realpath(__file__))
	destination_directory = os.path.dirname(sys.argv[2])
	report_file = os.path.realpath(sys.argv[1])
	destination_file = os.path.realpath(sys.argv[2])

	if not os.path.exists(report_file):
		print('ERROR: The report file {} does not exists.'.format(report_file))
		usage()
		exit(1)

	# Create the destination directory if not exists.
	if destination_directory and not os.path.isdir(destination_directory):
		os.makedirs(destination_directory)

	# Copy files from html_resources.
	resource_files = glob.glob(os.sep.join([source_directory, 'html_resources/*']))
	for rs in resource_files:
		print(rs)
		if os.path.isfile(rs):
			shutil.copy(rs, destination_directory)
		else:
			dirname = os.path.split(rs)[-1]
			target = os.sep.join([destination_directory, dirname])

			if os.path.exists(target):
				shutil.rmtree(target)
			shutil.copytree(rs, os.sep.join([destination_directory, dirname]))

	# Generate html.
	print('Start generation:')
	print('  input  : {}'.format(report_file))
	print('  output : {}'.format(destination_file))
	if generate_md(report_file, destination_file):
		print('Markdown was generated successful.')
